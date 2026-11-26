"""Training workflow for Breast Cancer Wisconsin Disease Prediction."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from disease_prediction.config import load_config
from disease_prediction.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_dataset_fingerprint,
    load_breast_cancer_data,
    validate_medical_features,
    validate_target_series,
)
from disease_prediction.preprocessing import build_preprocessor

BUNDLE_SCHEMA_VERSION = "1.0.0"


class FoldWeightedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Clone-compatible XGBoost wrapper computing scale_pos_weight inside each fit."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FoldWeightedXGBClassifier":
        """Compute scale_pos_weight strictly from the provided training fold labels."""
        y_arr = np.asarray(y, dtype=int)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        self.scale_pos_weight_ = float(n_neg / max(n_pos, 1))
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight_,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, y_arr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose underlying XGBoost feature importances."""
        return self.estimator_.feature_importances_


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Create candidate classification models for disease prediction."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                random_state=random_state,
            ),
            cv=3,
            method="sigmoid",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": FoldWeightedXGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compute_binary_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics and Brier score for Malignant (1)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def select_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select training-only threshold maximizing F2-score on OOF probabilities.

    Note: F2 optimization is an educational statistical objective weighting recall
    above precision, not a clinically validated decision threshold.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thr = 0.50
    best_f2 = -1.0
    for thr in thresholds:
        score = float(fbeta_score(y_true, (y_prob >= thr).astype(int), beta=2.0, zero_division=0))
        if score > best_f2:
            best_f2 = score
            best_thr = float(thr)
    return round(best_thr, 2)


def compute_threshold_tradeoff(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    """Compute precision/recall/F1/F2 trade-off across candidate thresholds."""
    rows: list[dict[str, float]] = []
    for thr in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        y_pred = (y_prob >= thr).astype(int)
        rows.append(
            {
                "threshold": thr,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "f2_score": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
            }
        )
    return rows


def train_and_select_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    selection_metric: str = "roc_auc",
    threshold_strategy: str = "f2_oof",
) -> dict[str, Any]:
    """Run stratified train/test split, CV comparison, selection, and final training."""
    if threshold_strategy not in ("default", "f2_oof"):
        raise ValueError(
            f"Unsupported threshold_strategy {threshold_strategy!r}; "
            "expected 'default' or 'f2_oof'."
        )

    X = validate_medical_features(df[FEATURE_COLUMNS])
    y = validate_target_series(df[TARGET_COLUMN], require_both_classes=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    candidates = build_candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results: dict[str, dict[str, Any]] = {}
    oof_probs_by_model: dict[str, np.ndarray] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        oof_probs = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        oof_probs_by_model[name] = oof_probs

        f2_thr = select_f2_threshold(y_train, oof_probs)
        metrics_default = compute_binary_metrics(y_train, oof_probs, threshold=0.5)
        metrics_f2 = compute_binary_metrics(y_train, oof_probs, threshold=f2_thr)
        cv_results[name] = {
            **metrics_default,
            "f2_optimal_threshold": f2_thr,
            "f2_threshold_metrics": metrics_f2,
        }

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    best_model_name = max(cv_results, key=lambda k: cv_results[k][selection_metric])
    best_pipeline = fitted_pipelines[best_model_name]
    selected_threshold = (
        float(cv_results[best_model_name]["f2_optimal_threshold"])
        if threshold_strategy == "f2_oof"
        else 0.50
    )
    oof_tradeoff = compute_threshold_tradeoff(y_train, oof_probs_by_model[best_model_name])

    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model": best_pipeline,
        "model_name": best_model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": best_pipeline.classes_.tolist(),
        "target_mapping": {0: "Benign", 1: "Malignant"},
        "threshold": selected_threshold,
        "threshold_strategy": threshold_strategy,
        "cv_results": cv_results,
        "oof_threshold_tradeoff": oof_tradeoff,
        "metadata": {
            "dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "dataset_fingerprint": compute_dataset_fingerprint(
                df[FEATURE_COLUMNS + [TARGET_COLUMN]]
            ),
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "selection_metric": selection_metric,
            "training_config": {
                "test_size": test_size,
                "cv_folds": cv_folds,
                "random_state": random_state,
                "selection_metric": selection_metric,
                "threshold_strategy": threshold_strategy,
            },
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    return {
        "bundle": bundle,
        "fitted_pipelines": fitted_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def save_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Persist trained model bundle with joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training the disease prediction model."""
    parser = argparse.ArgumentParser(description="Train Disease Prediction Model.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        df = load_breast_cancer_data(cfg["dataset"]["local_csv_path"])
        result = train_and_select_model(
            df=df,
            test_size=cfg["dataset"]["test_size"],
            cv_folds=cfg["training"]["cv_folds"],
            random_state=cfg["training"]["random_state"],
            selection_metric=cfg["training"]["selection_metric"],
            threshold_strategy=cfg["training"].get("threshold_strategy", "f2_oof"),
        )
        model_path = save_bundle(result["bundle"], cfg["training"]["model_output_path"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    bundle = result["bundle"]
    print(
        f"Selected best model (by CV {bundle['metadata']['selection_metric']}): "
        f"{bundle['model_name']} | threshold={bundle['threshold']:.2f} "
        f"({bundle['threshold_strategy']})"
    )
    print(json.dumps(bundle["cv_results"], indent=2))
    print(f"Saved model bundle to: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
