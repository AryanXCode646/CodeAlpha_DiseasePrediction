"""Training workflow for Breast Cancer Wisconsin Disease Prediction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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
    load_breast_cancer_data,
    validate_medical_features,
)
from disease_prediction.preprocessing import build_preprocessor


def build_candidate_models(random_state: int = 42, scale_pos_weight: float = 1.0) -> dict[str, Any]:
    """Create candidate classification models for disease prediction."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compute_binary_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics with Malignant (1) as positive class."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_and_select_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    selection_metric: str = "roc_auc",
) -> dict[str, Any]:
    """Run stratified train/test split, CV comparison, selection, and final training."""
    X = validate_medical_features(df[FEATURE_COLUMNS])
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    candidates = build_candidate_models(
        random_state=random_state, scale_pos_weight=scale_pos_weight
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results: dict[str, dict[str, float]] = {}
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
        cv_results[name] = compute_binary_metrics(y_train, oof_probs, threshold=0.5)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    best_model_name = max(cv_results, key=lambda k: cv_results[k][selection_metric])
    best_pipeline = fitted_pipelines[best_model_name]

    bundle = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "threshold": 0.5,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {0: "Benign", 1: "Malignant"},
        "cv_results": cv_results,
        "metadata": {
            "dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "selection_metric": selection_metric,
            "sklearn_version": sklearn.__version__,
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
        )
        model_path = save_bundle(result["bundle"], cfg["training"]["model_output_path"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    bundle = result["bundle"]
    print(
        f"Selected best model (by CV {bundle['metadata']['selection_metric']}): "
        f"{bundle['model_name']}"
    )
    print(json.dumps(bundle["cv_results"], indent=2))
    print(f"Saved model bundle to: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
