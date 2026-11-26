"""Tests for preprocessing, training, XGBoost weighting, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disease_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN
from disease_prediction.evaluate import evaluate_bundle, run_split_sensitivity
from disease_prediction.preprocessing import build_preprocessor
from disease_prediction.train import (
    FoldWeightedXGBClassifier,
    build_candidate_models,
    compute_binary_metrics,
    save_bundle,
    select_f2_threshold,
    train_and_select_model,
)


def _make_synthetic_medical_df(n_rows: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([1 if i % 4 == 0 else 0 for i in range(n_rows)], dtype=int)
    data = {col: rng.uniform(5.0, 15.0, size=n_rows) + labels * 6.0 for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)


def test_preprocessing_imputes_and_preserves_shape() -> None:
    df = _make_synthetic_medical_df(n_rows=12)[FEATURE_COLUMNS]
    df.loc[0, FEATURE_COLUMNS[0]] = np.nan
    transformed = build_preprocessor().fit_transform(df)
    assert transformed.shape == (12, 30)
    assert not np.isnan(transformed).any()


def test_fold_weighted_xgboost_computes_fold_specific_weight() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y_3_to_1 = np.array([0] * 15 + [1] * 5)
    clf = FoldWeightedXGBClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_3_to_1)
    assert clf.scale_pos_weight_ == pytest.approx(3.0)

    y_1_to_1 = np.array([0] * 10 + [1] * 10)
    clf.fit(X, y_1_to_1)
    assert clf.scale_pos_weight_ == pytest.approx(1.0)


def test_build_candidate_models_contains_all_four() -> None:
    candidates = build_candidate_models(random_state=42)
    assert set(candidates.keys()) == {
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "xgboost",
    }


def test_training_reproducibility_and_threshold_strategies() -> None:
    df = _make_synthetic_medical_df(n_rows=80, seed=7)
    res1 = train_and_select_model(
        df, test_size=0.2, cv_folds=3, random_state=42, threshold_strategy="f2_oof"
    )
    res2 = train_and_select_model(
        df, test_size=0.2, cv_folds=3, random_state=42, threshold_strategy="f2_oof"
    )
    assert res1["bundle"]["model_name"] == res2["bundle"]["model_name"]
    assert res1["bundle"]["threshold"] == res2["bundle"]["threshold"]

    res_default = train_and_select_model(
        df, test_size=0.2, cv_folds=3, random_state=42, threshold_strategy="default"
    )
    assert res_default["bundle"]["threshold"] == 0.50


def test_select_f2_threshold_and_binary_metrics() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
    thr = select_f2_threshold(y_true, y_prob)
    assert 0.1 <= thr <= 0.9

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["roc_auc"] == 1.0


def test_evaluate_bundle_generates_reports_and_figures(tmp_path: Path) -> None:
    df = _make_synthetic_medical_df(n_rows=80, seed=11)
    res = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)
    save_bundle(res["bundle"], tmp_path / "bundle.joblib")

    fig_dir = tmp_path / "figures"
    report_path = tmp_path / "eval.md"
    metrics_path = tmp_path / "metrics.json"
    payload = evaluate_bundle(
        bundle=res["bundle"],
        df=df,
        figures_dir=fig_dir,
        report_path=report_path,
        metrics_json_path=metrics_path,
    )

    assert report_path.exists()
    assert metrics_path.exists()
    assert (fig_dir / "model_comparison.png").exists()
    assert (fig_dir / "calibration_curve.png").exists()
    assert (fig_dir / "precision_recall_tradeoff.png").exists()
    assert "selected_model_test_default_0_5" in payload
    assert "selected_model_test_selected_threshold" in payload

    sens = run_split_sensitivity(
        df[FEATURE_COLUMNS], df[TARGET_COLUMN].to_numpy(), "logistic_regression"
    )
    assert sens["includes_primary_seed"] is True
    assert len(sens["per_seed_results"]) == 5
