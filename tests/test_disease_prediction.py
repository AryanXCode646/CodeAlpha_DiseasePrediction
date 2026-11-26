"""Unit tests for Disease Prediction data, preprocessing, training, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disease_prediction.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_breast_cancer_data,
    validate_medical_features,
)
from disease_prediction.predict import load_model_bundle, predict_disease
from disease_prediction.preprocessing import build_preprocessor
from disease_prediction.train import (
    compute_binary_metrics,
    save_bundle,
    train_and_select_model,
)


def _make_synthetic_medical_df(n_rows: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    labels = np.array([i % 2 for i in range(n_rows)], dtype=int)
    for col in FEATURE_COLUMNS:
        base = rng.uniform(5.0, 20.0, size=n_rows)
        data[col] = base + labels * 8.0
    data[TARGET_COLUMN] = labels
    return pd.DataFrame(data)


def test_load_breast_cancer_data_and_validation() -> None:
    df = load_breast_cancer_data()
    assert df.shape == (569, 31)
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}

    validated = validate_medical_features(df[FEATURE_COLUMNS])
    assert validated.shape == (569, 30)


def test_validate_rejects_missing_or_negative() -> None:
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_medical_features(pd.DataFrame([{"mean radius": 12.0}]))

    df = _make_synthetic_medical_df(n_rows=5)[FEATURE_COLUMNS]
    df.loc[0, FEATURE_COLUMNS[0]] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_medical_features(df)


def test_preprocessing_imputes_missing_values() -> None:
    df = _make_synthetic_medical_df(n_rows=12)[FEATURE_COLUMNS]
    df.loc[0, FEATURE_COLUMNS[0]] = np.nan
    transformed = build_preprocessor().fit_transform(df)
    assert transformed.shape == (12, 30)
    assert not np.isnan(transformed).any()


def test_training_reproducibility_serialization_and_predict(tmp_path: Path) -> None:
    df = _make_synthetic_medical_df(n_rows=60, seed=9)
    res1 = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)
    res2 = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)

    assert res1["bundle"]["model_name"] == res2["bundle"]["model_name"]
    assert res1["bundle"]["cv_results"] == res2["bundle"]["cv_results"]

    bundle_path = save_bundle(res1["bundle"], tmp_path / "disease.joblib")
    loaded = load_model_bundle(bundle_path)

    sample = df[FEATURE_COLUMNS].iloc[0].to_dict()
    preds = predict_disease(sample, loaded)
    assert len(preds) == 1
    assert preds[0]["predicted_class"] in (0, 1)
    assert pytest.approx(1.0, abs=1e-3) == (
        preds[0]["probability_malignant"] + preds[0]["probability_benign"]
    )


def test_compute_binary_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
