"""Tests for bundle contract validation, inference safety, and Streamlit app helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app as disease_app
from disease_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN
from disease_prediction.predict import (
    load_model_bundle,
    predict_disease,
    validate_model_bundle,
)
from disease_prediction.train import save_bundle, train_and_select_model


def _make_bundle(tmp_path: Path) -> tuple[dict, pd.DataFrame, Path]:
    rng = np.random.default_rng(42)
    n_rows = 60
    labels = np.array([i % 2 for i in range(n_rows)], dtype=int)
    data = {col: rng.uniform(5.0, 15.0, size=n_rows) + labels * 6.0 for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = labels
    df = pd.DataFrame(data)
    res = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)
    path = save_bundle(res["bundle"], tmp_path / "bundle.joblib")
    return res["bundle"], df, path


class _FlippedClassesModel:
    """Mock estimator with reversed classes_ [1, 0] to verify dynamic column lookup."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Column 0 corresponds to class 1 (Malignant=0.85), Column 1 to class 0 (Benign=0.15)
        return np.tile([0.85, 0.15], (len(X), 1))


def test_predict_single_and_batch_and_reload(tmp_path: Path) -> None:
    _, df, path = _make_bundle(tmp_path)
    loaded = load_model_bundle(path)

    single = predict_disease(df[FEATURE_COLUMNS].iloc[0].to_dict(), loaded)
    assert len(single) == 1
    assert single[0]["predicted_class"] in (0, 1)
    assert pytest.approx(1.0, abs=1e-3) == (
        single[0]["score_malignant"] + single[0]["score_benign"]
    )

    batch = predict_disease(df[FEATURE_COLUMNS].iloc[:5], loaded)
    assert len(batch) == 5


def test_predict_respects_reversed_classes_ordering(tmp_path: Path) -> None:
    bundle, df, _ = _make_bundle(tmp_path)
    bundle = dict(bundle)
    bundle["model"] = _FlippedClassesModel()
    bundle["threshold"] = 0.50

    res = predict_disease(df[FEATURE_COLUMNS].iloc[0].to_dict(), bundle)[0]
    assert res["score_malignant"] == pytest.approx(0.85)
    assert res["score_benign"] == pytest.approx(0.15)
    assert res["predicted_class"] == 1


def test_validate_model_bundle_rejects_corrupted_artifacts(tmp_path: Path) -> None:
    bundle, _, _ = _make_bundle(tmp_path)

    with pytest.raises(ValueError, match="dictionary"):
        validate_model_bundle(["not", "a", "dict"])

    missing_keys = dict(bundle)
    del missing_keys["threshold"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_model_bundle(missing_keys)

    bad_ver = dict(bundle, schema_version="99.0.0")
    with pytest.raises(ValueError, match="schema_version"):
        validate_model_bundle(bad_ver)

    bad_thr = dict(bundle, threshold=1.0)
    with pytest.raises(ValueError, match="Threshold"):
        validate_model_bundle(bad_thr)

    bad_cols = dict(bundle, feature_columns=["wrong"])
    with pytest.raises(ValueError, match="feature_columns"):
        validate_model_bundle(bad_cols)


def test_app_helpers_and_disclaimer(tmp_path: Path) -> None:
    assert "not clinically validated" in disease_app.MEDICAL_DISCLAIMER.lower()
    df_examples = disease_app.load_dataset_examples()
    assert len(df_examples) == 569
    label = disease_app.format_example_label(0, df_examples)
    assert "Dataset Example #0" in label

    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "missing.joblib")
