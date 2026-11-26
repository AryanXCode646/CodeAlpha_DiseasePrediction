"""Tests for strict data loading and input validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from disease_prediction.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_breast_cancer_data,
    validate_medical_features,
    validate_target_series,
)


def _sample_feature_df(n_rows: int = 3) -> pd.DataFrame:
    df = load_breast_cancer_data()
    return df[FEATURE_COLUMNS].iloc[:n_rows].copy().reset_index(drop=True)


def test_load_breast_cancer_data_schema_and_mapping() -> None:
    df = load_breast_cancer_data()
    assert df.shape == (569, 31)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert int((df[TARGET_COLUMN] == 1).sum()) == 212
    assert int((df[TARGET_COLUMN] == 0).sum()) == 357


def test_validate_reorders_columns_and_preserves_nan() -> None:
    df = _sample_feature_df(4)
    reversed_cols = list(reversed(FEATURE_COLUMNS))
    reordered = df[reversed_cols].copy()
    reordered.loc[0, "mean radius"] = np.nan

    validated = validate_medical_features(reordered)
    assert list(validated.columns) == FEATURE_COLUMNS
    assert np.isnan(validated.loc[0, "mean radius"])


@pytest.mark.parametrize(
    "bad_val",
    ["not_a_number", "", True, float("inf"), float("-inf"), -0.01],
)
def test_validate_rejects_invalid_values(bad_val: object) -> None:
    df = _sample_feature_df(2).astype(object)
    df.loc[1, "mean area"] = bad_val
    with pytest.raises(ValueError, match="mean area"):
        validate_medical_features(df)


def test_validate_rejects_empty_and_all_nan_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_medical_features(pd.DataFrame(columns=FEATURE_COLUMNS))

    df = _sample_feature_df(2)
    df.iloc[1] = np.nan
    with pytest.raises(ValueError, match="no usable"):
        validate_medical_features(df)


def test_validate_rejects_missing_extra_and_duplicate_columns() -> None:
    df = _sample_feature_df(2)
    with pytest.raises(ValueError, match="Missing required"):
        validate_medical_features(df.drop(columns=["mean radius"]))

    extra_df = df.copy()
    extra_df["unexpected_col"] = 1.0
    with pytest.raises(ValueError, match="Unexpected feature columns"):
        validate_medical_features(extra_df)

    dup_df = pd.concat([df, df[["mean radius"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate feature columns"):
        validate_medical_features(dup_df)


@pytest.mark.parametrize(
    "bad_targets",
    [
        pd.Series([0, 1, np.nan]),
        pd.Series([0, 1, 2]),
        pd.Series([0, 1, -1]),
        pd.Series([0, 1, "malignant"]),
        pd.Series([0, 0, 0]),
    ],
)
def test_validate_target_series_strict(bad_targets: pd.Series) -> None:
    with pytest.raises(ValueError):
        validate_target_series(bad_targets, require_both_classes=True)
