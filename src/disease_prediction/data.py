"""Dataset loading, preparation, and strict validation for Breast Cancer Wisconsin data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

TARGET_COLUMN = "is_malignant"


def get_feature_names() -> list[str]:
    """Return canonical list of 30 Breast Cancer Wisconsin feature names."""
    bunch = load_breast_cancer()
    return [str(name) for name in bunch.feature_names]


FEATURE_COLUMNS: list[str] = get_feature_names()


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint of a DataFrame's schema and values."""
    ordered_cols = sorted(df.columns.tolist())
    canonical = df[ordered_cols].copy()
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    hasher = hashlib.sha256()
    hasher.update(",".join(ordered_cols).encode("utf-8"))
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def prepare_breast_cancer_csv(output_path: str | Path = "data/breast_cancer_wisconsin.csv") -> Path:
    """Export Breast Cancer Wisconsin dataset to CSV with explicit malignant=1 target."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn's raw target: 0 = malignant, 1 = benign.
    # Remap so positive class 1 = malignant (clinically critical detection target).
    df[TARGET_COLUMN] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def load_breast_cancer_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load Breast Cancer Wisconsin dataset from CSV or scikit-learn fallback."""
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at '{path}'. Run `python scripts/prepare_data.py` first."
            )
        df = pd.read_csv(path)
    else:
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        df[TARGET_COLUMN] = (df["target"] == 0).astype(int)
        df = df.drop(columns=["target"])

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()].tolist()))
        raise ValueError(f"Dataset contains duplicate columns: {dupes}")

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    validate_medical_features(df[FEATURE_COLUMNS])
    validate_target_series(df[TARGET_COLUMN], require_both_classes=True)
    return df


def _parse_strict_float(value: object, col_name: str, row_idx: object) -> float:
    """Parse a single cell strictly as float or NaN; reject strings, bools, and infinities."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, bool):
        raise ValueError(
            f"Invalid boolean value {value!r} in feature '{col_name}' at row {row_idx}."
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        num = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                f"Empty string in feature '{col_name}' at row {row_idx}; "
                "use explicit NaN/null for missing values."
            )
        try:
            num = float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value {value!r} in feature '{col_name}' at row {row_idx}."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported type {type(value).__name__} in feature '{col_name}' at row {row_idx}."
        )

    if np.isposinf(num):
        raise ValueError(f"Positive infinity (+inf) in feature '{col_name}' at row {row_idx}.")
    if np.isneginf(num):
        raise ValueError(f"Negative infinity (-inf) in feature '{col_name}' at row {row_idx}.")
    if num < 0.0:
        raise ValueError(
            f"Negative measurement ({num}) in feature '{col_name}' at row {row_idx}; "
            "cell-nuclei measurements must be >= 0."
        )
    return num


def validate_medical_features(
    df: pd.DataFrame,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """Strictly validate and canonically order medical feature columns.

    Policy:
        - Duplicate columns are rejected.
        - Missing required columns are rejected.
        - Unexpected extra columns are rejected unless ``allow_extra_columns=True``.
        - Non-numeric strings, booleans, +/-inf, and negative numbers are rejected.
        - Explicit NaN/None values are preserved for downstream Pipeline median imputation.
        - Rows where all 30 features are missing are rejected.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()].tolist()))
        raise ValueError(f"Duplicate feature columns detected: {dupes}")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if not allow_extra_columns:
        extra = [col for col in df.columns if col not in FEATURE_COLUMNS]
        if extra:
            raise ValueError(
                f"Unexpected feature columns not in schema: {extra}. "
                "Pass only the 30 canonical Breast Cancer Wisconsin features."
            )

    validated = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            raise ValueError(f"Feature '{col}' has boolean dtype; expected numeric.")
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=np.float64)
            if np.isposinf(arr).any():
                raise ValueError(f"Positive infinity (+inf) detected in feature '{col}'.")
            if np.isneginf(arr).any():
                raise ValueError(f"Negative infinity (-inf) detected in feature '{col}'.")
            finite_mask = ~np.isnan(arr)
            if (arr[finite_mask] < 0.0).any():
                raise ValueError(
                    f"Negative measurement detected in feature '{col}'; "
                    "cell-nuclei measurements must be >= 0."
                )
            validated[col] = arr
        else:
            parsed = [_parse_strict_float(val, col, idx) for idx, val in series.items()]
            validated[col] = np.array(parsed, dtype=np.float64)

    all_missing_rows = validated.isna().all(axis=1)
    if all_missing_rows.any():
        bad_indices = validated.index[all_missing_rows].tolist()
        raise ValueError(
            f"Rows {bad_indices} contain no usable (non-missing) feature measurements."
        )

    return validated[FEATURE_COLUMNS]


def validate_target_series(
    y: pd.Series | np.ndarray,
    require_both_classes: bool = True,
) -> np.ndarray:
    """Strictly validate binary target labels as {0, 1} integers."""
    series = pd.Series(y)
    if series.empty:
        raise ValueError("Target series must not be empty.")
    if series.isna().any():
        raise ValueError("Target labels contain missing (NaN/None) values.")
    if pd.api.types.is_bool_dtype(series):
        raise ValueError("Target labels must be integers {0, 1}, not booleans.")

    parsed: list[int] = []
    for idx, val in series.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"Invalid target value {val!r} at index {idx}; expected binary integer 0 or 1."
            )
        fval = float(val)
        if not np.isfinite(fval) or fval not in (0.0, 1.0):
            raise ValueError(
                f"Invalid target value {val!r} at index {idx}; expected binary integer 0 or 1."
            )
        parsed.append(int(fval))

    arr = np.array(parsed, dtype=int)
    if require_both_classes and len(np.unique(arr)) < 2:
        found = np.unique(arr).tolist()
        raise ValueError(f"Target labels must contain both classes {{0, 1}}, found only {found}.")
    return arr
