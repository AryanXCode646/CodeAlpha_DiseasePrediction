"""Dataset loading, preparation, and validation for Breast Cancer Wisconsin data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

TARGET_COLUMN = "is_malignant"


def get_feature_names() -> list[str]:
    """Return canonical list of 30 Breast Cancer Wisconsin feature names."""
    bunch = load_breast_cancer()
    return [str(name) for name in bunch.feature_names]


FEATURE_COLUMNS: list[str] = get_feature_names()


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

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def validate_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and order medical feature columns for training or inference."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    validated = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")

    if (validated.dropna() < 0).any().any():
        raise ValueError("Cell-nuclei measurements cannot be negative.")
    return validated
