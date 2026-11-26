"""Inference interface and bundle contract validation for Disease Prediction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from disease_prediction.data import (
    FEATURE_COLUMNS,
    load_breast_cancer_data,
    validate_medical_features,
)

SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
REQUIRED_BUNDLE_KEYS = {
    "schema_version",
    "model",
    "model_name",
    "feature_columns",
    "classes",
    "target_mapping",
    "threshold",
}


def validate_model_bundle(bundle: object) -> dict[str, Any]:
    """Validate the serialized model bundle contract before inference."""
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle: expected a dictionary.")

    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle.keys()))
    if missing:
        raise ValueError(f"Invalid model bundle: missing required keys {missing}.")

    schema_version = bundle["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported bundle schema_version {schema_version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "Model bundle feature_columns do not match canonical Breast Cancer Wisconsin schema."
        )

    threshold = bundle["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Invalid threshold {threshold!r}; expected float in (0, 1).")
    if not math.isfinite(float(threshold)) or not (0.0 < float(threshold) < 1.0):
        raise ValueError(f"Threshold {threshold} out of valid range (0.0, 1.0).")

    model = bundle["model"]
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise ValueError("Model in bundle does not implement predict_proba().")
    if not hasattr(model, "classes_"):
        raise ValueError("Model in bundle does not expose classes_ attribute.")

    model_classes = [int(c) for c in model.classes_]
    if set(model_classes) != {0, 1} or len(model_classes) != 2:
        raise ValueError(
            f"Unexpected model.classes_ {model_classes}; expected binary classes {{0, 1}}."
        )

    target_mapping = bundle["target_mapping"]
    if not isinstance(target_mapping, dict) or set(target_mapping.keys()) != {0, 1}:
        raise ValueError(f"Invalid target_mapping {target_mapping!r}; expected keys {{0, 1}}.")

    return bundle


def load_model_bundle(
    model_path: str | Path = "models/disease_prediction_bundle.joblib",
) -> dict[str, Any]:
    """Load and validate a saved disease prediction model bundle.

    Security Note:
        Joblib uses pickle serialization. Only load model bundles from trusted
        sources created by this project's training pipeline.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. Run `python -m disease_prediction.train` first."
        )
    bundle = joblib.load(path)
    return validate_model_bundle(bundle)


def predict_disease(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict Benign (0) vs Malignant (1) class and uncalibrated model scores."""
    validated_bundle = validate_model_bundle(bundle)

    if isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        raise ValueError("Input records must be a dict, list of dicts, or DataFrame.")

    validated = validate_medical_features(df)
    model = validated_bundle["model"]
    thr = float(validated_bundle["threshold"])
    mapping = validated_bundle["target_mapping"]

    model_classes = [int(c) for c in model.classes_]
    idx_benign = model_classes.index(0)
    idx_malignant = model_classes.index(1)

    probs = model.predict_proba(validated)
    results: list[dict[str, Any]] = []
    for row_prob in probs:
        score_benign = float(row_prob[idx_benign])
        score_malig = float(row_prob[idx_malignant])
        pred_class = int(score_malig >= thr)
        results.append(
            {
                "predicted_class": pred_class,
                "predicted_label": mapping[pred_class],
                "score_malignant": round(score_malig, 4),
                "score_benign": round(score_benign, 4),
                # Backward-compatible aliases
                "probability_malignant": round(score_malig, 4),
                "probability_benign": round(score_benign, 4),
                "threshold_used": thr,
                "model_name": validated_bundle["model_name"],
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for disease prediction inference."""
    parser = argparse.ArgumentParser(description="Predict Breast Cancer class from features.")
    parser.add_argument(
        "--model",
        default="models/disease_prediction_bundle.joblib",
        help="Path to saved model bundle.",
    )
    parser.add_argument("--input-json", help="Path to JSON file with feature dict(s).")
    parser.add_argument("--sample", action="store_true", help="Run on first dataset sample.")
    args = parser.parse_args(argv)

    try:
        bundle = load_model_bundle(args.model)
        if args.input_json:
            payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        elif args.sample:
            df = load_breast_cancer_data()
            payload = df[FEATURE_COLUMNS].iloc[0].to_dict()
        else:
            raise ValueError("Provide either --input-json <path> or --sample.")
        predictions = predict_disease(payload, bundle)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(predictions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
