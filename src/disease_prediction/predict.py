"""Inference interface for Breast Cancer Wisconsin Disease Prediction."""

from __future__ import annotations

import argparse
import json
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


def load_model_bundle(
    model_path: str | Path = "models/disease_prediction_bundle.joblib",
) -> dict[str, Any]:
    """Load a saved disease prediction model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. Run `python -m disease_prediction.train` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in '{path}'.")
    return bundle


def predict_disease(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict Benign (0) vs Malignant (1) class and probabilities."""
    if isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        raise ValueError("Input records must be a dict, list of dicts, or DataFrame.")

    validated = validate_medical_features(df)
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))
    mapping = bundle.get("target_mapping", {0: "Benign", 1: "Malignant"})

    probs = model.predict_proba(validated)
    results: list[dict[str, Any]] = []
    for row_prob in probs:
        prob_benign = float(row_prob[0])
        prob_malig = float(row_prob[1])
        pred_class = int(prob_malig >= thr)
        results.append(
            {
                "predicted_class": pred_class,
                "predicted_label": mapping[pred_class],
                "probability_malignant": round(prob_malig, 4),
                "probability_benign": round(prob_benign, 4),
                "model_name": bundle.get("model_name", "unknown"),
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
