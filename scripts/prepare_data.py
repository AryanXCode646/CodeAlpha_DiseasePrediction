"""Export Breast Cancer Wisconsin dataset to local CSV."""

from __future__ import annotations

import argparse

from disease_prediction.config import load_config
from disease_prediction.data import prepare_breast_cancer_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Breast Cancer Wisconsin CSV.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dest = prepare_breast_cancer_csv(cfg["dataset"]["local_csv_path"])
    print(f"Dataset saved to: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
