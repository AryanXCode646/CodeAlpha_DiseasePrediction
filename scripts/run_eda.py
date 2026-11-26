"""Generate EDA plots and summary statistics for Breast Cancer Wisconsin data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from disease_prediction.config import load_config  # noqa: E402
from disease_prediction.data import TARGET_COLUMN, load_breast_cancer_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA on Breast Cancer Wisconsin dataset.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = load_breast_cancer_data(cfg["dataset"]["local_csv_path"])
    out_dir = Path(cfg["evaluation"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    ax.bar(["Benign (0)", "Malignant (1)"], counts.values, color=["#2b8a3e", "#c92a2a"])
    ax.set_title("Class Distribution — Breast Cancer Wisconsin")
    ax.set_ylabel("Count")
    for idx, val in enumerate(counts.values):
        ax.text(idx, val + 6, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    # 2. Key feature distributions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="mean radius", ax=axes[0])
    axes[0].set_xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    axes[0].set_title("Mean Radius by Diagnosis")

    sns.boxplot(data=df, x=TARGET_COLUMN, y="worst perimeter", ax=axes[1])
    axes[1].set_xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    axes[1].set_title("Worst Perimeter by Diagnosis")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_feature_distributions.png", dpi=150)
    plt.close()

    print(f"EDA complete: {len(df)} rows, {df.isna().sum().sum()} missing values.")
    print(f"Saved EDA figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
