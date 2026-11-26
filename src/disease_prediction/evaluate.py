"""Evaluation, split sensitivity analysis, and plotting for Disease Prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from disease_prediction.config import load_config  # noqa: E402
from disease_prediction.data import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_breast_cancer_data,
    validate_medical_features,
)
from disease_prediction.preprocessing import build_preprocessor  # noqa: E402
from disease_prediction.train import build_candidate_models, compute_binary_metrics  # noqa: E402


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame | None:
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return None
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _save_plots(
    y_test: np.ndarray,
    test_probs_by_model: dict[str, np.ndarray],
    best_model_name: str,
    best_pipeline: Pipeline,
    test_metrics_by_model: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison
    df_comp = pd.DataFrame(test_metrics_by_model).T[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]
    ax = df_comp.plot(kind="bar", figsize=(9.5, 5), rot=0)
    ax.set_title("Test Set Performance Comparison — Breast Cancer Wisconsin")
    ax.set_ylabel("Score")
    ax.set_ylim(0.8, 1.02)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=150)
    plt.close()

    # 2. Confusion matrix
    best_preds = (test_probs_by_model[best_model_name] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["Benign (0)", "Malignant (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_model_name}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, probs in test_probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = test_metrics_by_model[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()

    # 4. Feature importance
    df_imp = _extract_feature_importance(best_pipeline, top_n=15)
    if df_imp is not None and not df_imp.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        df_plot = df_imp.iloc[::-1]
        ax.barh(df_plot["feature"], df_plot["importance"], color="teal")
        ax.set_xlabel("Importance / |Coefficient|")
        ax.set_title(f"Top 15 Features — {best_model_name}")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png", dpi=150)
        plt.close()


def evaluate_bundle(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    figures_dir: str | Path = "reports/figures",
    report_path: str | Path = "reports/model_evaluation.md",
    metrics_json_path: str | Path = "reports/metrics.json",
) -> dict[str, Any]:
    """Evaluate trained bundle and candidates on the isolated test split."""
    meta = bundle["metadata"]
    test_size = float(meta["test_size"])
    random_state = int(meta["random_state"])

    X = validate_medical_features(df[FEATURE_COLUMNS])
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    candidates = build_candidate_models(
        random_state=random_state, scale_pos_weight=scale_pos_weight
    )

    test_metrics_by_model: dict[str, dict[str, float]] = {}
    test_probs_by_model: dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        if name == bundle["model_name"]:
            pipe = bundle["model"]
        else:
            pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
            )
            pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        test_probs_by_model[name] = probs
        test_metrics_by_model[name] = compute_binary_metrics(y_test, probs, threshold=0.5)

    best_name = bundle["model_name"]
    best_probs = test_probs_by_model[best_name]
    best_preds = (best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    false_negatives = int(cm[1, 0])
    cls_report = classification_report(
        y_test,
        best_preds,
        target_names=["Benign (0)", "Malignant (1)"],
        zero_division=0,
    )

    # Split sensitivity across 5 seeds for the selected model architecture
    seed_recalls: list[float] = []
    seed_aucs: list[float] = []
    for seed in (7, 21, 42, 84, 123):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        est = build_candidate_models(random_state=seed, scale_pos_weight=scale_pos_weight)[
            best_name
        ]
        p = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", est)])
        p.fit(X_tr, y_tr)
        pr = p.predict_proba(X_te)[:, 1]
        m = compute_binary_metrics(y_te, pr)
        seed_recalls.append(m["recall"])
        seed_aucs.append(m["roc_auc"])

    _save_plots(
        y_test=y_test,
        test_probs_by_model=test_probs_by_model,
        best_model_name=best_name,
        best_pipeline=bundle["model"],
        test_metrics_by_model=test_metrics_by_model,
        figures_dir=Path(figures_dir),
    )

    payload = {
        "selected_model": best_name,
        "cv_results": bundle["cv_results"],
        "test_metrics": test_metrics_by_model,
        "confusion_matrix": cm.tolist(),
        "false_negatives_malignant": false_negatives,
        "classification_report": cls_report,
        "split_sensitivity": {
            "seeds": [7, 21, 42, 84, 123],
            "recall_mean": float(np.mean(seed_recalls)),
            "recall_std": float(np.std(seed_recalls)),
            "roc_auc_mean": float(np.mean(seed_aucs)),
            "roc_auc_std": float(np.std(seed_aucs)),
        },
    }

    metrics_out = Path(metrics_json_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_test = test_metrics_by_model[best_name]
    sens = payload["split_sensitivity"]
    lines = [
        "# Disease Prediction (Breast Cancer Wisconsin) — Evaluation Report",
        "",
        "> **Medical Disclaimer**: This model is an educational ML demonstration trained on "
        "the UCI Breast Cancer Wisconsin dataset. It is **not** clinically validated and must "
        "**never** be used as a medical diagnosis or treatment tool.",
        "",
        f"- **Selected Model**: `{best_name}` (chosen via 5-fold CV `{meta['selection_metric']}`)",
        f"- **Train / Test Split**: {meta['n_train']} / {meta['n_test']} "
        f"(stratified, seed={random_state})",
        f"- **Test False Negatives (Missed Malignant)**: `{false_negatives}`",
        "",
        "## 1. Cross-Validation Comparison (Training Set Only)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in bundle["cv_results"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Held-Out Test Set Comparison",
            "",
            "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, m in test_metrics_by_model.items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Selected Model ({best_name}) Details & Split Sensitivity",
            "",
            f"- **Test Accuracy**: {best_test['accuracy']:.4f}",
            f"- **Test Recall (Malignant)**: {best_test['recall']:.4f}",
            f"- **Test ROC-AUC**: {best_test['roc_auc']:.4f}",
            f"- **5-Seed Split Sensitivity (Recall)**: "
            f"{sens['recall_mean']:.4f} +/- {sens['recall_std']:.4f}",
            f"- **5-Seed Split Sensitivity (ROC-AUC)**: "
            f"{sens['roc_auc_mean']:.4f} +/- {sens['roc_auc_std']:.4f}",
            "",
            "### Classification Report",
            "",
            "```text",
            cls_report.strip(),
            "```",
            "",
            "## 4. Limitations & Bias Notes",
            "",
            "- **Class Imbalance**: 62.7% Benign vs. 37.3% Malignant; balanced class weights are "
            "applied during training.",
            "- **Cohort Bias**: Single-center 1995 FNA cohort (n=569) without demographic or "
            "multi-site validation.",
            "- **False Negatives**: In oncology screening, false negatives carry severe risk; "
            "models cannot replace pathologist review.",
        ]
    )
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for evaluating the saved disease prediction model."""
    parser = argparse.ArgumentParser(description="Evaluate Disease Prediction Model.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        model_path = Path(cfg["training"]["model_output_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run `python -m disease_prediction.train` first."
            )
        bundle = joblib.load(model_path)
        df = load_breast_cancer_data(cfg["dataset"]["local_csv_path"])
        payload = evaluate_bundle(
            bundle=bundle,
            df=df,
            figures_dir=cfg["evaluation"]["figures_dir"],
            report_path=cfg["evaluation"]["report_path"],
            metrics_json_path=cfg["evaluation"]["metrics_json_path"],
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    best = payload["selected_model"]
    print(f"Evaluated {best}: {json.dumps(payload['test_metrics'][best], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
