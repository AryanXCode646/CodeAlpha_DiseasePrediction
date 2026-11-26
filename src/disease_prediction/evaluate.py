"""Evaluation, calibration analysis, split sensitivity, and plotting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
    validate_target_series,
)
from disease_prediction.predict import load_model_bundle, validate_model_bundle  # noqa: E402
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
    selected_threshold: float,
    best_pipeline: Pipeline,
    test_metrics_by_model: dict[str, dict[str, float]],
    oof_tradeoff: list[dict[str, float]],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Model comparison with honest 0.0 - 1.05 y-axis scale
    df_comp = pd.DataFrame(test_metrics_by_model).T[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    df_comp.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Held-Out Test Performance Comparison (Threshold = 0.50)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    comp_path = figures_dir / "model_comparison.png"
    fig.savefig(comp_path, dpi=150)
    plt.close(fig)
    saved.append(comp_path)

    # 2. Confusion matrix at selected threshold
    best_probs = test_probs_by_model[best_model_name]
    best_preds = (best_probs >= selected_threshold).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["Benign (0)", "Malignant (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_model_name} (thr={selected_threshold:.2f})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    saved.append(cm_path)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, probs in test_probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = test_metrics_by_model[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = figures_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    saved.append(roc_path)

    # 4. Calibration / reliability curve
    brier = test_metrics_by_model[best_model_name]["brier_score"]
    prob_true, prob_pred = calibration_curve(y_test, best_probs, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label=f"{best_model_name} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean Model Score (Uncalibrated)")
    ax.set_ylabel("Observed Fraction of Malignant Cases")
    ax.set_title(f"Reliability Curve — {best_model_name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    cal_path = figures_dir / "calibration_curve.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    saved.append(cal_path)

    # 5. OOF Precision-Recall Trade-off
    if oof_tradeoff:
        df_tr = pd.DataFrame(oof_tradeoff)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(df_tr["threshold"], df_tr["precision"], marker="o", label="Precision")
        ax.plot(df_tr["threshold"], df_tr["recall"], marker="s", label="Recall")
        ax.plot(df_tr["threshold"], df_tr["f2_score"], marker="^", label="F2-Score")
        ax.axvline(selected_threshold, color="red", linestyle="--", label="Selected Threshold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Decision Threshold (Training OOF)")
        ax.set_ylabel("Score")
        ax.set_title(f"Training-Only Threshold Trade-Off — {best_model_name}")
        ax.legend(loc="lower left")
        fig.tight_layout()
        tr_path = figures_dir / "precision_recall_tradeoff.png"
        fig.savefig(tr_path, dpi=150)
        plt.close(fig)
        saved.append(tr_path)

    # 6. Feature importance
    df_imp = _extract_feature_importance(best_pipeline, top_n=15)
    if df_imp is not None and not df_imp.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        df_plot = df_imp.iloc[::-1]
        ax.barh(df_plot["feature"], df_plot["importance"], color="teal")
        ax.set_xlabel("Importance / |Coefficient|")
        ax.set_title(f"Top 15 Features — {best_model_name}")
        fig.tight_layout()
        imp_path = figures_dir / "feature_importance.png"
        fig.savefig(imp_path, dpi=150)
        plt.close(fig)
        saved.append(imp_path)

    return saved


def run_split_sensitivity(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator_name: str,
    test_size: float = 0.2,
    seeds: tuple[int, ...] = (7, 21, 42, 84, 123),
) -> dict[str, Any]:
    """Evaluate estimator stability across independent stratified splits at threshold 0.50."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        est = build_candidate_models(random_state=seed)[estimator_name]
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", est)])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        m = compute_binary_metrics(y_te, probs, threshold=0.5)
        per_seed.append(
            {
                "seed": seed,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "train_malignant_rate": round(float(y_tr.mean()), 4),
                **m,
            }
        )

    recalls = [r["recall"] for r in per_seed]
    aucs = [r["roc_auc"] for r in per_seed]
    f1s = [r["f1_score"] for r in per_seed]
    return {
        "selected_estimator": estimator_name,
        "seeds": list(seeds),
        "includes_primary_seed": 42 in seeds,
        "per_seed_results": per_seed,
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
    }


def evaluate_bundle(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    figures_dir: str | Path = "reports/figures",
    report_path: str | Path = "reports/model_evaluation.md",
    metrics_json_path: str | Path = "reports/metrics.json",
) -> dict[str, Any]:
    """Evaluate trained bundle and candidates on the isolated test split."""
    validated_bundle = validate_model_bundle(bundle)
    meta = validated_bundle["metadata"]
    test_size = float(meta["test_size"])
    random_state = int(meta["random_state"])

    X = validate_medical_features(df[FEATURE_COLUMNS])
    y = validate_target_series(df[TARGET_COLUMN], require_both_classes=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    candidates = build_candidate_models(random_state=random_state)
    test_metrics_default: dict[str, dict[str, float]] = {}
    test_probs_by_model: dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        if name == validated_bundle["model_name"]:
            pipe = validated_bundle["model"]
        else:
            pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
            )
            pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        test_probs_by_model[name] = probs
        test_metrics_default[name] = compute_binary_metrics(y_test, probs, threshold=0.5)

    best_name = validated_bundle["model_name"]
    selected_thr = float(validated_bundle["threshold"])
    best_probs = test_probs_by_model[best_name]

    default_metrics = test_metrics_default[best_name]
    selected_metrics = compute_binary_metrics(y_test, best_probs, threshold=selected_thr)

    cm_default = confusion_matrix(y_test, (best_probs >= 0.5).astype(int))
    cm_selected = confusion_matrix(y_test, (best_probs >= selected_thr).astype(int))
    cls_report_selected = classification_report(
        y_test,
        (best_probs >= selected_thr).astype(int),
        target_names=["Benign (0)", "Malignant (1)"],
        zero_division=0,
    )

    sensitivity = run_split_sensitivity(X=X, y=y, estimator_name=best_name, test_size=test_size)

    _save_plots(
        y_test=y_test,
        test_probs_by_model=test_probs_by_model,
        best_model_name=best_name,
        selected_threshold=selected_thr,
        best_pipeline=validated_bundle["model"],
        test_metrics_by_model=test_metrics_default,
        oof_tradeoff=validated_bundle.get("oof_threshold_tradeoff", []),
        figures_dir=Path(figures_dir),
    )

    payload = {
        "selected_model": best_name,
        "selected_threshold": selected_thr,
        "threshold_strategy": validated_bundle.get("threshold_strategy", "unknown"),
        "cv_results": validated_bundle["cv_results"],
        "oof_threshold_tradeoff": validated_bundle.get("oof_threshold_tradeoff", []),
        "test_metrics_default_threshold": test_metrics_default,
        "selected_model_test_default_0_5": {
            **default_metrics,
            "confusion_matrix": cm_default.tolist(),
            "false_negatives": int(cm_default[1, 0]),
            "false_positives": int(cm_default[0, 1]),
        },
        "selected_model_test_selected_threshold": {
            **selected_metrics,
            "confusion_matrix": cm_selected.tolist(),
            "false_negatives": int(cm_selected[1, 0]),
            "false_positives": int(cm_selected[0, 1]),
        },
        "classification_report_selected": cls_report_selected,
        "split_sensitivity": sensitivity,
    }

    metrics_out = Path(metrics_json_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Disease Prediction (Breast Cancer Wisconsin) — Evaluation Report",
        "",
        "> **Medical Disclaimer**: This model is an educational ML demonstration trained on "
        "the UCI Breast Cancer Wisconsin dataset. It is **not** clinically validated and must "
        "**never** be used for diagnosis, screening, or treatment decisions.",
        "",
        f"- **Selected Model**: `{best_name}` (chosen via 5-fold CV `{meta['selection_metric']}`)",
        f"- **Threshold Strategy**: `{payload['threshold_strategy']}` "
        f"(training-only OOF threshold = `{selected_thr:.2f}`)",
        f"- **Train / Test Split**: {meta['n_train']} / {meta['n_test']} "
        f"(stratified, seed={random_state})",
        "",
        "## 1. Cross-Validation Comparison (Training Set Only, Threshold = 0.50)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | F2 Opt Thr |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in validated_bundle["cv_results"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} | "
            f"{m['f2_optimal_threshold']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Held-Out Test Set Comparison (Default Threshold = 0.50)",
            "",
            "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for name, m in test_metrics_default.items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |"
        )

    sel_def = payload["selected_model_test_default_0_5"]
    sel_thr = payload["selected_model_test_selected_threshold"]
    lines.extend(
        [
            "",
            f"## 3. Threshold Comparison on Held-Out Test Set ({best_name})",
            "",
            "| Threshold | Accuracy | Precision | Recall | F1-Score | F2-Score | FN | FP |",
            "|---|---|---|---|---|---|---|---|",
            f"| 0.50 (Default) | {sel_def['accuracy']:.4f} | {sel_def['precision']:.4f} | "
            f"{sel_def['recall']:.4f} | {sel_def['f1_score']:.4f} | "
            f"{sel_def['f2_score']:.4f} | {sel_def['false_negatives']} | "
            f"{sel_def['false_positives']} |",
            f"| {selected_thr:.2f} (OOF F2-Tuned) | {sel_thr['accuracy']:.4f} | "
            f"{sel_thr['precision']:.4f} | {sel_thr['recall']:.4f} | "
            f"{sel_thr['f1_score']:.4f} | {sel_thr['f2_score']:.4f} | "
            f"{sel_thr['false_negatives']} | {sel_thr['false_positives']} |",
            "",
            "Note: F2 optimization is an educational statistical objective weighting recall higher "
            "than precision on training OOF predictions, not a clinically validated threshold.",
            "",
            "### Classification Report (Selected Threshold)",
            "",
            "```text",
            cls_report_selected.strip(),
            "```",
            "",
            "## 4. Probability Calibration & Score Interpretation",
            "",
            f"- **Test Brier Score**: `{selected_metrics['brier_score']:.4f}`",
            "- **Class-Weighting Effect**: Balanced class weights (`class_weight='balanced'` / "
            "`scale_pos_weight`) shift predicted scores toward the minority Malignant class.",
            "- **Interpretation**: Output scores are uncalibrated model confidence scores for "
            "ranking/classification, **not** calibrated real-world clinical risk probabilities.",
            "",
            "## 5. Five-Seed Split Sensitivity (Threshold = 0.50)",
            "",
            f"- **Estimator**: `{sensitivity['selected_estimator']}`",
            f"- **Seeds**: `{sensitivity['seeds']}` (includes primary seed 42)",
            f"- **Recall**: {sensitivity['recall_mean']:.4f} +/- {sensitivity['recall_std']:.4f}",
            f"- **F1-Score**: {sensitivity['f1_mean']:.4f} +/- {sensitivity['f1_std']:.4f}",
            f"- **ROC-AUC**: {sensitivity['roc_auc_mean']:.4f} "
            f"+/- {sensitivity['roc_auc_std']:.4f}",
            "- Note: This measures variance across random 80/20 splits of the same 569-sample "
            "cohort, not external multi-center validation.",
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
        bundle = load_model_bundle(cfg["training"]["model_output_path"])
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
    print(
        f"Evaluated {best} (thr={payload['selected_threshold']:.2f}): "
        f"{json.dumps(payload['selected_model_test_selected_threshold'], indent=2)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
