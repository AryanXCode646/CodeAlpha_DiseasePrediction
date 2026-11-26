"""Educational Streamlit demonstration for Breast Cancer Wisconsin ML model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from disease_prediction.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_breast_cancer_data,
)
from disease_prediction.predict import load_model_bundle, predict_disease

MEDICAL_DISCLAIMER = (
    "This application is an educational machine learning demonstration. "
    "It is not clinically validated and must never be used for diagnosis, "
    "screening, treatment decisions, or personal medical advice."
)


@st.cache_resource
def load_cached_bundle(model_path: str = "models/disease_prediction_bundle.joblib") -> dict:
    """Load and validate model bundle once per Streamlit session."""
    return load_model_bundle(model_path)


@st.cache_data
def load_dataset_examples() -> pd.DataFrame:
    """Load public Breast Cancer Wisconsin examples without patient identifiers."""
    return load_breast_cancer_data()


def format_example_label(idx: int, df: pd.DataFrame) -> str:
    """Format a non-identifiable dataset example label for the selector."""
    label = "Malignant" if int(df.loc[idx, TARGET_COLUMN]) == 1 else "Benign"
    radius = float(df.loc[idx, "mean radius"])
    return f"Dataset Example #{idx} (Ground Truth: {label}, mean radius={radius:.2f})"


def render_prediction_results(res: dict[str, object], ground_truth: str | None = None) -> None:
    """Render predicted class, uncalibrated model scores, threshold, and limitations."""
    st.subheader(f"Predicted Class: {res['predicted_label']}")
    if ground_truth is not None:
        st.caption(f"Public Dataset Ground Truth: **{ground_truth}**")

    col1, col2 = st.columns(2)
    col1.metric("Model Score — Malignant", f"{float(res['score_malignant']):.1%}")
    col2.metric("Model Score — Benign", f"{float(res['score_benign']):.1%}")

    st.info(
        f"Model: `{res['model_name']}` | Decision Threshold: `{float(res['threshold_used']):.2f}` "
        "| Note: Output scores are uncalibrated classification scores, not clinical risk."
    )


def main() -> None:
    """Render the educational Breast Cancer Wisconsin Streamlit application."""
    st.set_page_config(page_title="Disease Prediction — CodeAlpha", layout="centered")
    st.title("Breast Cancer Wisconsin — Educational ML Demo")

    # Section 2: Permanent Medical Disclaimer (visible before any predictions)
    st.error(f"**MEDICAL DISCLAIMER**: {MEDICAL_DISCLAIMER}")

    # Section 1: Introduction
    with st.expander("About This Educational Project", expanded=True):
        st.markdown(
            "- **What it does**: Demonstrates a supervised binary classification pipeline "
            "predicting **Benign (0)** vs. **Malignant (1)** cell-nuclei profiles.\n"
            "- **Dataset**: UCI Breast Cancer Wisconsin (Diagnostic), 569 samples from digitized "
            "fine-needle aspirate (FNA) microscopy images.\n"
            "- **Features**: 30 quantitative nuclear morphology measurements "
            "(mean, standard error, and worst values for radius, texture, "
            "perimeter, area, smoothness, etc.)."
        )

    try:
        bundle = load_cached_bundle()
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"Model artifact unavailable: {exc}")
        st.stop()

    df_examples = load_dataset_examples()

    # Section 3: Public Dataset Example Demonstration
    st.header("1. Public Dataset Example Demonstration")
    st.caption(
        "Select a pre-loaded example from the public UCI Breast Cancer Wisconsin dataset "
        "(no patient-identifiable information is present)."
    )
    example_indices = [0, 19, 20, 21, 1, 2, 3, 5]
    selected_idx = st.selectbox(
        "Choose an educational dataset example",
        options=example_indices,
        format_func=lambda idx: format_example_label(idx, df_examples),
    )

    if st.button("Run Prediction on Selected Example", type="primary"):
        row_features = df_examples.loc[selected_idx, FEATURE_COLUMNS].to_dict()
        gt_label = (
            "Malignant" if int(df_examples.loc[selected_idx, TARGET_COLUMN]) == 1 else "Benign"
        )
        try:
            res = predict_disease(row_features, bundle)[0]
            render_prediction_results(res, ground_truth=gt_label)
        except ValueError as exc:
            st.error(f"Validation error: {exc}")

    # Section 4: Advanced Experimental Feature Exploration
    with st.expander("2. Advanced Experimental Feature Exploration (FNA Microscopy Inputs)"):
        st.warning(
            "Experimental Mode Only: These 30 inputs represent digitized FNA image measurements "
            "and cannot be substituted with ordinary patient symptoms or blood tests."
        )
        defaults = {col: float(df_examples[col].median()) for col in FEATURE_COLUMNS}
        with st.form("experimental_fna_form"):
            cols = st.columns(3)
            manual_inputs: dict[str, float] = {}
            for idx, feat in enumerate(FEATURE_COLUMNS):
                with cols[idx % 3]:
                    manual_inputs[feat] = st.number_input(
                        feat,
                        min_value=0.0,
                        value=round(defaults[feat], 4),
                        format="%.4f",
                    )
            submitted = st.form_submit_button("Run Experimental Feature Prediction")

        if submitted:
            try:
                res = predict_disease(manual_inputs, bundle)[0]
                render_prediction_results(res)
            except ValueError as exc:
                st.error(f"Invalid experimental input: {exc}")


if __name__ == "__main__":
    main()
