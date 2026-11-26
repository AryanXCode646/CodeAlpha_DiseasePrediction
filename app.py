"""Streamlit UI for educational Breast Cancer Disease Prediction."""

from __future__ import annotations

import streamlit as st

from disease_prediction.data import FEATURE_COLUMNS, load_breast_cancer_data
from disease_prediction.predict import load_model_bundle, predict_disease

st.set_page_config(page_title="Disease Prediction — CodeAlpha", layout="centered")
st.title("Breast Cancer Prediction (Educational Demo)")
st.error(
    "MEDICAL DISCLAIMER: This application is an educational ML demonstration trained on the "
    "UCI Breast Cancer Wisconsin dataset. It is NOT clinically validated and must NEVER be "
    "used for medical diagnosis, screening, or treatment decisions."
)


@st.cache_resource
def _get_bundle():
    return load_model_bundle("models/disease_prediction_bundle.joblib")


@st.cache_data
def _get_defaults() -> dict[str, float]:
    df = load_breast_cancer_data()
    return {col: float(df[col].median()) for col in FEATURE_COLUMNS}


try:
    bundle = _get_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

defaults = _get_defaults()

with st.form("medical_form"):
    st.caption("Adjust cell-nuclei measurements (initialized to dataset medians):")
    cols = st.columns(3)
    inputs: dict[str, float] = {}
    for idx, feat in enumerate(FEATURE_COLUMNS):
        with cols[idx % 3]:
            inputs[feat] = st.number_input(
                feat,
                min_value=0.0,
                value=round(defaults[feat], 4),
                format="%.4f",
            )
    submitted = st.form_submit_button("Run Educational Prediction")

if submitted:
    res = predict_disease(inputs, bundle)[0]
    st.subheader(f"Model Output: {res['predicted_label']}")
    col1, col2 = st.columns(2)
    col1.metric("Malignant Probability", f"{res['probability_malignant']:.1%}")
    col2.metric("Benign Probability", f"{res['probability_benign']:.1%}")
    st.caption(f"Model: `{res['model_name']}` (educational demonstration only)")
