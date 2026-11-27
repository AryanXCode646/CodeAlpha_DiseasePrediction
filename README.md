# Disease Prediction from Medical Data

An educational machine learning project for classifying breast mass fine-needle aspirate (FNA) nuclear morphology measurements as **Benign (0)** or **Malignant (1)** using multiple machine learning algorithms and a reproducible evaluation pipeline.

> **Medical Disclaimer:** This project is an educational machine learning benchmark trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset. It is **not clinically validated** and must **never** be used for diagnosis, screening, treatment decisions, or personal medical advice.

## 1. Problem Statement & Objectives

Classify breast mass fine-needle aspirate (FNA) nuclear morphology measurements as **Benign (`0`)** or **Malignant (`1`)**, prioritizing Recall and False-Negative analysis in a leakage-free, reproducible Python pipeline.

## 2. Dataset Provenance, Target Encoding & Licensing

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source & Attribution:** UCI Machine Learning Repository (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian), bundled via `sklearn.datasets.load_breast_cancer`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Size & Features:** 569 instances, 30 real-valued cell-nuclei features (mean, standard error, and worst values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), 0 missing values.
- **Target Encoding (`is_malignant`):**
  - `1` = Malignant (212 instances, 37.26%) — positive class.
  - `0` = Benign (357 instances, 62.74%) — negative class.

## 3. Preprocessing & Algorithms

- **Strict Validation:** Non-numeric strings, booleans, `+/-inf`, negative measurements, duplicate/unexpected columns, and invalid/missing targets are rejected explicitly; legitimate `NaN` feature values are preserved for downstream imputation.
- **Leakage-Safe Pipeline:** `SimpleImputer(strategy="median")` + `StandardScaler()` are fit strictly inside training folds.

**Algorithms Compared:**

1. **Logistic Regression** (`class_weight="balanced"`)
2. **RBF Support Vector Machine** (`CalibratedClassifierCV(SVC(class_weight="balanced"), cv=3)`)
3. **Random Forest** (`class_weight="balanced"`)
4. **XGBoost** (`FoldWeightedXGBClassifier` computing `scale_pos_weight` inside each training fold)

## 4. Cross-Validation, Model Selection & Threshold Selection

- **Split:** Stratified 80% train (`n=455`) / 20% held-out test (`n=114`), `random_state=42`.
- **Model Selection:** 5-fold Stratified Cross-Validation on the training split using **ROC-AUC**. Logistic Regression was selected with CV ROC-AUC of `0.9950`.
- **Threshold Selection:** Training-only out-of-fold (OOF) F2-score optimization (`threshold_strategy="f2_oof"`), selecting `0.37` without accessing held-out test labels.

*F2 optimization is an educational statistical objective, not a clinically validated threshold.*

## 5. Installation

Python `>=3.10` is supported.

```bash
# 1. Core ML pipeline only
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

# 2. Development & testing tools (pytest, ruff)
pip install -e ".[dev]"

# 3. Optional Streamlit demonstration UI
pip install -e ".[ui]"
```

## 6. Usage

```bash
# Prepare dataset CSV & run EDA
python scripts/prepare_data.py
python scripts/run_eda.py

# Train & select best model (saves schema-validated bundle)
python -m disease_prediction.train

# Evaluate on held-out test split (generates metrics.json, report, and figures)
python -m disease_prediction.evaluate

# Run sample prediction
python -m disease_prediction.predict --sample

# Launch optional educational Streamlit app
streamlit run app.py
```

> **Security Note:** Saved `.joblib` bundles use pickle serialization and schema validation (`schema_version="1.0.0"`). Only load artifacts from trusted sources.

## 7. Actual Evaluation Results

### 5-Fold Cross-Validation

<text-placeholder />

**Training Split:** `n=455`, threshold = `0.50`

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | F2 Opt Thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| **logistic_regression (Selected)** | **0.9714** | **0.9645** | **0.9588** | **0.9617** | **0.9950** | **0.0201** | **0.37** |
| svm_rbf | 0.9714 | 0.9701 | 0.9529 | 0.9614 | 0.9941 | 0.0245 | 0.29 |
| random_forest | 0.9604 | 0.9419 | 0.9529 | 0.9474 | 0.9904 | 0.0324 | 0.46 |
| xgboost | 0.9670 | 0.9586 | 0.9529 | 0.9558 | 0.9930 | 0.0248 | 0.37 |

### Held-Out Test Set

**Test Set:** `n=114`, selected model: `logistic_regression`

| Threshold | Accuracy | Precision | Recall | F1-Score | F2-Score | ROC-AUC | Brier Score | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.50` (Default) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9569 | 0.9954 | 0.0222 | 2 | 1 |
| `0.37` (OOF F2-Tuned) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9569 | 0.9954 | 0.0222 | 2 | 1 |

**5-Seed Split Sensitivity (`[7, 21, 42, 84, 123]`):**

- Recall: `0.9619 ± 0.0117`
- F1-Score: `0.9552 ± 0.0080`
- ROC-AUC: `0.9941 ± 0.0033`

## 8. Calibration & Medical Limitations

- **Uncalibrated Scores:** Balanced class weights shift predicted scores toward the minority Malignant class. Outputs are classification confidence scores, not calibrated clinical risk probabilities. See `reports/figures/calibration_curve.png` and `reports/model_evaluation.md`.
- **Cohort Limitations:** Single-center 1995 FNA cohort (`n=569`) without patient demographics or multi-site validation.
- **Clinical Limitations:** The dataset and model have not undergone clinical validation. Results cannot establish real-world diagnostic performance or clinical usefulness.

## 9. Testing & Code Quality

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

## 10. Future Improvements

- Nested training-only isotonic/Platt probability calibration evaluation.
- External multi-center dataset validation and decision-curve analysis.
- Further evaluation of false-negative cases and model generalization.

## Author

**Aryan Saxena**
