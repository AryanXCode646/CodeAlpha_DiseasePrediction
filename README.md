# Disease Prediction from Medical Data — CodeAlpha ML Internship (Task 4)

> **Medical Disclaimer**: This project is an educational machine learning benchmark trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset. It is **not** clinically validated and must **never** be used for medical diagnosis, screening, or treatment decisions.

## Problem Statement & Objectives

Predict whether a breast mass fine-needle aspirate (FNA) sample is **Benign (`0`)** or **Malignant (`1`)** from 30 structured cell-nuclei measurements, with special focus on **Recall** and minimizing **False Negatives** (missed malignant cases).

## Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: UCI Machine Learning Repository (via `scikit-learn.datasets.load_breast_cancer`)
- **License**: CC BY 4.0
- **Size**: 569 instances, 30 real-valued features, 0 missing values
- **Class Distribution**: 357 Benign (`62.74%`), 212 Malignant (`37.26%`)

## Technology Stack

- Python 3.12 (`>=3.10`)
- `scikit-learn`, `xgboost`, `pandas`, `numpy`
- `matplotlib`, `seaborn`, `joblib`
- `streamlit` (optional interactive UI)
- `pytest`, `ruff`

## Repository Structure

```text
CodeAlpha_DiseasePrediction/
├── app.py                     # Optional Streamlit prediction interface
├── configs/config.json        # Dataset, training, and evaluation configuration
├── data/                      # Dataset directory (gitignored CSV)
├── models/                    # Saved model bundle (gitignored)
├── reports/
│   ├── figures/               # EDA & evaluation plots
│   ├── metrics.json           # Machine-readable evaluation metrics
│   └── model_evaluation.md    # Detailed evaluation report
├── scripts/
│   ├── prepare_data.py        # Exports Breast Cancer Wisconsin CSV
│   └── run_eda.py             # Generates EDA summary & figures
├── src/disease_prediction/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```bash
# 1. Prepare dataset CSV
python scripts/prepare_data.py

# 2. Run exploratory data analysis
python scripts/run_eda.py

# 3. Train & select best model (5-fold Stratified CV on 80% train split)
python -m disease_prediction.train

# 4. Evaluate on isolated 20% test split and generate figures
python -m disease_prediction.evaluate

# 5. Run sample prediction
python -m disease_prediction.predict --sample

# 6. Launch optional Streamlit UI
streamlit run app.py
```

## Actual Evaluation Results

Models compared on 5-fold Stratified CV (`random_state=42`, `n=455` train) and evaluated on the isolated 20% test set (`n=114`):

### 5-Fold Cross-Validation (Training Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Selected)** | **0.9714** | **0.9645** | **0.9588** | **0.9617** | **0.9950** |
| SVM (RBF) | 0.9714 | 0.9645 | 0.9588 | 0.9617 | 0.9946 |
| Random Forest | 0.9604 | 0.9419 | 0.9529 | 0.9474 | 0.9904 |
| XGBoost | 0.9670 | 0.9586 | 0.9529 | 0.9558 | 0.9930 |

### Held-Out Test Set (`n=114`, `logistic_regression`)

- **Accuracy**: `0.9737`
- **Precision (Malignant)**: `0.9756`
- **Recall (Malignant)**: `0.9524`
- **F1-Score (Malignant)**: `0.9639`
- **ROC-AUC**: `0.9954`
- **False Negatives (Missed Malignant)**: `2` out of 42
- **5-Seed Split Sensitivity**: Recall `0.9619 +/- 0.0117`, ROC-AUC `0.9941 +/- 0.0033`

See [reports/model_evaluation.md](reports/model_evaluation.md) and `reports/figures/` for confusion matrix, ROC curves, and top feature importances.

## Testing & Code Quality

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

## Limitations & Future Improvements

- **Cohort Size & Diversity**: Single-center 1995 cohort (`n=569`) without demographic metadata.
- **Future Work**: Probability calibration (`CalibratedClassifierCV`), threshold optimization to minimize false negatives, and external multi-center validation.

## Author

**Aryan Saxena** — CodeAlpha Machine Learning Internship
