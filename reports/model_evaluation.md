# Disease Prediction (Breast Cancer Wisconsin) — Evaluation Report

> **Medical Disclaimer**: This model is an educational ML demonstration trained on the UCI Breast Cancer Wisconsin dataset. It is **not** clinically validated and must **never** be used for diagnosis, screening, or treatment decisions.

- **Selected Model**: `logistic_regression` (chosen via 5-fold CV `roc_auc`)
- **Threshold Strategy**: `f2_oof` (training-only OOF threshold = `0.37`)
- **Train / Test Split**: 455 / 114 (stratified, seed=42)

## 1. Cross-Validation Comparison (Training Set Only, Threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | F2 Opt Thr |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.9714 | 0.9645 | 0.9588 | 0.9617 | 0.9950 | 0.0201 | 0.37 |
| svm_rbf | 0.9714 | 0.9701 | 0.9529 | 0.9614 | 0.9941 | 0.0245 | 0.29 |
| random_forest | 0.9604 | 0.9419 | 0.9529 | 0.9474 | 0.9904 | 0.0324 | 0.46 |
| xgboost | 0.9670 | 0.9586 | 0.9529 | 0.9558 | 0.9930 | 0.0248 | 0.37 |

## 2. Held-Out Test Set Comparison (Default Threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9954 | 0.0222 |
| svm_rbf | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9950 | 0.0220 |
| random_forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9980 | 0.0264 |
| xgboost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9970 | 0.0180 |

## 3. Threshold Comparison on Held-Out Test Set (logistic_regression)

| Threshold | Accuracy | Precision | Recall | F1-Score | F2-Score | FN | FP |
|---|---|---|---|---|---|---|---|
| 0.50 (Default) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9569 | 2 | 1 |
| 0.37 (OOF F2-Tuned) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9569 | 2 | 1 |

Note: F2 optimization is an educational statistical objective weighting recall higher than precision on training OOF predictions, not a clinically validated threshold.

### Classification Report (Selected Threshold)

```text
precision    recall  f1-score   support

   Benign (0)       0.97      0.99      0.98        72
Malignant (1)       0.98      0.95      0.96        42

     accuracy                           0.97       114
    macro avg       0.97      0.97      0.97       114
 weighted avg       0.97      0.97      0.97       114
```

## 4. Probability Calibration & Score Interpretation

- **Test Brier Score**: `0.0222`
- **Class-Weighting Effect**: Balanced class weights (`class_weight='balanced'` / `scale_pos_weight`) shift predicted scores toward the minority Malignant class.
- **Interpretation**: Output scores are uncalibrated model confidence scores for ranking/classification, **not** calibrated real-world clinical risk probabilities.

## 5. Five-Seed Split Sensitivity (Threshold = 0.50)

- **Estimator**: `logistic_regression`
- **Seeds**: `[7, 21, 42, 84, 123]` (includes primary seed 42)
- **Recall**: 0.9619 +/- 0.0117
- **F1-Score**: 0.9552 +/- 0.0080
- **ROC-AUC**: 0.9941 +/- 0.0033
- Note: This measures variance across random 80/20 splits of the same 569-sample cohort, not external multi-center validation.
