# Disease Prediction (Breast Cancer Wisconsin) — Evaluation Report

> **Medical Disclaimer**: This model is an educational ML demonstration trained on the UCI Breast Cancer Wisconsin dataset. It is **not** clinically validated and must **never** be used as a medical diagnosis or treatment tool.

- **Selected Model**: `logistic_regression` (chosen via 5-fold CV `roc_auc`)
- **Train / Test Split**: 455 / 114 (stratified, seed=42)
- **Test False Negatives (Missed Malignant)**: `2`

## 1. Cross-Validation Comparison (Training Set Only)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.9714 | 0.9645 | 0.9588 | 0.9617 | 0.9950 |
| svm_rbf | 0.9714 | 0.9645 | 0.9588 | 0.9617 | 0.9946 |
| random_forest | 0.9604 | 0.9419 | 0.9529 | 0.9474 | 0.9904 |
| xgboost | 0.9670 | 0.9586 | 0.9529 | 0.9558 | 0.9930 |

## 2. Held-Out Test Set Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9954 |
| svm_rbf | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9954 |
| random_forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9980 |
| xgboost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9970 |

## 3. Selected Model (logistic_regression) Details & Split Sensitivity

- **Test Accuracy**: 0.9737
- **Test Recall (Malignant)**: 0.9524
- **Test ROC-AUC**: 0.9954
- **5-Seed Split Sensitivity (Recall)**: 0.9619 +/- 0.0117
- **5-Seed Split Sensitivity (ROC-AUC)**: 0.9941 +/- 0.0033

### Classification Report

```text
precision    recall  f1-score   support

   Benign (0)       0.97      0.99      0.98        72
Malignant (1)       0.98      0.95      0.96        42

     accuracy                           0.97       114
    macro avg       0.97      0.97      0.97       114
 weighted avg       0.97      0.97      0.97       114
```

## 4. Limitations & Bias Notes

- **Class Imbalance**: 62.7% Benign vs. 37.3% Malignant; balanced class weights are applied during training.
- **Cohort Bias**: Single-center 1995 FNA cohort (n=569) without demographic or multi-site validation.
- **False Negatives**: In oncology screening, false negatives carry severe risk; models cannot replace pathologist review.
