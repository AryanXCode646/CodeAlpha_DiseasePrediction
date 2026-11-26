# Dataset Documentation — Breast Cancer Wisconsin (Diagnostic)

- **Name**: Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source**: UCI Machine Learning Repository (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian) via `sklearn.datasets.load_breast_cancer`
- **License**: CC BY 4.0
- **Size**: 569 instances, 30 real-valued cell-nuclei features (mean, standard error, worst), 0 missing values
- **Target Variable (`is_malignant`)**:
  - `1` = Malignant (212 instances, 37.26%)
  - `0` = Benign (357 instances, 62.74%)

## Setup

Export a clean local CSV copy via:

```bash
python scripts/prepare_data.py
```

## Medical Disclaimer & Limitations

- Derived from digitized fine-needle aspirate (FNA) images from a single clinical cohort (1995).
- Lacks patient demographics, clinical history, and multi-center validation.
- **Must never be used for clinical diagnosis or treatment decisions.**
