# Module 2 — Titanic Data Analysis and Predictive Modeling

## Overview

This module performs exploratory data analysis, preprocessing, classification, class-imbalance handling, model tuning, regression, evaluation, and model persistence using the Titanic dataset.

Dataset: `titanic.csv`

---

## 1. Exploratory Data Analysis

The dataset contains **891 rows and 15 columns**.

Missing values were found in:

* `age`: 177
* `embarked`: 2
* `deck`: 688
* `embark_town`: 2

Cleaning performed:

* Median imputation for `age`
* Removed rows with missing `embarked` and `embark_town`
* Removed `deck` because 77.22% of its values were missing

Final cleaned dataset:

* **889 rows**
* **14 columns**

### Key Findings

* Female survival rate: **74.04%**
* Male survival rate: **18.89%**
* First-class survival rate: **62.62%**
* Second-class survival rate: **47.28%**
* Third-class survival rate: **24.24%**
* Fare distribution is right-skewed.
* Strongest correlation: `fare` ↔ `pclass` = **-0.5482**

---

## 2. Preprocessing

An 80/20 stratified train-test split was used.

* Training samples: **711**
* Test samples: **178**

Preprocessing included:

* Missing-value imputation
* One-hot encoding
* Standardization

Preprocessing was fitted only on the training data to prevent data leakage.

---

## 3. Classification Models

Three models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 | 0.8196 |

Random Forest achieved the highest baseline F1 score.

---

## 4. Class Imbalance

Three approaches were compared:

| Strategy     | Precision | Recall |     F1 |
| ------------ | --------: | -----: | -----: |
| Baseline     |    0.7833 | 0.6912 | 0.7344 |
| Class Weight |    0.7183 | 0.7500 | 0.7338 |
| SMOTE        |    0.7353 | 0.7353 | 0.7353 |

SMOTE achieved the highest F1 score and was applied only to the training data.

---

## 5. Random Forest Tuning

GridSearchCV selected:

```text
max_depth = 5
max_features = sqrt
n_estimators = 200
```

Best cross-validation F1:

```text
0.7408
```

Tuned Random Forest:

```text
Accuracy: 0.8315
Precision: 0.8654
Recall: 0.6618
F1: 0.7500
AUC: 0.8389
```

---

## 6. Regression

Linear Regression was used to predict `fare`.

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 18.3945 |
| RMSE        | 41.3578 |
| R²          |  0.3589 |
| Adjusted R² |  0.2818 |

Residual analysis showed evidence of heteroscedasticity.

---

## 7. Final Recommendation

Random Forest was selected as the preferred baseline classifier because it achieved the highest F1 score (**0.7424**) among the three required models.

After tuning, the Random Forest improved to an F1 score of **0.7500**.

The final pipeline was saved as:

```text
analytics/artifacts/best_titanic_pipeline.joblib
```

The saved pipeline was successfully reloaded and produced identical predictions.

---

## 8. Project Structure

```text
analytics/
├── 01_eda.py
├── 02_modeling.py
├── README.md
├── titanic.csv
├── titanic_cleaned.csv
├── artifacts/
│   └── best_titanic_pipeline.joblib
└── charts/
    ├── age_boxplot.png
    ├── age_fare_survival.png
    ├── age_histogram.png
    ├── confusion_matrices.png
    ├── correlation_heatmap.png
    ├── decision_tree.png
    ├── fare_boxplot.png
    ├── fare_by_survival.png
    ├── fare_histogram.png
    ├── fare_residuals.png
    ├── roc_curves.png
    ├── survival_by_pclass.png
    ├── survival_by_sex.png
    └── survival_sex_pclass.png
```
