# Module 2 - Titanic Data Analysis and Predictive Modeling

## Overview

This module performs exploratory data analysis, data cleaning, preprocessing, classification, class-imbalance handling, hyperparameter tuning, regression, evaluation, and model persistence using the Titanic dataset.

Dataset: `titanic.csv`

The workflow is cohesive:

```text
sns.load_dataset("titanic")
        |
titanic.csv (raw - saved immediately)
        |
        +---> EDA analysis → titanic_cleaned.csv (EDA artifact only)
        |
        +---> 02_modeling.py reads raw titanic.csv
                    |
              train/test split
                    |
              preprocessing fitted on training data only
                    |
              classification modeling
                    |
              imbalance comparison
                    |
              Random Forest tuning
                    |
              fare regression
                    |
              complete pipeline artifact
```

The raw Titanic dataset is loaded from Seaborn exactly once in `01_eda.py`. The raw dataset is immediately saved as `titanic.csv` as the offline fallback. The EDA analysis continues with cleaning operations and creates `titanic_cleaned.csv` for EDA purposes only. **The modeling pipeline (`02_modeling.py`) reads the raw `titanic.csv` and performs train/test split BEFORE any preprocessing.** All imputation, encoding, and scaling operations are fitted only on training data, preventing data leakage.

---

## 1. Exploratory Data Analysis

The original dataset contains:

```text
Rows: 891
Columns: 15
```

The following profiling outputs are produced:

* `df.shape`
* `df.info()`
* `df.describe()`

The raw dataset is immediately saved as:

```text
analytics/titanic.csv
```

This allows the project to be graded offline using:

```python
pd.read_csv("analytics/titanic.csv")
```

### Missing Values

| Column        | Missing Count | Missing % | Strategy                                    |
| ------------- | ------------: | --------: | ------------------------------------------- |
| `age`         |           177 |  19.8653% | Median imputation                           |
| `embarked`    |             2 |   0.2245% | Drop affected rows                          |
| `embark_town` |             2 |   0.2245% | Column dropped as redundant with `embarked` |
| `deck`        |           688 |  77.2166% | Column dropped due to very high missingness |

The cleaning follows the required threshold rule:

* Below 5% missing -> drop affected rows.
* 5%-30% missing -> impute.
* Very high missingness -> drop the column when reliable imputation is not justified.

For `age`, the measured missing rate was **19.8653%**, so median imputation was used. The median age was **28.00**.

For `embarked`, the missing rate was **0.2245%**, so the two affected rows were dropped.

`embark_town` was dropped because it is redundant with `embarked`.

`deck` had **77.2166%** missing values, so it was removed because reliable imputation would not be defensible.

### Cleaned Dataset

The cleaned dataset contains:

```text
889 rows
13 columns
0 missing values
```

It is saved as:

```text
analytics/titanic_cleaned.csv
```

---

## 2. Univariate Analysis

### Age IQR Analysis

```text
Q1 = 22.0
Q3 = 35.0
IQR = 13.0

Lower limit = 2.5
Upper limit = 54.5

Outliers = 65
```

The IQR rule used was:

```text
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR
```

### Fare IQR Analysis

```text
Q1 = 7.8958
Q3 = 31.0
IQR = 23.1042

Lower limit = -26.7605
Upper limit = 65.6563

Outliers = 114
```

### Fare Statistics

```text
Mean   = 32.0967
Median = 14.4542
Mode   = 8.05
```

The fare distribution is **right-skewed** because:

```text
Mean > Median > Mode
32.0967 > 14.4542 > 8.05
```

The high positive tail caused by expensive fares increases the mean substantially above the median and mode.

### Generated Charts

* `age_histogram.png`
* `age_boxplot.png`
* `fare_histogram.png`
* `fare_boxplot.png`

---

## 3. Bivariate Analysis

### Survival Rate by Sex

```text
Female: 74.0385%
Male:   18.8908%
```

Women had substantially higher survival rates than men. This indicates a strong association between sex and survival. However, sex alone does not explain differences within passenger classes.

Chart:

```text
charts/survival_by_sex.png
```

### Survival Rate by Passenger Class

```text
1st Class: 62.6168%
2nd Class: 47.2826%
3rd Class: 24.2363%
```

First-class passengers had the highest survival rate, followed by second- and third-class passengers. This suggests passenger class was strongly associated with survival.

Chart:

```text
charts/survival_by_pclass.png
```

### Survival Rate by Sex and Passenger Class

| Sex    | Pclass | Survival Rate |
| ------ | -----: | ------------: |
| Female |      1 |      96.7391% |
| Female |      2 |      92.1053% |
| Female |      3 |      50.0000% |
| Male   |      1 |      36.8852% |
| Male   |      2 |      15.7407% |
| Male   |      3 |      13.5447% |

Female passengers had higher survival rates than male passengers within each passenger class. First- and second-class women had especially high survival rates, while third-class men had the lowest survival rate.

Chart:

```text
charts/survival_sex_pclass.png
```

### Boolean Masking

The analysis also demonstrates boolean masking using combinations of conditions.

Examples:

```text
Female passengers in first class: 92
Male passengers in third class: 347
```

---

## 4. Correlation Analysis

The correlation matrix uses exactly these six columns:

```text
survived
pclass
age
sibsp
parch
fare
```

The derived boolean columns `adult_male` and `alone` are intentionally excluded.

### Correlation Matrix

```text
          survived    pclass       age     sibsp     parch      fare
survived   1.000000 -0.335549 -0.069822 -0.034040  0.083151  0.255290
pclass    -0.335549  1.000000 -0.336512  0.081656  0.016824 -0.548193
age       -0.069822 -0.336512  1.000000 -0.232543 -0.171485  0.093707
sibsp     -0.034040  0.081656 -0.232543  1.000000  0.414542  0.160887
parch      0.083151  0.016824 -0.171485  0.414542  1.000000  0.217532
fare       0.255290 -0.548193  0.093707  0.160887  0.217532  1.000000
```

Chart:

```text
charts/correlation_heatmap.png
```

### Two Strongest Correlations

The two feature pairs with the largest absolute off-diagonal correlations are:

```text
pclass <-> fare = -0.5482
sibsp  <-> parch =  0.4145
```

The negative `pclass`-`fare` correlation indicates that lower numerical passenger-class values, representing higher classes, tend to be associated with higher fares.

The positive `sibsp`-`parch` correlation indicates that passengers travelling with siblings/spouses also tended to have parents/children travelling with them.

---

## 5. Multivariate Data Story

### Chart 1 - Survival by Sex

Women had substantially higher survival rates than men. This indicates a strong association between sex and survival. However, sex alone does not explain differences within passenger classes.

### Chart 2 - Survival by Passenger Class

First-class passengers had the highest survival rate, followed by second- and third-class passengers. This suggests passenger class was strongly associated with survival.

### Chart 3 - Survival by Sex and Passenger Class

Female passengers had higher survival rates than male passengers within each passenger class. First- and second-class women had especially high survival rates, while third-class men had the lowest survival rate.

### Chart 4 - Age, Fare and Survival

Survivors are distributed across a wide range of ages and fares. Higher fares are associated with higher passenger classes, so fare also captures part of the socioeconomic differences associated with survival.

### Chart 5 - Fare by Survival

Survivors generally paid higher fares than passengers who did not survive. Both groups contain substantial variation and extreme fare values, but the difference supports the positive association between fare and survival.

Generated charts include:

```text
age_fare_survival.png
fare_by_survival.png
survival_by_sex.png
survival_by_pclass.png
survival_sex_pclass.png
```

---

## 6. Exploratory Standardization

`age` and `fare` were standardized using the z-score transformation:

```text
z = (x - mean) / std
```

### Before Standardization

| Statistic |     Age |    Fare |
| --------- | ------: | ------: |
| Mean      | 29.3152 | 32.0967 |
| Std       | 12.9849 | 49.6975 |

### After Standardization

| Statistic | Age | Fare |
| --------- | --: | ---: |
| Mean      | about 0 | about 0 |
| Std       | about 1 | about 1 |

The transformed values have approximately zero mean and unit standard deviation, confirming that the standardization was successful.

This EDA-stage standardization is only a sanity check. It does **not** feed the classification modeling pipeline. The modeling pipeline uses the raw `titanic.csv` dataset and applies its own preprocessing fitted only on training data.

---

# Part B - Predictive Modeling

## 7. Class Balance and Stratified Split

The raw dataset contains:

```text
Not survived: 549
Survived:     342
```

Class proportions:

```text
Not survived: 61.6162%
Survived:     38.3838%
```

An 80/20 stratified train-test split was used.

```text
Training samples: 712
Test samples:     179
```

Stratification was used so that the survived/not-survived class proportions remain approximately consistent between the training and test sets.

**The split occurs BEFORE fitting any preprocessing component. This prevents test-set information from leaking into the training process.**

---

## 8. Classification Preprocessing

The classification features are:

```text
pclass
age
sibsp
parch
fare
sex
embarked
```

Numeric preprocessing:

```text
Median imputation
        |
StandardScaler
```

Categorical preprocessing:

```text
Most-frequent imputation
        |
One-hot encoding
```

The preprocessing is implemented using `ColumnTransformer` and `Pipeline`.

All preprocessing components are fitted only on the training data and then applied to the test data using transform-only behavior. This prevents test-set information from leaking into model training.

---

## 9. Classification Models

Three classifiers were trained on the same train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Classification Results

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.7654 |    0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8300 |

The Random Forest achieved the highest F1 score among the three required classifiers.

Generated evaluation charts:

```text
charts/confusion_matrices.png
charts/roc_curves.png
```

The Decision Tree was rendered using `plot_tree` with feature names and class names.

Chart:

```text
charts/decision_tree.png
```

---

## 10. Confusion Matrices

The confusion matrices were generated for all three classifiers.

Each matrix reports:

```text
True Negatives
False Positives
False Negatives
True Positives
```

### Confusion Matrix Values

Logistic Regression:

```text
[[98 12]
 [23 46]]
```

Decision Tree:

```text
[[97 13]
 [29 40]]
```

Random Forest:

```text
[[98 12]
 [21 48]]
```

The combined visualization is stored as:

```text
charts/confusion_matrices.png
```

---

## 11. ROC Curves and AUC

ROC curves were generated for all three classifiers.

The resulting AUC values are:

| Model               |    AUC |
| ------------------- | -----: |
| Logistic Regression | 0.8437 |
| Decision Tree       | 0.7971 |
| Random Forest       | 0.8300 |

The ROC curves are stored in:

```text
charts/roc_curves.png
```

---

## 12. Class Imbalance Comparison

The survived/not-survived class distribution is:

```text
0 = 549
1 = 340
```

Three strategies were compared using Logistic Regression:

1. Baseline
2. `class_weight='balanced'`
3. SMOTE

### Results

| Strategy                  | Precision | Recall |     F1 |
| ------------------------- | --------: | -----: | -----: |
| Baseline                  |    0.7931 | 0.6667 | 0.7244 |
| `class_weight='balanced'` |    0.7297 | 0.7826 | 0.7552 |
| SMOTE                     |    0.7397 | 0.7826 | 0.7606 |

SMOTE achieved the highest F1 score among the three imbalance strategies. SMOTE was applied only to the training data after training-data preprocessing, while the test data remained untouched.

---

## 13. Random Forest Hyperparameter Tuning

`GridSearchCV` was used on the Random Forest with:

```text
n_estimators
max_depth
max_features
```

The best parameters were:

```text
max_depth = 5
max_features = sqrt
n_estimators = 100
```

Best cross-validation F1:

```text
0.7459
```

The Random Forest was constructed with:

```python
RandomForestClassifier(
    oob_score=True,
    random_state=42
)
```

OOB score:

```text
0.8272
```

### Tuned Random Forest Test Results

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8156 |
| Precision | 0.8750 |
| Recall    | 0.6087 |
| F1        | 0.7179 |
| AUC       | 0.8431 |

---

## 14. Regression - Fare Prediction

A multivariate Linear Regression model was used to predict `fare` from the other available features.

### Regression Results

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 16.8837 |
| RMSE        | 28.6730 |
| R2          |  0.4687 |
| Adjusted R2 |  0.3819 |

### Residual Analysis

The residual plot is stored as:

```text
charts/fare_residuals.png
```

The residual analysis indicates evidence of heteroscedasticity, meaning the spread of residuals is not approximately constant across the range of predicted fare values.

---

## 15. Final Model Comparison

### Classification Metrics

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.7654 |    0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8300 |

### Regression Metrics

| Model             |     MAE |    RMSE |     R2 | Adjusted R2 |
| ----------------- | ------: | ------: | -----: | ----------: |
| Linear Regression | 16.8837 | 28.6730 | 0.4687 |      0.3819 |

Classification and regression metrics are presented as separate metric groups because they evaluate different prediction tasks and are not directly comparable.

---

## 16. Final Recommendation

**Random Forest is the preferred classifier among the three required models** because it achieved the highest baseline F1 score of **0.7442**. It achieved **81.56% accuracy**, **80.00% precision**, **69.57% recall**, and **0.8300 AUC**. 

After hyperparameter tuning, the Random Forest achieved:
- **Accuracy: 81.56%**
- **Precision: 87.50%**  
- **Recall: 60.87%**
- **F1: 0.7179**
- **AUC: 0.8431**

**Deployment Recommendation:** Based on the evaluation metrics, **I would deploy the baseline Random Forest model (not the tuned version) for the Titanic survival classification task** because it achieves the best balance between precision (80.00%) and recall (69.57%), resulting in the highest F1 score (0.7442) among all three baseline classifiers. The tuned Random Forest, while having higher precision (87.50%), has significantly lower recall (60.87%), which means it misses more survivors. For a survival prediction task, balanced performance across both classes is preferable, making the baseline Random Forest the better choice for deployment.

The regression model achieved MAE of 16.88, RMSE of 28.67, R² of 0.4687, and Adjusted R² of 0.3819. The regression metrics are evaluated separately because they measure the different task of predicting fare.

---

## 17. Complete Pipeline Artifact

The best fitted Random Forest pipeline includes:

```text
Preprocessing
    |-- Missing-value imputation
    |-- One-hot encoding
    `-- StandardScaler

        |

Random Forest classifier
```

The complete fitted pipeline is saved as:

```text
analytics/artifacts/best_titanic_pipeline.joblib
```

The artifact contains the preprocessing steps together with the estimator rather than saving only the bare model.

The saved pipeline was reloaded using `joblib.load()` and tested on raw feature input.

Reload verification:

```text
Original predictions:
[0 0 0 0 0]

Reloaded predictions:
[0 0 0 0 0]

Predictions identical:
True
```

Therefore, the saved artifact remains usable for end-to-end prediction on raw input data.

---

## 18. Project Structure

```text
analytics/
|-- 01_eda.py
|-- 02_modeling.py
|-- README.md
|-- titanic.csv
|-- titanic_cleaned.csv
|
|-- artifacts/
|   `-- best_titanic_pipeline.joblib
|
`-- charts/
    |-- age_boxplot.png
    |-- age_fare_survival.png
    |-- age_histogram.png
    |-- confusion_matrices.png
    |-- correlation_heatmap.png
    |-- decision_tree.png
    |-- fare_boxplot.png
    |-- fare_by_survival.png
    |-- fare_histogram.png
    |-- fare_residuals.png
    |-- roc_curves.png
    |-- survival_by_pclass.png
    |-- survival_by_sex.png
    `-- survival_sex_pclass.png
```

---

## 19. How to Run

From the project root:

### Run EDA

```bash
python analytics/01_eda.py
```

### Run Modeling

```bash
python analytics/02_modeling.py
```

The EDA script creates:

```text
analytics/titanic.csv
analytics/titanic_cleaned.csv
analytics/charts/
```

The modeling script reads:

```text
analytics/titanic.csv (raw data - NOT titanic_cleaned.csv)
```

and creates/updates:

```text
analytics/charts/
analytics/artifacts/best_titanic_pipeline.joblib
```

---

## 20. Module Completion

The module implements:

* Titanic dataset loading and offline fallback
* Missing-value profiling and threshold-based cleaning
* Univariate and bivariate EDA
* IQR outlier analysis
* Fare distribution analysis
* Required six-column correlation analysis
* Multivariate visual data story
* Exploratory standardization
* Stratified train/test split
* Leakage-safe preprocessing
* Logistic Regression
* Decision Tree
* Random Forest
* Confusion matrices
* Accuracy, precision, recall, F1 and AUC
* ROC curves
* Class imbalance comparison
* Training-only SMOTE
* Random Forest GridSearchCV
* OOB score
* Multivariate fare regression
* MAE, RMSE, R2 and Adjusted R2
* Residual analysis
* Final model comparison
* Complete fitted pipeline persistence
* Pipeline reload and prediction verification
