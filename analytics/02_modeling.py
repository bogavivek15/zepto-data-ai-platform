import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    "analytics/charts",
    exist_ok=True
)

os.makedirs(
    "analytics/artifacts",
    exist_ok=True
)


# ============================================================
# LOAD THE SAME COMMITTED CLEANED DATA
# ============================================================

df = pd.read_csv(
    "analytics/titanic_cleaned.csv"
)

print(
    "\n========== MODELING DATASET =========="
)

print(
    "Shape:",
    df.shape
)


# ============================================================
# CLASSIFICATION DATA
# ============================================================

target = "survived"

classification_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked"
]

X = df[
    classification_features
]

y = df[
    target
]


# ============================================================
# CLASS BALANCE
# ============================================================

print(
    "\n========== CLASS BALANCE =========="
)

print(
    y.value_counts()
)

print(
    "\nClass proportions:"
)

print(
    y.value_counts(
        normalize=True
    )
)


# ============================================================
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\n========== TRAIN / TEST SPLIT =========="
)

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test:",
    y_test.shape
)

print(
    "\nStratification preserves approximately the same "
    "survived/not-survived class proportions in training and test data."
)


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_columns = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_columns = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# THREE REQUIRED CLASSIFIERS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# ============================================================
# TRAIN + EVALUATE THREE CLASSIFIERS
# ============================================================

classification_results = []

roc_data = {}

confusion_matrices = {}

trained_pipelines = {}


for model_name, estimator in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    if hasattr(
        pipeline,
        "predict_proba"
    ):

        y_score = pipeline.predict_proba(
            X_test
        )[:, 1]

    else:

        y_score = pipeline.decision_function(
            X_test
        )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc_score = roc_auc_score(
        y_test,
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc_score
    })

    roc_data[
        model_name
    ] = (
        y_test,
        y_score
    )

    confusion_matrices[
        model_name
    ] = cm

    trained_pipelines[
        model_name
    ] = pipeline

    print(
        f"\n========== {model_name.upper()} =========="
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1:",
        f1
    )

    print(
        "AUC:",
        auc_score
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


# ============================================================
# CLASSIFICATION COMPARISON TABLE
# ============================================================

classification_comparison = pd.DataFrame(
    classification_results
)

print(
    "\n========== CLASSIFICATION MODEL COMPARISON =========="
)

print(
    classification_comparison.to_string(
        index=False
    )
)


# ============================================================
# ROC CURVES
# ============================================================

plt.figure(
    figsize=(9, 7)
)

for model_name, (
    actual,
    scores
) in roc_data.items():

    fpr, tpr, _ = roc_curve(
        actual,
        scores
    )

    auc_value = roc_auc_score(
        actual,
        scores
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Titanic Classifiers"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    "analytics/charts/roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, (
    model_name,
    cm
) in zip(
    axes,
    confusion_matrices.items()
):

    ax.imshow(
        cm
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

plt.tight_layout()

plt.savefig(
    "analytics/charts/confusion_matrices.png",
    dpi=300
)

plt.close()


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

decision_tree_pipeline = trained_pipelines[
    "Decision Tree"
]

tree_model = decision_tree_pipeline[
    "model"
]

tree_preprocessor = decision_tree_pipeline[
    "preprocessor"
]

tree_feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(
    figsize=(24, 12)
)

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=[
        "Did not survive",
        "Survived"
    ],
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title(
    "Decision Tree - Titanic Survival"
)

plt.tight_layout()

plt.savefig(
    "analytics/charts/decision_tree.png",
    dpi=300
)

plt.close()


# ============================================================
# IMBALANCE COMPARISON
# ============================================================

print(
    "\n========== IMBALANCE COMPARISON =========="
)

imbalance_results = []


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(
    X_train,
    y_train
)

baseline_pred = baseline_pipeline.predict(
    X_test
)

imbalance_results.append({
    "Strategy": "Baseline",
    "Precision": precision_score(
        y_test,
        baseline_pred
    ),
    "Recall": recall_score(
        y_test,
        baseline_pred
    ),
    "F1": f1_score(
        y_test,
        baseline_pred
    )
})


# ------------------------------------------------------------
# CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train
)

balanced_pred = balanced_pipeline.predict(
    X_test
)

imbalance_results.append({
    "Strategy": "class_weight='balanced'",
    "Precision": precision_score(
        y_test,
        balanced_pred
    ),
    "Recall": recall_score(
        y_test,
        balanced_pred
    ),
    "F1": f1_score(
        y_test,
        balanced_pred
    )
})


# ------------------------------------------------------------
# SMOTE — TRAINING FOLD ONLY
# ------------------------------------------------------------

smote_preprocessor = preprocessor

X_train_processed = (
    smote_preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    smote_preprocessor.transform(
        X_test
    )
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_processed,
        y_train
    )
)

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_pred = smote_model.predict(
    X_test_processed
)

imbalance_results.append({
    "Strategy": "SMOTE",
    "Precision": precision_score(
        y_test,
        smote_pred
    ),
    "Recall": recall_score(
        y_test,
        smote_pred
    ),
    "F1": f1_score(
        y_test,
        smote_pred
    )
})


imbalance_comparison = pd.DataFrame(
    imbalance_results
)

print(
    imbalance_comparison.to_string(
        index=False
    )
)

best_imbalance = imbalance_comparison.loc[
    imbalance_comparison["F1"].idxmax()
]

print(
    "\nBest imbalance strategy by F1:",
    best_imbalance["Strategy"]
)

print(
    "The preferred strategy is selected using F1 because it balances "
    "precision and recall for the imbalanced target."
)


# ============================================================
# RANDOM FOREST GRID SEARCH + OOB
# ============================================================

print(
    "\n========== RANDOM FOREST GRID SEARCH =========="
)

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=42
            )
        )
    ]
)

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],
    "model__max_depth": [
        None,
        5,
        10
    ],
    "model__max_features": [
        "sqrt",
        "log2"
    ]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(
    X_train,
    y_train
)

best_rf_pipeline = (
    grid_search.best_estimator_
)

best_rf_model = best_rf_pipeline[
    "model"
]

print(
    "Best parameters:"
)

print(
    grid_search.best_params_
)

print(
    "Best cross-validation F1:",
    grid_search.best_score_
)

print(
    "OOB score:",
    best_rf_model.oob_score_
)


# ============================================================
# TUNED RANDOM FOREST TEST PERFORMANCE
# ============================================================

tuned_rf_pred = best_rf_pipeline.predict(
    X_test
)

tuned_rf_score = (
    best_rf_pipeline.predict_proba(
        X_test
    )[:, 1]
)

tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_score
)

print(
    "\n========== TUNED RANDOM FOREST =========="
)

print(
    "Accuracy:",
    tuned_rf_accuracy
)

print(
    "Precision:",
    tuned_rf_precision
)

print(
    "Recall:",
    tuned_rf_recall
)

print(
    "F1:",
    tuned_rf_f1
)

print(
    "AUC:",
    tuned_rf_auc
)


# ============================================================
# REGRESSION — PREDICT FARE
# ============================================================

print(
    "\n========== REGRESSION — FARE PREDICTION =========="
)

regression_features = [
    column
    for column in df.columns
    if column not in [
        "fare",
        "survived",
        "alive"
    ]
]

X_regression = df[
    regression_features
]

y_regression = df[
    "fare"
]

reg_numeric_columns = (
    X_regression
    .select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    )
    .columns
    .tolist()
)

reg_categorical_columns = (
    X_regression
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)

reg_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

reg_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_pipeline,
            reg_numeric_columns
        ),
        (
            "categorical",
            reg_categorical_pipeline,
            reg_categorical_columns
        )
    ]
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

reg_predictions = regression_pipeline.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        reg_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    reg_predictions
)

n = len(
    y_reg_test
)

p = (
    regression_pipeline[
        "preprocessor"
    ]
    .transform(
        X_reg_test
    )
    .shape[1]
)

adjusted_r2 = (
    1
    -
    (
        (1 - r2)
        * (n - 1)
        /
        (n - p - 1)
    )
)

print(
    "MAE:",
    mae
)

print(
    "RMSE:",
    rmse
)

print(
    "R²:",
    r2
)

print(
    "Adjusted R²:",
    adjusted_r2
)


# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test.values
    - reg_predictions
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    reg_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Fare Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    "analytics/charts/fare_residuals.png",
    dpi=300
)

plt.close()


# Simple heteroscedasticity assessment
abs_residual_correlation = np.corrcoef(
    reg_predictions,
    np.abs(residuals)
)[0, 1]

if abs_residual_correlation > 0.30:
    heteroscedasticity = "The residuals show evidence of heteroscedasticity."
else:
    heteroscedasticity = (
        "The residuals do not show strong evidence of heteroscedasticity."
    )

print(
    "\nHeteroscedasticity conclusion:"
)

print(
    heteroscedasticity
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

final_classification = classification_comparison.copy()

regression_row = pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Adjusted R²": adjusted_r2
}])

print(
    "\n========== FINAL CLASSIFICATION COMPARISON =========="
)

print(
    final_classification.to_string(
        index=False
    )
)

print(
    "\n========== REGRESSION METRICS =========="
)

print(
    regression_row.to_string(
        index=False
    )
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

best_classifier = final_classification.loc[
    final_classification["F1"].idxmax()
]

print(
    "\n========== FINAL RECOMMENDATION =========="
)

print(
    f"{best_classifier['Model']} has the highest F1 score "
    f"among the three required classifiers at "
    f"{best_classifier['F1']:.4f}. "
    f"Its accuracy is {best_classifier['Accuracy']:.4f}, "
    f"precision is {best_classifier['Precision']:.4f}, "
    f"recall is {best_classifier['Recall']:.4f}, "
    f"and AUC is {best_classifier['AUC']:.4f}. "
    "It is therefore the preferred classifier based on the balance "
    "between precision and recall. "
    "The regression model is evaluated separately because its "
    "MAE, RMSE, R² and Adjusted R² metrics measure a different task "
    "and are not directly comparable with classification metrics."
)


# ============================================================
# SAVE COMPLETE BEST PIPELINE
# ============================================================

full_pipeline_path = (
    "analytics/artifacts/"
    "best_titanic_pipeline.joblib"
)

joblib.dump(
    best_rf_pipeline,
    full_pipeline_path
)

print(
    "\nComplete pipeline saved to:"
)

print(
    full_pipeline_path
)


# ============================================================
# RELOAD SAVED PIPELINE
# ============================================================

loaded_pipeline = joblib.load(
    full_pipeline_path
)

raw_sample = X_test.iloc[
    :5
]

original_predictions = (
    best_rf_pipeline.predict(
        raw_sample
    )
)

reloaded_predictions = (
    loaded_pipeline.predict(
        raw_sample
    )
)

print(
    "\n========== SAVED PIPELINE RELOAD CHECK =========="
)

print(
    "Original predictions:",
    original_predictions
)

print(
    "Reloaded predictions:",
    reloaded_predictions
)

print(
    "Predictions identical:",
    np.array_equal(
        original_predictions,
        reloaded_predictions
    )
)

print(
    "\n========== MODELING COMPLETE =========="
)