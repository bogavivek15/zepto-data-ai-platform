import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models import train_models


# --------------------------------
# Train models
# --------------------------------

models = train_models()

X_test = models["X_test"]
y_test = models["y_test"]


# =================================
# ROC CURVES
# =================================

plt.figure(figsize=(8, 6))

for model_name, model in models.items():

    if model_name in [
        "X_test",
        "y_test",
        "preprocessor"
    ]:
        continue

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_score = model.decision_function(
            X_test
        )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves - Titanic Survival Models"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    "analytics/roc_curves.png",
    dpi=300
)

plt.close()

print(
    "\nROC curve saved to:"
)

print(
    "analytics/roc_curves.png"
)


# =================================
# CONFUSION MATRICES
# =================================

for model_name, model in models.items():

    if model_name in [
        "X_test",
        "y_test",
        "preprocessor"
    ]:
        continue

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Did Not Survive",
            "Survived"
        ]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    file_path = (
        "analytics/"
        + model_name
        + "_confusion_matrix.png"
    )

    plt.savefig(
        file_path,
        dpi=300
    )

    plt.close()

    print(
        "Confusion matrix saved to:"
    )

    print(
        file_path
    )


# =================================
# RANDOM FOREST FEATURE IMPORTANCE
# =================================

feature_names = (
    models["preprocessor"]
    .get_feature_names_out()
)

random_forest = models[
    "random_forest"
]

rf_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": random_forest.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="importance",
    ascending=False
)

print(
    "\n========== RANDOM FOREST FEATURE IMPORTANCE =========="
)

print(
    rf_importance.head(10).to_string(
        index=False
    )
)


top_features = rf_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 10 Random Forest Feature Importances"
)

plt.tight_layout()

plt.savefig(
    "analytics/random_forest_feature_importance.png",
    dpi=300
)

plt.close()

print(
    "\nFeature importance chart saved to:"
)

print(
    "analytics/random_forest_feature_importance.png"
)


# =================================
# LOGISTIC REGRESSION COEFFICIENTS
# =================================

logistic_model = models[
    "logistic_regression"
]

logistic_coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logistic_model.coef_[0]
})

logistic_coefficients[
    "absolute_coefficient"
] = (
    logistic_coefficients[
        "coefficient"
    ].abs()
)

logistic_coefficients = (
    logistic_coefficients.sort_values(
        by="absolute_coefficient",
        ascending=False
    )
)

print(
    "\n========== LOGISTIC REGRESSION COEFFICIENTS =========="
)

print(
    logistic_coefficients.head(10).to_string(
        index=False
    )
)