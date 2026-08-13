import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from preprocessing import prepare_data


def train_models():

    # --------------------------------
    # Prepare data
    # --------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor
    ) = prepare_data()


    # --------------------------------
    # Logistic Regression
    # --------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_model.fit(
        X_train,
        y_train
    )


    # --------------------------------
    # Ridge Classifier
    # --------------------------------

    ridge_model = RidgeClassifier(
        random_state=42
    )

    ridge_model.fit(
        X_train,
        y_train
    )


    # --------------------------------
    # Random Forest
    # --------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    random_forest.fit(
        X_train,
        y_train
    )


    # --------------------------------
    # Return models and test data
    # --------------------------------

    return {
        "logistic_regression": logistic_model,
        "ridge_classifier": ridge_model,
        "random_forest": random_forest,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor
    }


if __name__ == "__main__":

    models = train_models()

    X_test = models["X_test"]
    y_test = models["y_test"]


    # --------------------------------
    # Evaluate models
    # --------------------------------

    results = []


    for model_name, model in models.items():

        if model_name in [
            "X_test",
            "y_test",
            "preprocessor"
        ]:
            continue


        # Make predictions
        y_pred = model.predict(
            X_test
        )


        # Get probability/decision scores
        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = model.decision_function(
                X_test
            )


        # Calculate metrics
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

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )

        conf_matrix = confusion_matrix(
            y_test,
            y_pred
        )


        # --------------------------------
        # Display individual results
        # --------------------------------

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
            "ROC-AUC:",
            roc_auc
        )

        print(
            "Confusion Matrix:"
        )

        print(
            conf_matrix
        )


        # --------------------------------
        # Store results
        # --------------------------------

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        })


    # --------------------------------
    # Model comparison
    # --------------------------------

    comparison = pd.DataFrame(
        results
    )


    print(
        "\n========== MODEL COMPARISON =========="
    )

    print(
        comparison.to_string(
            index=False
        )
    )


    # --------------------------------
    # Best model by F1
    # --------------------------------

    best_model = comparison.loc[
        comparison["F1"].idxmax()
    ]


    print(
        "\n========== BEST MODEL BY F1 =========="
    )

    print(
        best_model.to_string()
    )