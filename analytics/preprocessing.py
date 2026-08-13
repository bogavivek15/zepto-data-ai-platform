import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE


def prepare_data():

    df = pd.read_csv("analytics/titanic.csv")

    # Remove target leakage
    df = df.drop(columns=["alive"])

    # Separate target and features
    X = df.drop(columns=["survived"])
    y = df["survived"]

    # Identify columns
    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Numeric preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine preprocessing
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

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    # Apply SMOTE only to training data
    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_processed,
        y_train
    )

    return (
        X_train_smote,
        y_train_smote,
        X_test_processed,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    (
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor
    ) = prepare_data()

    print("\n========== PREPROCESSING COMPLETE ==========")

    print(
        "Training data after SMOTE:",
        len(y_train)
    )

    print(
        "Training data shape:",
        X_train.shape
    )

    print(
        "Test data shape:",
        X_test.shape
    )

    print("\nTraining class distribution:")

    print(
        y_train.value_counts()
    )

    print("\nTest class distribution:")

    print(
        y_test.value_counts()
    )