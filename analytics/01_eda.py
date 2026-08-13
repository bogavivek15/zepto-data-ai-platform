import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# --------------------------------
# Load Titanic dataset ONCE
# --------------------------------

df = sns.load_dataset("titanic")


# --------------------------------
# Save raw offline fallback
# --------------------------------

df.to_csv(
    "analytics/titanic.csv",
    index=False
)


# --------------------------------
# Clean data
# --------------------------------

df["age"] = df["age"].fillna(
    df["age"].median()
)

df = df.dropna(
    subset=["embarked"]
)

df = df.dropna(
    subset=["embark_town"]
)

df = df.drop(
    columns=["deck"]
)


# --------------------------------
# Save cleaned data
# --------------------------------

df.to_csv(
    "analytics/titanic_cleaned.csv",
    index=False
)


# ================================================
# TASK 8 — STANDARDIZATION
# ================================================


# Select columns to standardize

columns_to_scale = [
    "age",
    "fare"
]


# --------------------------------
# Before standardization
# --------------------------------

print("\n========== BEFORE STANDARDIZATION ==========")

print(
    df[columns_to_scale].agg(
        ["mean", "std"]
    )
)


# --------------------------------
# Create StandardScaler
# --------------------------------

scaler = StandardScaler()


# Fit and transform age and fare
# This is only an EDA experiment.
# It is NOT used in the modeling pipeline.

scaled_values = scaler.fit_transform(
    df[columns_to_scale]
)


# Create new columns

df["age_standardized"] = scaled_values[:, 0]

df["fare_standardized"] = scaled_values[:, 1]


# --------------------------------
# After standardization
# --------------------------------

print("\n========== AFTER STANDARDIZATION ==========")

print(
    df[
        [
            "age_standardized",
            "fare_standardized"
        ]
    ].agg(
        ["mean", "std"]
    )
)


# --------------------------------
# More precise verification
# --------------------------------

print("\n========== STANDARDIZATION CHECK ==========")

print(
    "Age standardized mean:",
    df["age_standardized"].mean()
)

print(
    "Age standardized std:",
    df["age_standardized"].std()
)

print(
    "Fare standardized mean:",
    df["fare_standardized"].mean()
)

print(
    "Fare standardized std:",
    df["fare_standardized"].std()
)


# --------------------------------
# Save final cleaned EDA dataset
# --------------------------------

df.to_csv(
    "analytics/titanic_cleaned.csv",
    index=False
)

print("\nUpdated cleaned dataset saved to:")
print("analytics/titanic_cleaned.csv")