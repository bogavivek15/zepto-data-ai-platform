import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETUP
# ============================================================

os.makedirs("analytics/charts", exist_ok=True)

# Load raw Titanic dataset exactly once
df = sns.load_dataset("titanic")

# Required offline fallback
df.to_csv(
    "analytics/titanic.csv",
    index=False
)

print("\nRaw dataset saved to:")
print("analytics/titanic.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== DATASET DESCRIBE ==========")
print(df.describe())


# ============================================================
# MISSING VALUES - BEFORE CLEANING
# ============================================================

missing = df.isnull().sum()
missing = missing[missing > 0]

missing_percentage = (
    missing / len(df) * 100
).sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage
})

print("\n========== MISSING VALUES ==========")
print(missing_report)


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\n========== CLEANING ==========")

# Age: 19.865% -> 5%-30% -> median imputation
age_median = df["age"].median()

df["age"] = df["age"].fillna(
    age_median
)

print(
    f"age: 19.8653% missing -> median imputation. "
    f"Median = {age_median:.2f}"
)

# Embarked: 0.224% -> under 5% -> drop affected rows
embarked_missing = df["embarked"].isna().sum()

df = df.dropna(
    subset=["embarked"]
)

print(
    f"embarked: 0.2245% missing -> "
    f"dropped {embarked_missing} rows."
)

# embark_town has the same missing rows as embarked.
# It is redundant with embarked, so drop it.
if "embark_town" in df.columns:
    df = df.drop(
        columns=["embark_town"]
    )

print(
    "embark_town: redundant with embarked -> column dropped."
)

# Deck: 77.2166% missing.
# Imputation is unreliable, therefore drop the column.
deck_missing_percentage = (
    missing_percentage.get(
        "deck",
        0
    )
)

if "deck" in df.columns:
    df = df.drop(
        columns=["deck"]
    )

print(
    f"deck: {deck_missing_percentage:.4f}% missing -> "
    "column dropped because missingness is too high for reliable imputation."
)


# ============================================================
# CLEANED DATASET
# ============================================================

print("\n========== CLEANED DATASET ==========")
print("Shape:", df.shape)

print("\nRemaining missing values:")
print(
    df.isnull().sum()
)

df.to_csv(
    "analytics/titanic_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved to:")
print("analytics/titanic_cleaned.csv")


# ============================================================
# UNIVARIATE ANALYSIS - AGE
# ============================================================

print("\n========== AGE IQR ANALYSIS ==========")

age_q1 = df["age"].quantile(0.25)
age_q3 = df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = df[
    (df["age"] < age_lower) |
    (df["age"] > age_upper)
]

print("Q1:", age_q1)
print("Q3:", age_q3)
print("IQR:", age_iqr)
print("Lower limit:", age_lower)
print("Upper limit:", age_upper)
print(
    "Number of outliers:",
    len(age_outliers)
)


plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=30)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(
    "analytics/charts/age_histogram.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 5))
plt.boxplot(df["age"])
plt.ylabel("Age")
plt.title("Age Box Plot")
plt.tight_layout()
plt.savefig(
    "analytics/charts/age_boxplot.png",
    dpi=300
)
plt.close()


# ============================================================
# UNIVARIATE ANALYSIS - FARE
# ============================================================

print("\n========== FARE IQR ANALYSIS ==========")

fare_q1 = df["fare"].quantile(0.25)
fare_q3 = df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = df[
    (df["fare"] < fare_lower) |
    (df["fare"] > fare_upper)
]

print("Q1:", fare_q1)
print("Q3:", fare_q3)
print("IQR:", fare_iqr)
print("Lower limit:", fare_lower)
print("Upper limit:", fare_upper)
print(
    "Number of outliers:",
    len(fare_outliers)
)


fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().tolist()

print("\n========== FARE STATISTICS ==========")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

if fare_mean > fare_median > fare_mode[0]:
    skew_text = "right-skewed"
elif fare_mean < fare_median < fare_mode[0]:
    skew_text = "left-skewed"
else:
    skew_text = "not clearly determined by mean/median/mode ordering"

print("\nFare distribution:", skew_text)


plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=30)
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig(
    "analytics/charts/fare_histogram.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 5))
plt.boxplot(df["fare"])
plt.ylabel("Fare")
plt.title("Fare Box Plot")
plt.tight_layout()
plt.savefig(
    "analytics/charts/fare_boxplot.png",
    dpi=300
)
plt.close()


# ============================================================
# BIVARIATE ANALYSIS
# ============================================================

print("\n========== SURVIVAL RATE BY SEX ==========")

survival_by_sex = (
    df.groupby(
        "sex",
        observed=True
    )["survived"]
    .mean()
    .mul(100)
)

print(survival_by_sex)


print("\n========== SURVIVAL RATE BY PCLASS ==========")

survival_by_pclass = (
    df.groupby(
        "pclass"
    )["survived"]
    .mean()
    .mul(100)
)

print(survival_by_pclass)


print(
    "\n========== SURVIVAL RATE BY SEX AND PCLASS =========="
)

survival_by_sex_pclass = (
    df.groupby(
        ["sex", "pclass"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
)

print(survival_by_sex_pclass)


# Boolean masking
female_first = df[
    (df["sex"] == "female") &
    (df["pclass"] == 1)
]

male_third = df[
    (df["sex"] == "male") &
    (df["pclass"] == 3)
]

print("\n========== BOOLEAN MASKING ==========")
print(
    "Female passengers in first class:",
    len(female_first)
)
print(
    "Male passengers in third class:",
    len(male_third)
)


# ============================================================
# CORRELATION MATRIX - EXACTLY SIX REQUIRED COLUMNS
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title(
    "Titanic Numeric Feature Correlation"
)

plt.tight_layout()

plt.savefig(
    "analytics/charts/correlation_heatmap.png",
    dpi=300
)

plt.close()


# Find two strongest absolute off-diagonal correlations
pairs = []

for i in range(
    len(correlation_columns)
):
    for j in range(i + 1, len(correlation_columns)):

        feature_a = correlation_columns[i]
        feature_b = correlation_columns[j]

        value = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        pairs.append(
            (
                feature_a,
                feature_b,
                value,
                abs(value)
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)

print(
    "\n========== TWO STRONGEST CORRELATIONS =========="
)

for feature_a, feature_b, value, absolute_value in pairs[:2]:

    print(
        f"{feature_a} <-> {feature_b}: "
        f"{value:.4f}"
    )


# ============================================================
# MULTIVARIATE DATA STORY - CHART 1
# ============================================================

plt.figure(figsize=(8, 5))

survival_by_sex.plot(
    kind="bar"
)

plt.ylabel("Survival Rate (%)")
plt.xlabel("Sex")
plt.title("Survival Rate by Sex")
plt.xticks(
    rotation=0
)
plt.tight_layout()

plt.savefig(
    "analytics/charts/survival_by_sex.png",
    dpi=300
)

plt.close()


print(
    "\n========== CHART 1 INTERPRETATION =========="
)

print(
    "Women had substantially higher survival rates than men. "
    "This indicates a strong association between sex and survival. "
    "However, sex alone does not explain differences within passenger classes."
)


# ============================================================
# CHART 2 - PCLASS
# ============================================================

plt.figure(figsize=(8, 5))

survival_by_pclass.plot(
    kind="bar"
)

plt.ylabel("Survival Rate (%)")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Passenger Class")
plt.xticks(
    rotation=0
)
plt.tight_layout()

plt.savefig(
    "analytics/charts/survival_by_pclass.png",
    dpi=300
)

plt.close()


print(
    "\n========== CHART 2 INTERPRETATION =========="
)

print(
    "First-class passengers had the highest survival rate, "
    "followed by second- and third-class passengers. "
    "This suggests passenger class was strongly associated with survival."
)


# ============================================================
# CHART 3 - SEX + PCLASS
# ============================================================

plot_data = (
    df.groupby(
        ["sex", "pclass"],
        observed=True
    )["survived"]
    .mean()
    .mul(100)
    .reset_index()
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=plot_data,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()

plt.savefig(
    "analytics/charts/survival_sex_pclass.png",
    dpi=300
)

plt.close()


print(
    "\n========== CHART 3 INTERPRETATION =========="
)

print(
    "Female passengers had higher survival rates than male passengers "
    "within each passenger class. First- and second-class women had especially "
    "high survival rates, while third-class men had the lowest survival rate."
)


# ============================================================
# CHART 4 - AGE / FARE / SURVIVAL
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age, Fare and Survival")
plt.tight_layout()

plt.savefig(
    "analytics/charts/age_fare_survival.png",
    dpi=300
)

plt.close()


print(
    "\n========== CHART 4 INTERPRETATION =========="
)

print(
    "Survivors are distributed across a wide range of ages and fares. "
    "Higher fares are associated with higher passenger classes, so fare also "
    "captures part of the socioeconomic differences associated with survival."
)


# ============================================================
# CHART 5 - FARE BY SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="fare"
)

plt.xlabel("Survived")
plt.ylabel("Fare")
plt.title("Fare Distribution by Survival")
plt.tight_layout()

plt.savefig(
    "analytics/charts/fare_by_survival.png",
    dpi=300
)

plt.close()


print(
    "\n========== CHART 5 INTERPRETATION =========="
)

print(
    "Survivors generally paid higher fares than passengers who did not survive. "
    "Both groups contain substantial variation and extreme fare values, but the "
    "difference supports the positive association between fare and survival."
)


# ============================================================
# EXPLORATORY STANDARDIZATION
# ============================================================

print(
    "\n========== STANDARDIZATION CHECK =========="
)

standardizer = StandardScaler()

standardized_values = standardizer.fit_transform(
    df[["age", "fare"]]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=[
        "age_standardized",
        "fare_standardized"
    ]
)

print("\nBefore standardization:")
print(
    df[
        ["age", "fare"]
    ].agg(
        ["mean", "std"]
    )
)

print("\nAfter standardization:")
print(
    standardized_df.agg(
        ["mean", "std"]
    )
)

print(
    "\nThe standardized age and fare columns have approximately "
    "zero mean and unit standard deviation."
)


print("\n========== EDA COMPLETE ==========")
print(
    "Cleaned dataset:",
    "analytics/titanic_cleaned.csv"
)
print(
    "Charts directory:",
    "analytics/charts/"
)
