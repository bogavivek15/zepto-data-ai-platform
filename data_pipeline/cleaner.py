import pandas as pd


# Read raw scraped data
df = pd.read_csv("data_pipeline/raw_books.csv")


# -----------------------------
# Clean price
# -----------------------------

df["price_gbp"] = pd.to_numeric(
    df["price"].str.replace("£", "", regex=False),
    errors="coerce"
)


# If price parsing fails,
# replace missing values with the median price
if df["price_gbp"].isna().any():

    median_price = df["price_gbp"].median()

    df["price_gbp"] = df["price_gbp"].fillna(
        median_price
    )


# -----------------------------
# Clean star rating
# -----------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


df["rating"] = df["star_rating"].map(
    rating_map
)


# If rating parsing fails,
# replace missing values with median rating
if df["rating"].isna().any():

    median_rating = df["rating"].median()

    df["rating"] = df["rating"].fillna(
        median_rating
    )


df["rating"] = df["rating"].astype(int)


# -----------------------------
# Clean availability
# -----------------------------

df["in_stock"] = (
    df["availability"] == "In stock"
)


# -----------------------------
# Convert GBP to INR
# -----------------------------

GBP_TO_INR = 105.50


df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
).round(2)


# -----------------------------
# Display results
# -----------------------------

print("\nCleaned data:")
print(df.head())


print("\nData types:")
print(df.dtypes)


print("\nMissing values:")
print(df.isna().sum())


# -----------------------------
# Save cleaned data
# -----------------------------

df.to_csv(
    "data_pipeline/cleaned_books.csv",
    index=False
)


print("\nCleaned data saved to:")
print("data_pipeline/cleaned_books.csv")