import sqlite3
import pandas as pd


# -----------------------------
# Read cleaned data
# -----------------------------

df = pd.read_csv(
    "data_pipeline/cleaned_books.csv"
)


# -----------------------------
# Connect to SQLite database
# -----------------------------

conn = sqlite3.connect(
    "data_pipeline/zepto_books.db"
)

cursor = conn.cursor()


# Enable foreign keys
cursor.execute(
    "PRAGMA foreign_keys = ON"
)


# -----------------------------
# Remove old tables
# -----------------------------

cursor.execute(
    "DROP TABLE IF EXISTS books"
)

cursor.execute(
    "DROP TABLE IF EXISTS categories"
)


# -----------------------------
# Create categories table
# -----------------------------

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
)
""")


# -----------------------------
# Create books table
# -----------------------------

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# -----------------------------
# Create category DataFrame
# -----------------------------

categories_df = pd.DataFrame({
    "category_name": df["category"].unique()
})


categories_df["category_id"] = range(
    1,
    len(categories_df) + 1
)


categories_df = categories_df[
    [
        "category_id",
        "category_name"
    ]
]


# -----------------------------
# Insert categories
# -----------------------------

categories_df.to_sql(
    "categories",
    conn,
    if_exists="append",
    index=False
)


# -----------------------------
# Create category ID mapping
# -----------------------------

category_mapping = dict(
    zip(
        categories_df["category_name"],
        categories_df["category_id"]
    )
)


df["category_id"] = df["category"].map(
    category_mapping
)


# -----------------------------
# Prepare books DataFrame
# -----------------------------

books_df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id"
    ]
].copy()


# SQLite stores boolean values as 0 or 1
books_df["in_stock"] = (
    books_df["in_stock"].astype(int)
)


# -----------------------------
# Insert books
# -----------------------------

books_df.to_sql(
    "books",
    conn,
    if_exists="append",
    index=False
)


# Save changes
conn.commit()


# -----------------------------
# Verify database
# -----------------------------

print("\nCategories:")
print(
    pd.read_sql(
        "SELECT * FROM categories",
        conn
    )
)


print("\nFirst 5 books:")
print(
    pd.read_sql(
        "SELECT * FROM books LIMIT 5",
        conn
    )
)


print("\nTotal books:")

print(
    pd.read_sql(
        "SELECT COUNT(*) AS total_books FROM books",
        conn
    )
)


# Close database
conn.close()


print("\nDatabase created successfully:")
print("data_pipeline/zepto_books.db")