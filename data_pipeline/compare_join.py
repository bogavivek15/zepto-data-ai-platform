import sqlite3
import pandas as pd


# Connect to database
conn = sqlite3.connect(
    "data_pipeline/zepto_books.db"
)


# --------------------------------
# Read books table into pandas
# --------------------------------

books_df = pd.read_sql(
    "SELECT * FROM books",
    conn
)


# --------------------------------
# Read categories table into pandas
# --------------------------------

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    conn
)


# --------------------------------
# SQL JOIN
# --------------------------------

sql_query = """
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
"""

sql_result = pd.read_sql(
    sql_query,
    conn
)


# --------------------------------
# Pandas JOIN using pd.merge()
# --------------------------------

pandas_result = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


# Keep the same columns as SQL result
pandas_result = pandas_result[
    [
        "book_id",
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_name"
    ]
]


# --------------------------------
# Sort both results
# --------------------------------

sql_result = sql_result.sort_values(
    "book_id"
).reset_index(drop=True)


pandas_result = pandas_result.sort_values(
    "book_id"
).reset_index(drop=True)


# --------------------------------
# Display results
# --------------------------------

print("\n========== SQL JOIN ==========")

print(sql_result.head(10))


print("\n========== PANDAS MERGE ==========")

print(pandas_result.head(10))


# --------------------------------
# Compare both results
# --------------------------------

print("\n========== COMPARISON ==========")

print(
    "Are both results equivalent?",
    sql_result.equals(pandas_result)
)


print("\nSQL rows:", len(sql_result))
print("Pandas rows:", len(pandas_result))


# Close database
conn.close()