import sqlite3
import pandas as pd


# Connect to database
conn = sqlite3.connect(
    "data_pipeline/zepto_books.db"
)


# SQL queries
queries = {

    "QUERY 1 - SELECT + WHERE": """
SELECT *
FROM books
WHERE rating >= 4
""",

    "QUERY 2 - ORDER BY": """
SELECT *
FROM books
ORDER BY price_gbp DESC
""",

    "QUERY 3 - LIMIT": """
SELECT *
FROM books
ORDER BY price_gbp DESC
LIMIT 10
""",

    "QUERY 4 - DISTINCT": """
SELECT DISTINCT category_id
FROM books
""",

    "QUERY 5 - BETWEEN": """
SELECT *
FROM books
WHERE price_gbp BETWEEN 20 AND 40
""",

    "QUERY 6 - JOIN": """
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
}


# Open output file
with open(
    "data_pipeline/query_outputs/query_results.txt",
    "w",
    encoding="utf-8"
) as file:

    # Run every query
    for query_name, query in queries.items():

        print("\n" + "=" * 60)
        print(query_name)
        print("=" * 60)

        result = pd.read_sql(
            query,
            conn
        )

        # Print to terminal
        print(query)
        print(result)

        # Save query name
        file.write("\n")
        file.write("=" * 60)
        file.write("\n")
        file.write(query_name)
        file.write("\n")
        file.write("=" * 60)
        file.write("\n\n")

        # Save query
        file.write("SQL QUERY:\n")
        file.write(query.strip())
        file.write("\n\n")

        # Save output
        file.write("OUTPUT:\n")
        file.write(
            result.to_string(index=False)
        )
        file.write("\n\n")


# Close database
conn.close()


print("\nQuery results saved to:")
print("data_pipeline/query_outputs/query_results.txt")