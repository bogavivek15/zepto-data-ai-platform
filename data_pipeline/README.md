# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data pipeline for scraping, cleaning, converting, storing, and querying book catalogue data from Books to Scrape.

The pipeline follows this flow:

Scrape → Clean → Convert → Store → Query → Compare

## Data Source

The data is scraped from:

https://books.toscrape.com/

The scraper starts from the main website URL and automatically discovers the required category links.

The following four categories are scraped:

- Mystery
- Historical Fiction
- Romance
- Fiction

Pagination is handled automatically for each category.

The final dataset contains 158 books.

## Scraping

The scraping process uses:

- `requests`
- `BeautifulSoup`

The scraper collects:

- title
- price
- star rating
- availability
- category

The raw scraped data is saved as:

`raw_books.csv`

## Data Cleaning

The raw fields are converted into clean fields.

### Price

The pound symbol is removed from the scraped price and the value is converted to a numeric `price_gbp` column.

Invalid numeric values are converted to missing values using `errors="coerce"`.

If a numeric price value cannot be parsed, the median price is used for imputation.

### Rating

The text star rating is converted to an integer:

- One → 1
- Two → 2
- Three → 3
- Four → 4
- Five → 5

If a rating cannot be parsed, the median rating is used for imputation.

### Availability

The availability text is converted into a boolean `in_stock` column.

- `In stock` → `True`
- Other availability values → `False`

### Currency Conversion

The project-defined fixed conversion rate is:

**1 GBP = 105.50 INR**

The `price_inr` column is calculated using:

`price_inr = price_gbp × 105.50`

No external currency API is used.

## Database Design

The cleaned data is stored in a SQLite database:

`zepto_books.db`

The database contains two normalized tables.

### categories

| Column | Type | Description |
|---|---|---|
| category_id | INTEGER PRIMARY KEY | Unique category identifier |
| category_name | TEXT UNIQUE | Category name |

### books

| Column | Type | Description |
|---|---|---|
| book_id | INTEGER PRIMARY KEY | Unique book identifier |
| title | TEXT | Book title |
| price_gbp | REAL | Price in GBP |
| price_inr | REAL | Converted price in INR |
| rating | INTEGER | Rating from 1 to 5 |
| in_stock | INTEGER | Stock status stored as 0 or 1 in SQLite |
| category_id | INTEGER | Foreign key referencing categories |

The relationship is:

`books.category_id → categories.category_id`

This avoids repeatedly storing category names for every book.

## SQL Queries

Six SQL queries are included.

1. SELECT with WHERE
2. ORDER BY
3. LIMIT
4. DISTINCT
5. BETWEEN
6. JOIN

The SQL queries and their outputs are saved in:

`query_outputs/query_results.txt`

## Pandas SQL and JOIN Comparison

At least two SQL query results are read into pandas using `pd.read_sql()`.

The SQL JOIN result is also reproduced using:

`pd.merge()`

The SQL JOIN and pandas merge produce equivalent results for all 158 books.

## Project Files

- `scraper.py` — scrapes the website and saves raw data.
- `cleaner.py` — cleans and converts the scraped data.
- `database.py` — creates the SQLite database and loads the cleaned data.
- `queries.py` — executes the required SQL queries and saves their outputs.
- `compare_join.py` — compares the SQL JOIN with the pandas merge.
- `raw_books.csv` — raw scraped data.
- `cleaned_books.csv` — cleaned and converted data.
- `zepto_books.db` — SQLite database.
- `query_outputs/query_results.txt` — SQL queries and outputs.

## How to Run

From the project root:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape the data

```bash
python data_pipeline/scraper.py
```

### 3. Clean the data

```bash
python data_pipeline/cleaner.py
```

### 4. Create the database

```bash
python data_pipeline/database.py
```

### 5. Run SQL queries

```bash
python data_pipeline/queries.py
```

### 6. Compare SQL JOIN and pandas merge

```bash
python data_pipeline/compare_join.py
```

## Result

The completed pipeline produces 158 cleaned book records across four categories and stores them in a normalized SQLite database with a primary-key/foreign-key relationship.

## Data Quality

The final dataset contains 158 books and 0 missing values in the cleaned columns.

The cleaned columns have the following types:

- `price_gbp` — float
- `rating` — integer
- `in_stock` — boolean
- `price_inr` — float

## Reproducibility

The pipeline can be regenerated from the main Books to Scrape website using the provided Python scripts. The SQLite database can also be recreated from the cleaned CSV data.