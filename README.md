# Zepto Data & AI Platform

## Project Overview

This repository contains a comprehensive capstone project demonstrating three key data and AI engineering capabilities:

1. **Data Pipeline** - Web scraping, data cleaning, and SQL database operations
2. **Analytics Pipeline** - Exploratory data analysis, machine learning modeling, and evaluation
3. **GenAI Support Assistant** - RAG-based support system with LangGraph and ChromaDB

This project showcases end-to-end skills in data acquisition, transformation, analysis, predictive modeling, and generative AI applications.

## Repository Structure

```
zepto-data-ai-platform/
├── data_pipeline/
│   ├── scraper.py
│   ├── cleaner.py
│   ├── database.py
│   ├── queries.py
│   ├── compare_join.py
│   ├── raw_books.csv
│   ├── cleaned_books.csv
│   ├── zepto_books.db
│   ├── query_outputs/
│   └── README.md
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   ├── charts/
│   ├── artifacts/
│   └── README.md
├── support_assistant/
│   ├── main.py
│   ├── ingest.py
│   ├── create_docs.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── docs/
│   ├── chroma_db/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

### Module Descriptions

- **data_pipeline/** - Scrapes book data from Books to Scrape, cleans and converts pricing, loads into normalized SQLite database, and demonstrates SQL queries and pandas operations
- **analytics/** - Performs EDA and predictive modeling on the Titanic dataset with visualization, feature engineering, multiple classifiers, and imbalanced data handling
- **support_assistant/** - RAG-based support assistant using ChromaDB vector store, LangGraph routing, and FastAPI with Docker deployment support

## Module 1 — Data Pipeline

**Overview:**
- Scrapes 158 books across 4 categories from Books to Scrape website
- Uses `requests` and `BeautifulSoup` for web scraping
- Cleans data and converts GBP prices to INR using fixed rate (1 GBP = 105.50 INR)
- Loads data into normalized SQLite database with proper schema design
- Executes SQL queries using raw SQL and pandas `read_sql`
- Compares SQL joins with pandas `merge` operations

**Run Commands (from repository root):**
```bash
python data_pipeline/scraper.py
python data_pipeline/cleaner.py
python data_pipeline/database.py
python data_pipeline/queries.py
python data_pipeline/compare_join.py
```

See `data_pipeline/README.md` for detailed implementation information.

## Module 2 — Analytics Pipeline

**Overview:**
- Uses the Titanic dataset (loaded via `sns.load_dataset("titanic")` in `01_eda.py` with offline fallback to `analytics/titanic.csv`)
- **EDA Phase:** Missing value analysis, univariate/bivariate visualization, correlation analysis
  - Generates `analytics/titanic_cleaned.csv` as an EDA artifact (NOT used for modeling)
- **Modeling Phase:** 
  - Train/test split performed BEFORE preprocessing
  - Training-only imputation, encoding, and scaling to prevent data leakage
  - Trains Logistic Regression, Decision Tree, and Random Forest classifiers
  - Handles class imbalance using SMOTE (training-only application)
  - Hyperparameter tuning with GridSearchCV and OOB score evaluation
  - Includes regression side-task (fare prediction)
  - Saves complete pipeline to `analytics/artifacts/best_titanic_pipeline.joblib`

**Important Note:** The file `analytics/titanic_cleaned.csv` is generated during EDA as an intermediate artifact. The modeling script (`02_modeling.py`) loads the raw Titanic dataset independently and applies proper train/test splitting with training-only preprocessing to avoid data leakage.

**Run Commands (from repository root):**
```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

See `analytics/README.md` for detailed implementation information.

## Module 3 — Support Assistant

**Overview:**
- RAG-based support assistant using 8 policy documents
- Embeddings generated using SentenceTransformer (`all-MiniLM-L6-v2`)
- ChromaDB vector store for semantic search
- LangGraph StateGraph with routing logic:
  - `classify_intent` - Routes queries to retrieval or direct answer
  - `retrieve_and_answer` - RAG pipeline for policy-related questions
  - `direct_answer` - Handles greetings and general queries
- Pydantic response schema for structured output
- FastAPI REST API with `POST /ask` endpoint
- Runs in MOCK_LLM mode by default (offline baseline for grading)
- Docker deployment support

**Setup and Run (from repository root):**

First, navigate to the support_assistant directory:
```bash
cd support_assistant
```

Then run setup and server:
```bash
python create_docs.py
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

**Docker Deployment (from repository root):**
```bash
docker build -t zepto-support ./support_assistant
docker run -p 7860:7860 zepto-support
```

**Notes:**
- `MOCK_LLM=1` (default) enables offline mock mode for baseline verification
- Real LLM integration via Groq API is an optional extension (requires `GROQ_API_KEY`)
- Hugging Face Spaces deployment is an optional extension

See `support_assistant/README.md` for detailed implementation information.

## Installation

**Core Dependencies (Modules 1 & 2):**
```bash
pip install -r requirements.txt
```

**Module 3 Additional Dependencies:**
```bash
cd support_assistant
pip install -r requirements.txt
```

Or install all dependencies from root:
```bash
pip install -r requirements.txt
pip install -r support_assistant/requirements.txt
```

## Verification

**Module 1 - Data Pipeline:**
- Successfully scrapes 158 books across 4 categories
- Converts prices using fixed rate (1 GBP = 105.50 INR)
- Creates normalized SQLite database with proper schema
- Executes SQL queries and pandas operations successfully

**Module 2 - Analytics Pipeline:**
- Completes EDA with comprehensive visualizations in `analytics/charts/`
- Generates `analytics/titanic_cleaned.csv` during EDA phase
- Modeling uses independent raw data load with proper train/test split
- Training-only preprocessing prevents data leakage
- Saves complete pipeline to `analytics/artifacts/best_titanic_pipeline.joblib`

**Module 3 - Support Assistant:**
- Successfully creates 8 policy documents
- Ingests documents into ChromaDB vector store
- FastAPI server runs on port 7860 (local and Docker)
- MOCK_LLM mode verified for offline operation

## Notes

- Each module contains its own detailed README with implementation specifics
- **MOCK_LLM mode (default)** is the graded offline baseline for Module 3
- Real LLM integration and Hugging Face deployment are **optional extensions**, not required
- Module 2 modeling uses proper train/test split with training-only preprocessing to avoid data leakage
- `analytics/titanic_cleaned.csv` is an EDA artifact and is NOT used as modeling input

