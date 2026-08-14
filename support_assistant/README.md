# Zepto Support Assistant

An offline RAG-based support service for Zepto policies using **Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI**.

By default, the project runs with `MOCK_LLM=1`, so no external LLM API is required.

## Architecture

```text
8 Zepto Documents
       │
       ▼
create_docs.py → ingest.py
       │
       ▼
all-MiniLM-L6-v2
       │
       ▼
ChromaDB: zepto_policies
       │
       ▼
    User Query
       │
       ▼
 classify_intent
    │          │
    ▼          ▼
  Policy     General
    │          │
    ▼          ▼
retrieve_     direct_
and_answer    answer
    │
    ▼
Top 3 ChromaDB chunks
    │
    ▼
Pydantic JSON Response
```

## How It Works

### 1. Ingestion

`create_docs.py` creates 8 Zepto policy documents in `docs/`.

`ingest.py` reads the documents, creates one chunk per document, generates embeddings, and stores them in ChromaDB.

### 2. Embeddings

The local Sentence Transformer model `all-MiniLM-L6-v2` generates embeddings.

No embedding API is required.

### 3. Vector Database

Embeddings are stored in the ChromaDB collection:

```text
zepto_policies
```

All 8 documents are embedded and stored.

### 4. Retrieval

For policy questions, `retrieve_and_answer`:

1. Embeds the user query.
2. Searches ChromaDB.
3. Retrieves the top 3 most similar chunks.
4. Generates an answer using the retrieved context.

## LangGraph

The graph contains three nodes:

```text
classify_intent
      │
      ├── policy_question ──► retrieve_and_answer
      │
      └── general_question ─► direct_answer
```

Policy keywords include:

`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, and `support hours`.

## MOCK_LLM

When `MOCK_LLM` is unset or set to `1`:

* Intent classification uses keyword heuristics.
* ChromaDB retrieval works normally.
* Policy answers use deterministic mock logic.
* General questions receive a fixed response.
* No LLM API call is made.

When `MOCK_LLM=0`, the optional Groq LLM path is used.

## Structured Prompt

`main.py` contains `PROMPT_TEMPLATE` with:

* Role
* Context
* Task
* Format
* Length

It also includes a negative constraint requiring answers to use only the supplied context and a few-shot example.

## Structured Output

Responses are validated using the Pydantic `AnswerResponse` model:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

`confidence` is restricted to `0–1`.

The optional real-LLM path retries invalid responses up to two additional times.

## FastAPI

### Endpoint

```text
POST /ask
```

### Request

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

### Local API

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Verified Policy Example

**Query:**

```text
What is the delivery fee for orders below INR 149?
```

**Mock Response:**

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pincodes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}
```

The query was routed to `policy_question` and processed using ChromaDB retrieval.

## Verified General Example

**Query:**

```text
What is the capital of India?
```

**Response:**

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

The query was routed to `general_question` and handled by `direct_answer`.

## Local Run

Run the following commands from the repository root unless otherwise stated.

Navigate to the `support_assistant` directory:

```powershell
cd support_assistant
python -m pip install -r requirements.txt
python create_docs.py
python ingest.py
uvicorn main:app --reload
```

Local API will be available at:

```text
http://127.0.0.1:8000
```

## Docker

From the repository root:

```powershell
docker build -t zepto-support ./support_assistant
docker run -p 7860:7860 zepto-support
```

Docker API:

```text
http://localhost:7860
```

Docker uses `MOCK_LLM=1` by default and requires no LLM API key.

## Tech Stack

* **Python** — Core language
* **Sentence Transformers** — Local embeddings
* **ChromaDB** — Vector database
* **LangGraph** — Workflow and routing
* **Pydantic** — Structured output validation
* **FastAPI** — REST API
* **Groq** — Optional LLM
* **Docker** — Containerization

## Key Features

* Offline RAG pipeline
* Local embedding model
* ChromaDB semantic retrieval
* LangGraph intent routing
* Deterministic mock mode
* Optional Groq LLM
* Pydantic structured responses
* FastAPI API
* Docker support
