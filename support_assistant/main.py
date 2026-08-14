import os
import json
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END


# =========================================================
# SETTINGS
# =========================================================

# MOCK_LLM is ON by default.
# Only when MOCK_LLM=0 will the optional real LLM path run.

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# =========================================================
# EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# =========================================================
# PYDANTIC MODELS
# =========================================================

class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class SupportState(TypedDict, total=False):

    query: str

    intent: str

    retrieved_documents: list[str]

    retrieved_ids: list[str]

    answer: str

    sources: list[str]

    confidence: float


# =========================================================
# STRUCTURED PROMPT
# =========================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the Zepto policy information provided below.

TASK:
Answer the customer's question using only the provided context.

FORMAT:
Return valid JSON with exactly these fields:
{
    "answer": "string",
    "sources": ["document_id"],
    "confidence": 0.0
}

LENGTH:
Keep the answer short and clear, preferably 1 to 3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies.
If the context does not contain the answer, clearly say that the information is not available.

FEW-SHOT EXAMPLE:

Question:
How much is the delivery fee for orders below INR 149?

Context:
Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.

Answer:
{
    "answer": "Orders below INR 149 have a flat INR 25 delivery fee.",
    "sources": ["doc_01"],
    "confidence": 1.0
}

CUSTOMER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""


# =========================================================
# NODE 1 - CLASSIFY INTENT
# =========================================================

def classify_intent(state: SupportState):

    query = state["query"]

    query_lower = query.lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    # -----------------------------------------------------
    # MOCK MODE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        for keyword in policy_keywords:

            if keyword in query_lower:

                return {
                    "intent": "policy_question"
                }

        return {
            "intent": "general_question"
        }

    # -----------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # -----------------------------------------------------

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = f"""
Classify this question as exactly one of:

policy_question
general_question

Question:
{query}

Return only one label.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    if "policy_question" in result:
        return {
            "intent": "policy_question"
        }

    return {
        "intent": "general_question"
    }


# =========================================================
# NODE 2 - RETRIEVE AND ANSWER
# =========================================================

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    # -----------------------------------------------------
    # 1. Convert query into embedding
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()


    # -----------------------------------------------------
    # 2. Search ChromaDB
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )


    # -----------------------------------------------------
    # 3. Get retrieved documents
    # -----------------------------------------------------

    retrieved_documents = results["documents"][0]

    retrieved_ids = results["ids"][0]


    # -----------------------------------------------------
    # MOCK MODE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        top_chunk = retrieved_documents[0]

        # First approximately 200 characters
        snippet = top_chunk[:200]

        answer = (
            f"Based on the retrieved context: {snippet}"
        )

        return {
            "retrieved_documents": retrieved_documents,
            "retrieved_ids": retrieved_ids,
            "answer": answer,
            "sources": retrieved_ids,
            "confidence": 1.0
        }


    # -----------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # -----------------------------------------------------

    context = "\n\n".join(
        retrieved_documents
    )

    prompt = PROMPT_TEMPLATE.format(
        question=query,
        context=context
    )

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    last_error = ""

    # Try maximum 3 times
    for attempt in range(3):

        if attempt == 0:

            current_prompt = prompt

        else:

            current_prompt = prompt + f"""

Your previous output was invalid.

Validation error:
{last_error}

Return ONLY valid JSON matching this schema:

{{
    "answer": "string",
    "sources": ["document_id"],
    "confidence": 0.0
}}
"""


        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": current_prompt
                }
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content

        try:

            # Remove possible markdown fences
            raw_output = raw_output.replace(
                "```json", ""
            ).replace(
                "```", ""
            ).strip()

            data = json.loads(raw_output)

            validated = AnswerResponse(**data)

            return {
                "retrieved_documents": retrieved_documents,
                "retrieved_ids": retrieved_ids,
                "answer": validated.answer,
                "sources": validated.sources,
                "confidence": validated.confidence
            }

        except Exception as error:

            last_error = str(error)


    # -----------------------------------------------------
    # If all retries fail
    # -----------------------------------------------------

    return {
        "retrieved_documents": retrieved_documents,
        "retrieved_ids": retrieved_ids,
        "answer": "ERROR: The LLM response could not be validated.",
        "sources": retrieved_ids,
        "confidence": 0.0
    }


# =========================================================
# NODE 3 - DIRECT ANSWER
# =========================================================

def direct_answer(state: SupportState):

    query = state["query"]

    # -----------------------------------------------------
    # MOCK MODE
    # -----------------------------------------------------

    if MOCK_LLM != "0":

        return {
            "answer": (
                "I can only answer questions about Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0
        }


    # -----------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # -----------------------------------------------------

    prompt = f"""
You are a Zepto customer support assistant.

Answer the following question briefly.

Question:
{query}

Return valid JSON:

{{
    "answer": "string",
    "sources": [],
    "confidence": 0.0
}}
"""

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    last_error = ""

    for attempt in range(3):

        if attempt == 0:

            current_prompt = prompt

        else:

            current_prompt = prompt + f"""

Your previous response failed validation.

Error:
{last_error}

Return only valid JSON.
"""


        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": current_prompt
                }
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content

        try:

            raw_output = raw_output.replace(
                "```json", ""
            ).replace(
                "```", ""
            ).strip()

            data = json.loads(raw_output)

            validated = AnswerResponse(**data)

            return {
                "answer": validated.answer,
                "sources": [],
                "confidence": validated.confidence
            }

        except Exception as error:

            last_error = str(error)


    return {
        "answer": "ERROR: The LLM response could not be validated.",
        "sources": [],
        "confidence": 0.0
    }


# =========================================================
# LANGGRAPH
# =========================================================

graph_builder = StateGraph(SupportState)


# Add the 3 required nodes

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


# Start -> classify

graph_builder.add_edge(
    START,
    "classify_intent"
)


# =========================================================
# CONDITIONAL ROUTING
# =========================================================

def route_question(state: SupportState):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)


# Both paths finish

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


# Compile graph

graph = graph_builder.compile()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Zepto Support Assistant"
)


@app.get("/")
def home():

    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask_question(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query
        }
    )

    response = AnswerResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )

    return response