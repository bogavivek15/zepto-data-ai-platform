import os

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load embedding model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# -----------------------------
# 2. Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------
# 3. Read documents
# -----------------------------

documents = []
ids = []

for filename in sorted(os.listdir("docs")):

    if filename.endswith(".txt"):

        file_path = os.path.join("docs", filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)

        # doc_01.txt -> doc_01
        document_id = filename.replace(".txt", "")

        ids.append(document_id)


# -----------------------------
# 4. Create embeddings
# -----------------------------

print("Creating embeddings...")

embeddings = model.encode(documents).tolist()

print("Embeddings created.")


# -----------------------------
# 5. Store in ChromaDB
# -----------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents stored in ChromaDB.")
print("Total documents:", collection.count())