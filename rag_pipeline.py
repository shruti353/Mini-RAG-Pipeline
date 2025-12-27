import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Set OpenRouter key (keep env variable for security)
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# 1. Chunking
# ----------------------------
def chunk_text(text, chunk_size=250, overlap=40):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ----------------------------
# 2. Build FAISS Index
# ----------------------------
def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# ----------------------------
# 3. Retrieval
# ----------------------------
def retrieve(query, chunks, index, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


# ----------------------------
# 4. Answer Generation
# ----------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant for a construction marketplace.
Answer ONLY using the information in the provided context.
If the answer cannot be found, reply: "Information not available."

Context:
{context}

Question: {query}

Answer:
"""

    response = openai.ChatCompletion.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"].strip()


# ----------------------------
# 5. End-to-End Pipeline
# ----------------------------
def run_rag(query, documents):
    all_chunks = []

    # chunk all documents
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    # build vector index
    index, _ = build_faiss_index(all_chunks)

    # retrieve relevant chunks
    retrieved_chunks = retrieve(query, all_chunks, index)

    # generate grounded answer
    answer = generate_answer(query, retrieved_chunks)

    return retrieved_chunks, answer
