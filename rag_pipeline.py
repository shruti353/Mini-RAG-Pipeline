import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Load embedding model
# ------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------
# Chunking Function
# ------------------------------
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)

    return chunks


# ------------------------------
# Embedding Function
# ------------------------------
def get_embeddings(chunks):
    return embedder.encode(chunks)


# ------------------------------
# Embedding Based Search
# ------------------------------
def embedding_search(query, chunks, k=3):
    query_emb = embedder.encode([query])
    chunk_embs = embedder.encode(chunks)

    similarities = cosine_similarity(query_emb, chunk_embs)[0]

    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_k_idx]


# ------------------------------
# BM25 Keyword Search
# ------------------------------
def bm25_search(query, chunks, k=3):
    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())

    top_k_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_k_idx]


# ------------------------------
# RAG Runner (Optional)
# ------------------------------
def run_rag(query, documents):
    # Combine documents into chunks
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    # Embedding search
    retrieved = embedding_search(query, all_chunks, k=3)

    return retrieved
