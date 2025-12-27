import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------
# 0. OpenRouter Client
# ----------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ----------------------------
# 1. Embedding Model
# ----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ----------------------------
# 2. Chunk Cleaning
# ----------------------------
def clean_chunk(text):
    """Remove markdown artifacts, fix spacing, normalize formatting."""
    
    # Remove markdown headings (###, ##)
    text = re.sub(r'#\s*', '', text)
    text = re.sub(r'#+', '', text)

    # Fix bullet formatting
    text = re.sub(r'\s*-\s*', '\n- ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Restore bullet newlines
    text = text.replace('. -', '.\n-')

    return text.strip()


# ----------------------------
# 3. Chunking Function
# ----------------------------
def chunk_text(text, chunk_size=250, overlap=50):
    """Splits documents into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ----------------------------
# 4. FAISS Indexing
# ----------------------------
def build_faiss_index(chunks):
    """Creates FAISS vector index for semantic search."""
    embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# ----------------------------
# 5. Retrieval
# ----------------------------
def retrieve(query, chunks, index, k=3):
    """Retrieve top-k most relevant chunks for the query."""
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


# ----------------------------
# 6. Answer Generation (OpenRouter)
# ----------------------------
def generate_answer(query, retrieved_chunks):
    """Generate grounded answer using context only."""
    
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI assistant for a construction marketplace.

RULES:
- Use ONLY the provided context.
- Format the answer clearly using bullet points.
- Do NOT include markdown headings like ## or ###.
- No hallucinations. If answer is not in context, say "Information not available."
- Keep sentences short, clean, and formal.

Context:
{context}

Question: {query}

Provide a clean, well-formatted answer:
"""

    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # low temperature = more accurate
    )

    return completion.choices[0].message.content.strip()


# ----------------------------
# 7. Full RAG Pipeline
# ----------------------------
def run_rag(query, documents):
    """End-to-end RAG: chunk → clean → index → retrieve → answer."""
    
    # chunk documents
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    # clean chunks before embedding
    cleaned_chunks = [clean_chunk(c) for c in all_chunks]

    # build index using cleaned chunks
    index = build_faiss_index(cleaned_chunks)

    # retrieve relevant chunks
    retrieved_raw = retrieve(query, cleaned_chunks, index, k=3)

    # generate answer
    answer = generate_answer(query, retrieved_raw)

    return retrieved_raw, answer
