# Mini RAG Pipeline – Construction Marketplace Assistant

This project implements a Retrieval-Augmented Generation (RAG) system designed for answering questions using construction-related internal documents.

## The system uses:

MiniLM-L6-v2 for embeddings

FAISS for semantic search

Mistral-7B (OpenRouter) for grounded answer generation

Streamlit for a working chatbot UI

## Features

✔ Document chunking

✔ Embedding via SentenceTransformers

✔ FAISS vector index

✔ Semantic retrieval

✔ Grounded LLM answer generation

✔ Streamlit chatbot interface

✔ Transparent display of retrieved context

### Setup Instructions

1. Install dependencies

pip install -r requirements.txt

2. Export your OpenRouter API key

export OPENROUTER_API_KEY="your_key_here"

3. Run Streamlit app

streamlit run app.py

### Test Questions

Try:

“What factors affect construction project delays?”

“What safety rules are mentioned in the documents?”

“What are the material requirements?”

### Evaluation Criteria

Relevance of retrieved chunks

Groundedness (no hallucinations)

Completeness

Latency

Clarity of final answer