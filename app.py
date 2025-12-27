import streamlit as st
import PyPDF2
import json
import os
from datetime import datetime

from rag_pipeline import (
    run_rag,
    chunk_text,
    bm25_search,
    embedding_search
)

from openai import OpenAI

# -----------------------------
# OpenRouter Client
# -----------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------
# STYLE
# -----------------------------
st.set_page_config(page_title="Mini RAG Premium", layout="wide")

custom_css = """
<style>
    body { background: #0e1117; color: white; }
    .chunk-box { background: #1a1c23; padding: 12px; border-radius: 10px; margin-bottom: 10px; }
    .highlight { background-color: #ffd54f; color: black; padding: 3px 5px; border-radius: 3px; }
    .answer-box { background: #1f2129; padding: 15px; border-radius: 12px; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


# -----------------------------
# Sentence Highlighter
# -----------------------------
def highlight_matches(chunk, query):
    for word in query.split():
        if len(word) > 3:
            chunk = chunk.replace(word, f"<span class='highlight'>{word}</span>")
    return chunk


# -----------------------------
# Model Selector
# -----------------------------
model_options = {
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "LLaMA 3 8B": "meta-llama/llama-3-8b-instruct",
    "Qwen 2.5 7B": "qwen/qwen-2.5-7b-instruct",
    "DeepSeek R1": "deepseek/deepseek-r1-distill"
}


# -----------------------------
# CHAT + DOCUMENT STORAGE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunk_map" not in st.session_state:
    st.session_state.chunk_map = []

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []


# -----------------------------
# HEADER
# -----------------------------
st.title("🏗️ Mini-RAG Premium")
st.caption("Enterprise-grade RAG with Hybrid Search + Citations + Highlighting + JSON Export")


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload internal documents (.pdf/.txt)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    st.subheader("📘 Loaded Documents")

    for file in uploaded_files:

        if file.name.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            text = extract_text_from_pdf(file)

        documents.append(text)

        chunks = chunk_text(text)
        st.session_state.all_chunks.extend(chunks)

        st.markdown(f"**{file.name}** — {len(chunks)} chunks loaded")


# -----------------------------
# HYBRID SEARCH MODE
# -----------------------------
st.subheader("🔍 Retrieval Mode")
retrieval_mode = st.radio(
    "Choose retrieval method:",
    ["Hybrid (BM25 + Embeddings)", "Embeddings Only", "BM25 Only"]
)


# -----------------------------
# MODEL SELECTOR
# -----------------------------
st.subheader("🧠 Choose the LLM")
chosen_model = st.selectbox("Select model:", list(model_options.keys()))


# -----------------------------
# QUESTION BOX
# -----------------------------
st.subheader("💬 Ask a Question")
query = st.text_input("Your question:")

ask = st.button("Ask")


# -----------------------------
# RUN PIPELINE
# -----------------------------
if ask:

    if not documents:
        st.warning("Upload documents first.")
    else:
        with st.spinner("Retrieving context + generating answer..."):

            # ---- HYBRID SEARCH ----
            if retrieval_mode == "Hybrid (BM25 + Embeddings)":
                top_chunks = bm25_search(query, st.session_state.all_chunks, k=3)
                emb_chunks = embedding_search(query, st.session_state.all_chunks, k=3)
                retrieved = list(dict.fromkeys(top_chunks + emb_chunks))[:3]

            elif retrieval_mode == "BM25 Only":
                retrieved = bm25_search(query, st.session_state.all_chunks, k=3)

            else:
                retrieved = embedding_search(query, st.session_state.all_chunks, k=3)

            # ---- ADD HIGHLIGHTING ----
            highlighted = [highlight_matches(c, query) for c in retrieved]

            # ---- GENERATE ANSWER ----
            llm_model = model_options[chosen_model]

            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "Answer ONLY from the context. Provide citations (Chunk X)."},
                    {"role": "user", "content": f"Context:\n{retrieved}\n\nQuestion: {query}"}
                ]
            )
            answer = completion.choices[0].message.content

            # ---- SAVE HISTORY ----
            result = {
                "question": query,
                "model": chosen_model,
                "retrieval_mode": retrieval_mode,
                "retrieved_chunks": retrieved,
                "answer": answer,
                "time": str(datetime.now())
            }
            st.session_state.chat_history.append(result)


# -----------------------------
# SHOW RESULTS
# -----------------------------
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]

    st.subheader("📌 Retrieved Chunks (with Highlights)")
    for i, c in enumerate(last["retrieved_chunks"], 1):
        st.markdown(f"**Chunk {i}:**")
        st.markdown(f"<div class='chunk-box'>{highlight_matches(c, query)}</div>", unsafe_allow_html=True)

    st.subheader("🧠 Final Answer")
    st.markdown(f"<div class='answer-box'>{last['answer']}</div>", unsafe_allow_html=True)

    # -----------------------------
    # JSON EXPORT
    # -----------------------------
    json_data = json.dumps(last, indent=4)
    st.download_button(
        "📥 Download JSON Report",
        json_data,
        file_name="rag_result.json",
        mime="application/json"
    )


# -----------------------------
# CHAT HISTORY
# -----------------------------
st.subheader("📝 Chat History")
for chat in st.session_state.chat_history:
    st.write(f"### Q: {chat['question']}")
    st.markdown(f"<div class='answer-box'>{chat['answer']}</div>", unsafe_allow_html=True)
    st.write("---")
