import streamlit as st
from rag_pipeline import run_rag

st.set_page_config(page_title="Mini RAG Assistant", page_icon="🔍", layout="wide")

st.title("🔍 Mini RAG – Construction QA Assistant")
st.write("Ask any question based on the internal construction documents.")

# Load text files from /data on Streamlit Cloud
import os

def load_documents():
    docs = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error("❌ data/ folder not found. Please upload your documents inside the data folder.")
        return docs

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

documents = load_documents()

query = st.text_input("Enter your question:")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant information..."):
            retrieved, answer = run_rag(query, documents)

        st.subheader("📌 Retrieved Context (Top Matches)")
        for i, chunk in enumerate(retrieved, 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)

        st.subheader("💬 Final Answer")
        st.write(answer)
