import streamlit as st
from rag_pipeline import run_rag

st.set_page_config(page_title="Mini RAG – Construction Assistant")

st.title("🏗️ Mini RAG – Construction AI Assistant")
st.write("Upload documents and ask questions. Answers come ONLY from your data.")

uploaded = st.file_uploader("Upload .txt files", accept_multiple_files=True)

documents = []
if uploaded:
    for f in uploaded:
        documents.append(f.read().decode("utf-8"))

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not documents:
        st.warning("Please upload documents first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            retrieved, answer = run_rag(query, documents)

        st.subheader("📄 Retrieved Chunks")
        for i, chunk in enumerate(retrieved):
            st.markdown(f"**Chunk {i+1}:**\n{chunk}")

        st.subheader("💬 Final Answer")
        st.write(answer)
