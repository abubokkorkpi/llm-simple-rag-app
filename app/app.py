import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ingest import index_documents
from rag import ask_question

load_dotenv()

st.set_page_config(
    page_title="Simple RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat with your documents")
st.caption("A document Q&A assistant powered by LangChain and Gemini.")

docs_dir = Path(os.getenv("DOCUMENTS_DIR", "/app/documents"))
docs_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one file to upload.")
        else:
            try:
                for uploaded_file in uploaded_files:
                    target = docs_dir / Path(uploaded_file.name).name
                    target.write_bytes(uploaded_file.getvalue())

                with st.spinner("Creating embeddings and indexing..."):
                    chunk_count = index_documents()

                st.success(f"Indexed {chunk_count} chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("Available documents")

    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in {".pdf", ".txt"}:
            st.write(f"📄 {path.name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    label = source["source"]
                    if source["page"] is not None:
                        label += f" — page {source['page']}"
                    st.write(f"📄 {label}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and asking Gemini..."):
                answer, sources = ask_question(question)

            if isinstance(answer, str):
                st.markdown(answer)
            elif isinstance(answer, list):
                for block in answer:
                    if isinstance(block, dict) and block.get("type") == "text":
                        st.markdown(block.get("text", ""))
            else:
                st.markdown(str(answer))

            with st.expander("Sources"):
                for source in sources:
                    label = source["source"]
                    if source["page"] is not None:
                        label += f" — page {source['page']}"
                    st.write(f"📄 {label}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
        except Exception as exc:
            st.error(f"Could not answer your question: {exc}")