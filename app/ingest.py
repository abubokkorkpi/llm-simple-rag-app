import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = os.getenv("CHROMA_DB_DIR", "/app/data/chroma_db")
DOCS_DIR = os.getenv("DOCUMENTS_DIR", "/app/documents")
COLLECTION_NAME = "rag_documents"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001",
        )
    )


def index_documents():
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in docs_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )
        else:
            continue

        loaded = loader.load()

        for doc in loaded:
            doc.metadata["source"] = file_path.name

        documents.extend(loaded)

    if not documents:
        raise ValueError("No PDF or TXT documents found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as exc:
        if "does not exist" not in str(exc).lower():
            raise

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )

    vector_store.add_documents(chunks)

    return len(chunks)