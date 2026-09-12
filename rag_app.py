#!/usr/bin/env python3
"""
Week 3: Embeddings, Vector Search & RAG
A simple RAG (Retrieval-Augmented Generation) app that answers questions
grounded in your own PDF documents, instead of relying on the model's
internal training data.

Pipeline:
  1. Load PDFs from ./docs and extract text (pypdf)
  2. Split text into chunks (so retrieval can find specific relevant pieces,
     not whole documents)
  3. Embed each chunk into a vector using a free, local embedding model
     (sentence-transformers/all-MiniLM-L6-v2 — no API cost, runs on your
     machine)
  4. Store all chunk vectors in a local Chroma vector database
  5. Question loop: embed the user's question the same way, find the most
     similar chunks (cosine similarity, handled by Chroma), and pass those
     chunks + the question to Gemini as context. Gemini answers using ONLY
     that context, and we print which document(s) it came from.
"""

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "my_documents"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks so context isn't cut mid-idea
TOP_K = 5              # how many chunks to retrieve per question
GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are a question-answering assistant. You will be given some "
    "CONTEXT extracted from the user's own documents, followed by a "
    "QUESTION. Answer the question using ONLY the information in the "
    "CONTEXT. If the context does not contain enough information to "
    "answer, say clearly: \"I don't know based on the provided documents.\" "
    "Do not use outside knowledge. Do not guess. Always mention which "
    "document(s) (by filename) your answer is based on."
)


def load_pdfs(docs_dir: Path) -> list[dict]:
    """Read every PDF in docs_dir and return a list of {text, source} dicts,
    one per page, so we know which file (and page) each chunk came from."""
    pages = []
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {docs_dir}/. Add some and try again.")
        sys.exit(1)

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_num,
                })
        print(f"Loaded {pdf_path.name} ({len(reader.pages)} pages)")

    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """Split each page's text into overlapping chunks. Overlap avoids
    losing meaning when a sentence gets cut at a chunk boundary."""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_vector_store(chunks: list[dict]):
    """Embed all chunks and store them in a local Chroma collection.
    Recreates the collection each run so re-running the script doesn't
    duplicate data."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Free, local embedding model — downloads once, then runs on-device.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Start fresh each run so old data doesn't pile up.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )

    print(f"Embedded and stored {len(chunks)} chunks in Chroma.\n")
    return collection


def answer_question(collection, question: str, gemini_client) -> dict:
    """Retrieve the most relevant chunks for the question, then ask Gemini
    to answer using only those chunks as context."""
    results = collection.query(query_texts=[question], n_results=TOP_K)

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    if not retrieved_docs:
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    context_blocks = []
    sources = []
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        context_blocks.append(f"[Source: {meta['source']}, page {meta['page']}]\n{doc}")
        sources.append(f"{meta['source']} (page {meta['page']})")

    context = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    # Extract text directly from parts to avoid warnings on responses that
    # include non-text metadata parts.
    answer_text = "".join(
        part.text
        for candidate in response.candidates
        for part in candidate.content.parts
        if getattr(part, "text", None)
    )

    return {"answer": answer_text, "sources": sources}


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    gemini_client = genai.Client(api_key=api_key)

    print("Loading PDFs from ./docs ...")
    pages = load_pdfs(DOCS_DIR)

    print("\nChunking text ...")
    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.\n")

    print("Building vector store (embedding chunks) ...")
    collection = build_vector_store(chunks)

    print("Ready. Ask questions about your documents (type 'exit' to quit).\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = answer_question(collection, question, gemini_client)
        print(f"\nA: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print()


if __name__ == "__main__":
    main()