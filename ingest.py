"""
ingest.py
Reads a document (PDF or .txt), splits it into overlapping chunks,
generates OpenAI embeddings for each chunk, and saves everything
to a local JSON file (our lightweight "vector store").

Usage:
    python ingest.py path/to/document.pdf
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
STORE_FILE = "embeddings_store.json"

# --- Chunking settings ---
CHUNK_SIZE = 1000      # characters per chunk (roughly ~200-250 tokens)
CHUNK_OVERLAP = 150    # overlap so we don't cut sentences awkwardly at boundaries


def extract_text_from_pdf(filepath):
    """Reads a PDF and returns all text as a single string."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_txt(filepath):
    """Reads a plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_document(filepath):
    """Detects file type and extracts text accordingly."""
    if filepath.lower().endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif filepath.lower().endswith(".txt"):
        return extract_text_from_txt(filepath)
    else:
        raise ValueError("Unsupported file type. Use .pdf or .txt")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks.
    Overlap helps preserve context across chunk boundaries so we don't
    lose meaning when a sentence gets cut in half.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # move forward, but overlap with previous chunk

    # Remove any empty chunks (can happen at the very end)
    return [c for c in chunks if c]


def get_embedding(text):
    """Calls OpenAI's embedding API for a single piece of text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def ingest_document(filepath):
    print(f"Reading document: {filepath}")
    text = load_document(filepath)
    print(f"Extracted {len(text)} characters.")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks.")

    store = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        store.append({
            "id": i,
            "text": chunk,
            "embedding": embedding,
            "source": os.path.basename(filepath)
        })

    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f)

    print(f"\nDone! Saved {len(store)} chunks with embeddings to {STORE_FILE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)

    filepath = sys.argv[1]
    ingest_document(filepath)