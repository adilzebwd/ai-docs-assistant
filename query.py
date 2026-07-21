"""
query.py
Loads embeddings_store.json, embeds a user's question, and finds the
top-N most similar chunks using cosine similarity (plain numpy, no
vector database needed).

Usage:
    python query.py "What is the refund policy?"
"""

import os
import sys
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
STORE_FILE = "embeddings_store.json"
TOP_N = 5  # how many chunks to retrieve per question


def load_store():
    """Loads the saved chunks + embeddings from ingest.py."""
    if not os.path.exists(STORE_FILE):
        raise FileNotFoundError(
            f"{STORE_FILE} not found. Run ingest.py on a document first."
        )
    with open(STORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text):
    """Embeds a single piece of text (the user's question)."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    """
    Standard cosine similarity between two vectors.
    Returns a value from -1 to 1 -- closer to 1 means more similar.
    This is the "search" in our search engine: no database, just math.
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_relevant_chunks(question, store, top_n=TOP_N):
    """
    Embeds the question, compares it against every stored chunk,
    and returns the top_n most similar ones.
    """
    question_embedding = get_embedding(question)

    scored_chunks = []
    for item in store:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored_chunks.append((score, item))

    # Sort by similarity score, highest first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return scored_chunks[:top_n]


def print_results(results):
    print("\nTop matching chunks:\n" + "-" * 40)
    for score, item in results:
        print(f"Score: {score:.4f} | Source: {item['source']} | Chunk #{item['id']}")
        print(item["text"][:300].replace("\n", " ") + "...")
        print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    store = load_store()
    print(f"Loaded {len(store)} chunks from {STORE_FILE}")

    results = find_relevant_chunks(question, store)
    print_results(results)