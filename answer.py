"""
answer.py
Phase 3: Full RAG pipeline.
Takes a question -> retrieves the most relevant chunks (via query.py's
logic) -> feeds them to GPT with instructions to answer ONLY from that
context -> prints a grounded answer, with a safeguard against weak
matches so it doesn't hallucinate when the document doesn't cover the
question.

Usage:
    python answer.py "What is the policy on accepting gifts?"
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

from query import load_store, find_relevant_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"

# If the best-matching chunk scores below this, we don't trust the
# retrieval enough to let GPT answer -- we tell the user honestly
# instead of risking a made-up answer.
SIMILARITY_THRESHOLD = 0.25


def build_prompt(question, chunks):
    """
    Combines the retrieved chunks into a single context block and
    wraps it with strict instructions so GPT sticks to the source
    material instead of using its own general knowledge.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {item['source']}, chunk #{item['id']}]\n{item['text']}"
        for _, item in chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
    "provided document excerpts below. Do not use any outside knowledge "
    "or facts not present in the excerpts. You may perform simple "
    "arithmetic (percentages, sums, comparisons) using numbers explicitly "
    "stated in the excerpts -- for example, if the text gives a maximum "
    "percentage of income and the user states their income, you should "
    "calculate the resulting figure. If the excerpts genuinely do not "
    "contain information relevant to the question, say clearly that the "
    "document does not cover this, rather than guessing. Keep answers "
    "concise and reference the source when helpful."
    "If the question asks for a list, index, or regulation number/title, prioritize any chunk tagged table_of_contents."
    )

    user_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    return system_prompt, user_prompt


def generate_answer(question, store):
    chunks = find_relevant_chunks(question, store)
    top_score = chunks[0][0]

    if top_score < SIMILARITY_THRESHOLD:
        return (
            "I couldn't find anything in this document that answers that "
            "question confidently. It may not be covered in the source "
            "material.",
            chunks,
        )

    system_prompt, user_prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature -- we want faithful, not creative
    )

    answer = response.choices[0].message.content
    return answer, chunks


def print_sources(chunks):
    print("\nSources used:")
    for score, item in chunks:
        print(f"  - {item['source']} (chunk #{item['id']}, score {score:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python answer.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    store = load_store()

    answer, chunks = generate_answer(question, store)

    print("\nAnswer:\n" + "-" * 40)
    print(answer)
    print("-" * 40)

    print_sources(chunks)