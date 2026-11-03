import os
import sys
import time
import random
import argparse
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "cloud_notes"
EMBED_MODEL = "gemini-embedding-001"
MODEL = "gemini-3.8-flash"


def embed_query(text):
    attempt = 0
    while True:
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return response.embeddings[0].values
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Embedding failed after {attempt} retries: {e}") from e


def generate_with_retry(contents, system_instruction=None):
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system_instruction,
    ) if system_instruction else types.GenerateContentConfig(temperature=0.0)

    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            return response.text
        except APIError as e:
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            if code == 429:
                msg = str(getattr(e, "message", e))
                raw_wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                if "per day" in msg.lower() or raw_wait > 60:
                    raise RuntimeError("Daily API quota reached. Try again later.") from e
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Generation error ({e.code}): {e.message}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Generation failed after {attempt} retries: {e}") from e
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Generation error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Generation failed after {attempt} retries: {e}") from e


def retrieve(question, k=6):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i],
        })
    return chunks


def rag_answer(question):
    chunks = retrieve(question)
    sources = []
    context_blocks = []
    for chunk in chunks:
        label = f"{chunk['source']} (Page {chunk['page']})"
        if label not in sources:
            sources.append(label)
        context_blocks.append(f"[{label}]\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    system_instruction = (
        "Answer ONLY using the provided context. If the answer is not in the context, "
        "reply exactly: I could not find this in the provided notes. Cite the source file and page for each fact."
    )

    answer = generate_with_retry(prompt, system_instruction=system_instruction)
    return answer, sources


def plain_answer(question):
    return generate_with_retry(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--mode", choices=["rag", "plain", "compare", "retrieve"], default="rag", help="Execution mode")
    parser.add_argument("--full", action="store_true", help="Print full chunk text instead of first 300 chars (retrieve mode)")
    args = parser.parse_args()

    if args.mode == "retrieve":
        print(f"Query: {args.question}\n")
        results = retrieve(args.question)
        for rank, chunk in enumerate(results, start=1):
            text = chunk["text"] if args.full else chunk["text"][:300].replace("\n", " ")
            print(f"[{rank}] {chunk['source']} | page {chunk['page']} | distance {chunk['distance']:.4f}")
            print(f"    {text}" + ("" if args.full else "..."))
            print()

    elif args.mode == "plain":
        answer = plain_answer(args.question)
        print(answer)

    elif args.mode == "rag":
        answer, sources = rag_answer(args.question)
        print(answer)
        print("\nSources:")
        for s in sources:
            print(f"- {s}")

    elif args.mode == "compare":
        plain = plain_answer(args.question)
        rag, sources = rag_answer(args.question)

        sources_text = "\n".join(f"- {s}" for s in sources)
        print("=== Plain Answer ===")
        print(plain)
        print("\n=== RAG Answer ===")
        print(rag)
        print("\nSources:")
        print(sources_text)

        markdown_entry = f"## {args.question}\n\n### Plain Answer\n{plain}\n\n### RAG Answer\n{rag}\n\n**Sources:**\n{sources_text}\n\n---\n\n"
        with open("comparison_results.md", "a", encoding="utf-8") as f:
            f.write(markdown_entry)
