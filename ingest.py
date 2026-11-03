import os
import re
import time
import random
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

DATA_DIR = Path("data")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "cloud_notes"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MIN_CHARS = 50
BATCH_SIZE = 50
EMBED_MODEL = "gemini-embedding-001"


def split_chunks(text, source, page_num):
    """Split text into overlapping chunks at sentence/paragraph boundaries."""
    chunks = []
    # Prefer splitting at paragraph or sentence boundaries
    boundaries = [m.end() for m in re.finditer(r'(\n\n+|\. +|\? +|! +)', text)]
    boundaries = sorted(set(boundaries))

    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE

        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                chunks.append({"text": chunk, "source": source, "page": page_num})
            break

        # Find a boundary within [end - CHUNK_OVERLAP, end] to avoid mid-sentence cuts
        split_at = end
        search_start = max(start + 1, end - CHUNK_OVERLAP)
        candidates = [b for b in boundaries if search_start <= b <= end]
        if candidates:
            split_at = candidates[-1]

        chunk = text[start:split_at]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source, "page": page_num})

        # Step forward with overlap
        next_start = split_at - CHUNK_OVERLAP
        # Align next_start to a boundary if possible
        fwd_candidates = [b for b in boundaries if next_start <= b <= split_at]
        if fwd_candidates:
            next_start = fwd_candidates[0]

        start = max(next_start, start + 1)  # always make progress

    return chunks


def load_pdfs():
    pages = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if len(text) < MIN_CHARS:
                print(f"  [WARNING] {pdf_path.name} page {page_num}: only {len(text)} chars")
            pages.append({"text": text, "source": pdf_path.name, "page": page_num})
    return pdf_files, pages


def embed_batches(chunks):
    texts = [c["text"] for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        attempt = 0
        while True:
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    ),
                )
                all_embeddings.extend([e.values for e in response.embeddings])
                print(f"  Embedded batch {i // BATCH_SIZE + 1} / {-(-len(texts) // BATCH_SIZE)}")
                time.sleep(1)  # short sleep to respect rate limits
                break
            except Exception as e:
                attempt += 1
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...")
                time.sleep(wait)
                if attempt >= 6:
                    raise RuntimeError(f"Embedding failed after {attempt} retries: {e}") from e

    return all_embeddings


def main():
    print("=== Ingesting PDFs ===")

    # Load PDFs
    pdf_files, pages = load_pdfs()
    print(f"Found {len(pdf_files)} PDF(s), {len(pages)} page(s) total.")

    # Chunk all pages
    all_chunks = []
    for p in pages:
        all_chunks.extend(split_chunks(p["text"], p["source"], p["page"]))
    print(f"Created {len(all_chunks)} chunk(s).")

    if not all_chunks:
        print("No chunks to embed. Exiting.")
        return

    # Embed
    print("Embedding chunks...")
    embeddings = embed_batches(all_chunks)

    # Chroma setup — delete and recreate collection each run
    print("Storing in ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in all_chunks]
    documents = [c["text"] for c in all_chunks]

    for i in range(0, len(all_chunks), BATCH_SIZE):
        collection.add(
            ids=ids[i: i + BATCH_SIZE],
            embeddings=embeddings[i: i + BATCH_SIZE],
            documents=documents[i: i + BATCH_SIZE],
            metadatas=metadatas[i: i + BATCH_SIZE],
        )

    # Summary
    avg_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    print("\n=== Summary ===")
    print(f"  PDFs         : {len(pdf_files)}")
    print(f"  Pages        : {len(pages)}")
    print(f"  Chunks       : {len(all_chunks)}")
    print(f"  Avg chunk len: {avg_len:.0f} chars")
    print("\n--- 3 Sample Chunks ---")
    for c in all_chunks[:3]:
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{c['source']} | p{c['page']}] {preview}...")


if __name__ == "__main__":
    main()
