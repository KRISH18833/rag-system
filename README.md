# RAG System

A retrieval-augmented generation system over Cloud Computing for ML course notes, built with Chroma and Gemini.

---

## How It Works

1. **Ingest** — reads every `.pdf` from `data/`, page by page
2. **Chunk** — splits text into ~2000-character chunks with 200-character overlap at sentence/paragraph boundaries
3. **Embed** — encodes each chunk with `gemini-embedding-001` (task: `RETRIEVAL_DOCUMENT`)
4. **Store** — persists embeddings in a Chroma vector store (`./chroma_db`, cosine distance, collection `cloud_notes`)
5. **Retrieve** — embeds the query with `gemini-embedding-001` (task: `RETRIEVAL_QUERY`), returns top 6 chunks
6. **Generate** — sends retrieved context + question to `gemini-3.8-flash` (temperature 0, context-only system instruction)

---

## Document Set

| File | Pages |
|---|---|
| Cloud Unit 1.pdf | 20 |
| Cloud Unit 2.pdf | 29 |
| Cloud Unit 3.pdf | 23 |
| Cloud Unit 4.pdf | 47 |
| Cloud Unit 5.pdf | 16 |

**Total:** 5 PDFs · 135 pages · 175 chunks

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Edit .env:
GEMINI_API_KEY=your_key_here

# 4. Ingest PDFs
python ingest.py

# 5. Query
python query.py "your question" --mode compare
```

**Modes:** `rag` (default) · `plain` · `compare` · `retrieve`  
**Flags:** `--full` (with `--mode retrieve`, prints full chunk text)

---

## With RAG vs Without RAG

### Q1: In the W&B versioning example, which command loads version 2 of the model, and what file name is used?

**Plain answer:**  
Uses `run.use_artifact('model:v2')` to download, then loads `model.pt` or `model.h5` depending on the framework — inferred from general W&B documentation, no specific file name confirmed.

**RAG answer:**  
> `artifact = wandb.use_artifact('my-model:v2')` then `model.load_state_dict(torch.load(f"{artifact_dir}/model_v2.pth"))`. File name: **`model_v2.pth`**.  
> *Sources: Cloud Unit 5.pdf (Pages 10–13, 15–16)*

**Verdict:** RAG gives the exact command and filename from the actual notes; plain answer guesses generic framework filenames.

---

### Q2: In the I/O interference example, what mean bandwidth does the STREAM benchmark need versus disk writes?

**Plain answer:**  
STREAM requires ~10–40+ GB/s of memory bandwidth; disk writes use only ~100–500 MB/s. Disproportionate interference arises from DMA priority at the memory controller.

**RAG answer:**  
> 75 EC2 instances running STREAM require **1,355 MB/s** mean bandwidth; writing 1 GB files to local disk uses only **55 MB/s**.  
> *Sources: Cloud Unit 2.pdf (Pages 11, 22)*

**Verdict:** RAG retrieves the exact figures from the course slides; plain answer gives plausible but far larger ballpark numbers.

---

### Q3: How is elasticity expressed as an equation in the notes?

**Plain answer:**  
Assumes economics elasticity (E = %ΔQ / %ΔP) and offers four formula variants — completely off-topic for this course.

**RAG answer:**  
> Auto scaling = Scalability + Automation  
> Elasticity = Auto scaling + Optimization  
> *Sources: Cloud Unit 1.pdf (Page 7)*

**Verdict:** RAG answers precisely using the cloud computing definition from the notes; plain answer answers the wrong discipline entirely.

---

## Known Limitations

- **Image-only pages not indexed:** Unit 5 pages 6–9 and 14 contain only images; `pypdf` extracts no text from them, so their content is not searchable.
- **Rate limits:** Free-tier 429 errors and transient 503s are handled automatically by exponential backoff (up to 6 retries, capped at 60 s). Non-retryable 4xx errors (e.g. 404) raise immediately.
