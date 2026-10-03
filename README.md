# RAG Document Q&A API

A FastAPI backend that lets you upload documents (PDF, TXT, DOCX) and ask
questions about them in plain English. Answers are grounded in the source
text and include citations (filename + page number), not free-floating
LLM guesses.

Built with **FastAPI**, **LangChain**, **ChromaDB**, and **Google Gemini**.

---

## Features

- **Document upload** — PDF, TXT, and DOCX, chunked and embedded automatically
- **Question answering** — returns an answer plus the source chunks it was built from
- **Streaming answers** — Server-Sent Events for real-time, token-by-token responses
- **Per-document search** — restrict a query to a single uploaded document, or search across all of them
- **Optional API-key auth** — lock the API down with one shared secret
- **Clean error handling** — proper 400/401/413/502 responses instead of raw crashes

---

## Requirements

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)

---

## Setup

1. **Clone or copy this project**, then create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment:**

   ```bash
   cp .env.example .env
   ```

   Open `.env` and add your key:

   ```
   GEMINI_API_KEY=your-real-key-here
   ```

4. **Verify your setup before running the server.** This checks that your
   API key works and that the model names are current (Google renames or
   retires models from time to time):

   ```bash
   python check_setup.py
   ```

   You should see `OK` for both the embedding model and the chat model.
   If either fails, the script tells you which models your key can access —
   update `EMBEDDING_MODEL` or `LLM_MODEL` in `.env` accordingly.

5. **Run the server:**

   ```bash
   uvicorn main:app --reload
   ```

6. **Open the interactive docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Usage

### 1. Upload a document

```bash
curl -F "file=@report.pdf" http://localhost:8000/documents
```

Response:

```json
{
  "document_id": "a1b2c3d4e5f6...",
  "filename": "report.pdf",
  "chunks": 42
}
```

Save the `document_id` if you want to query this file specifically later.

### 2. Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Response:

```json
{
  "answer": "...",
  "sources": [
    { "document_id": "a1b2c3d4e5f6...", "filename": "report.pdf", "page": 3, "snippet": "..." }
  ]
}
```

Omit `document_id` to search across every uploaded document. Include it to
restrict the search to one file:

```json
{ "question": "What is the refund policy?", "document_id": "a1b2c3d4e5f6..." }
```

> **Note:** Don't send `"document_id": "string"` — that's just Swagger's
> placeholder text. Either remove the field entirely or use a real id from
> an upload response.

### 3. Stream an answer

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document"}'
```

Streams Server-Sent Events in order: `sources`, then one `token` event per
chunk of the answer, then `done`.

### 4. List or delete documents

```bash
curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/a1b2c3d4e5f6...
```

---

## Authentication (optional)

By default, the API has no auth — fine for local development. To lock it
down, set a value in `.env`:

```
API_KEY=some-random-secret
```

Every request then needs a matching header:

```bash
curl -H "X-API-Key: some-random-secret" http://localhost:8000/documents
```

Requests without a valid key get a `401 Unauthorized`. Leave `API_KEY`
blank to disable auth.

---

## Configuration reference (`.env`)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Your Google Gemini API key |
| `EMBEDDING_MODEL` | No | `models/gemini-embedding-001` | Embedding model used for indexing and search |
| `LLM_MODEL` | No | `gemini-3.6-flash` | Chat model used to generate answers |
| `CHROMA_DIR` | No | `./chroma_db` | Where vector data is stored on disk |
| `MAX_UPLOAD_MB` | No | `20` | Max upload size per file |
| `API_KEY` | No | *(empty = disabled)* | Shared secret required in the `X-API-Key` header |
| `DEBUG` | No | `false` | When `true`, error responses include the real exception message (dev only — leave `false` in production) |

---

## Architecture notes

- **Single shared document store.** All uploaded documents live in one
  vector collection, separated only by `document_id`. This fits a
  single-team or single-use-case deployment (e.g., one company's internal
  docs). It does **not** isolate data between separate end users — anyone
  with API access can list or search all documents. Multi-user isolation
  (per-user documents, separate credentials) is a larger feature built on
  top of this foundation, not included here.
- **Blocking work runs off the event loop.** File parsing and embedding
  are synchronous operations, so they run via `asyncio.to_thread` to keep
  the API responsive under concurrent requests.
- **Models are configurable, not hardcoded.** Google periodically retires
  model versions. If a model stops working, update `EMBEDDING_MODEL` or
  `LLM_MODEL` in `.env` — no code changes needed. Run `check_setup.py`
  after any change.
- **Changing the embedding model requires re-indexing.** Embeddings from
  different models aren't compatible. If you change `EMBEDDING_MODEL`,
  delete the `chroma_db/` folder and re-upload your documents.

---

## Project structure

```
.
├── main.py           # FastAPI app and endpoints
├── rag.py            # Core RAG logic: ingestion, retrieval, answering
├── config.py         # Environment-based settings
├── check_setup.py     # Verifies API key and model names before first run
├── requirements.txt
├── .env.example
└── README.md
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `GEMINI_API_KEY is not set` | `.env` is missing or not in the same folder as `main.py` |
| 500 error on upload | Run `check_setup.py` — usually a retired or misspelled model name |
| `422 Unprocessable Entity` | Invalid JSON in the request body (often a trailing comma) |
| Answer always says "I cannot find the answer" | No documents uploaded yet, or `document_id` doesn't match any uploaded document — check `GET /documents` first |
| Slow first request | The embedding model has some cold-start latency; subsequent requests are faster |

---

## License

All rights reserved. This code is shared for portfolio/demonstration purposes only. No permission is granted to copy, modify, distribute, or use this code without explicit written consent from the author.
