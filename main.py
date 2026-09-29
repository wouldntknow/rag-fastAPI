import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Optional auth: enforced only when API_KEY is set in the environment."""
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def error_detail(exc: Exception, default: str) -> str:
    """Show the real error only in DEBUG mode; keep it generic in production."""
    return f"{type(exc).__name__}: {exc}" if config.DEBUG else default


app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents, then ask questions and get answers with sources.",
    version="1.1.0",
    dependencies=[Depends(require_api_key)],
)


# ---------- Schemas ----------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = Field(
        default=None, description="Restrict the search to one document. Omit to search all."
    )


class Source(BaseModel):
    document_id: str | None
    filename: str | None
    page: int | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    document_found: bool = True


# ---------- Endpoints ----------

@app.get("/", dependencies=[])
async def root():
    return {
        "message": "RAG Document Q&A API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", dependencies=[])
async def health():
    return {"status": "healthy"}


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT or DOCX file. It is chunked, embedded and stored."""
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{ext}"
        size = 0
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413, detail=f"File exceeds {config.MAX_UPLOAD_MB} MB limit."
                    )
                out.write(chunk)

        try:
            # Loading + embedding are blocking, so run them off the event loop.
            return await asyncio.to_thread(rag.index_document, path, filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Failed to index %s", filename)
            raise HTTPException(
                status_code=500, detail=error_detail(e, "Failed to process the document.")
            )


@app.get("/documents")
async def list_documents():
    return await asyncio.to_thread(rag.list_documents)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    removed = await asyncio.to_thread(rag.delete_document, document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "chunks_removed": removed}


DOCUMENT_NOT_FOUND_MESSAGE = "No document was found with that document_id. Check GET /documents for valid ids."


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Ask a question. Returns the answer plus the source chunks used."""
    if body.document_id and not await asyncio.to_thread(rag.document_exists, body.document_id):
        return QueryResponse(answer=DOCUMENT_NOT_FOUND_MESSAGE, sources=[], document_found=False)

    try:
        docs = await rag.retrieve(body.question, body.document_id)
        text = await rag.answer(body.question, docs)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=502, detail=error_detail(e, "The AI service failed. Please try again.")
        )
    return QueryResponse(answer=text, sources=rag.format_sources(docs))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query/stream")
async def query_stream(body: QueryRequest):
    """Same as /query but streams Server-Sent Events: sources, token..., done."""

    async def events():
        try:
            if body.document_id and not await asyncio.to_thread(rag.document_exists, body.document_id):
                yield _sse("error", {"detail": DOCUMENT_NOT_FOUND_MESSAGE})
                return

            docs = await rag.retrieve(body.question, body.document_id)
            yield _sse("sources", {"sources": rag.format_sources(docs)})
            async for token in rag.stream_answer(body.question, docs):
                yield _sse("token", {"text": token})
            yield _sse("done", {})
        except Exception as e:
            logger.exception("Streaming query failed")
            yield _sse("error", {"detail": error_detail(e, "The AI service failed. Please try again.")})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )