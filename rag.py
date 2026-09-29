"""Core RAG logic: ingestion, retrieval, and answer generation.

Kept separate from the FastAPI layer so it can be tested and reused
(e.g. by the Chainlit demo UI).
"""
import uuid
from pathlib import Path
from typing import AsyncIterator

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

# Google's API occasionally returns 503 (overloaded) or 429 (rate limited).
# Both are usually transient, so retry a few times with backoff before
# giving up and surfacing an error to the client.
retry_on_transient_errors = retry(
    retry=retry_if_exception_type((ServiceUnavailable, ResourceExhausted)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    google_api_key=config.GEMINI_API_KEY,
)

# No temperature set on purpose: Gemini 3.x models work best with their defaults.
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
)

# One persistent collection; every chunk is tagged with its document_id.
vectorstore = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=config.CHROMA_DIR,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions using only the provided context.
If the answer is not in the context, say "I cannot find the answer to that question in the documents."

Context:
{context}

Question: {question}

Answer:"""
)

NOT_FOUND_MESSAGE = "I cannot find the answer to that question in the documents."


def _text(message) -> str:
    """Extract plain text from a model message.

    Gemini 3.x can return content as a list of blocks (text, reasoning, ...),
    so use the message's .text accessor, which keeps only the text parts.
    Works across langchain-core versions (property vs method).
    """
    value = message.text
    return value() if callable(value) else value


# ---------- Ingestion (blocking; call via asyncio.to_thread) ----------

def _get_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    raise ValueError(f"Unsupported file type: {ext}")


def index_document(path: Path, filename: str) -> dict:
    """Load, chunk, embed and store a document. Returns its id and stats."""
    docs = _get_loader(path).load()
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("Could not extract any text from this file.")

    document_id = uuid.uuid4().hex
    for chunk in chunks:
        # Replace loader metadata (which contains a temp path) with clean fields.
        metadata = {"document_id": document_id, "filename": filename}
        page = chunk.metadata.get("page")
        if isinstance(page, int):
            metadata["page"] = page + 1  # loaders are 0-indexed
        chunk.metadata = metadata

    vectorstore.add_documents(chunks)
    return {"document_id": document_id, "filename": filename, "chunks": len(chunks)}


def list_documents() -> list[dict]:
    data = vectorstore.get(include=["metadatas"])
    found: dict[str, dict] = {}
    for meta in data.get("metadatas") or []:
        entry = found.setdefault(
            meta["document_id"],
            {"document_id": meta["document_id"], "filename": meta.get("filename"), "chunks": 0},
        )
        entry["chunks"] += 1
    return list(found.values())


def document_exists(document_id: str) -> bool:
    """Check whether a document_id has any indexed chunks."""
    existing = vectorstore.get(where={"document_id": document_id}, limit=1)
    return bool(existing.get("ids"))


def delete_document(document_id: str) -> int:
    """Delete all chunks of a document. Returns number of chunks removed."""
    existing = vectorstore.get(where={"document_id": document_id})
    ids = existing.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


# ---------- Query (async) ----------

async def retrieve(question: str, document_id: str | None = None) -> list[Document]:
    search_filter = {"document_id": document_id} if document_id else None
    return await vectorstore.asimilarity_search(question, k=config.TOP_K, filter=search_filter)


def format_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "document_id": d.metadata.get("document_id"),
            "filename": d.metadata.get("filename"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:200].strip(),
        }
        for d in docs
    ]


def _build_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        label = d.metadata.get("filename", "document")
        if d.metadata.get("page"):
            label += f", page {d.metadata['page']}"
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


@retry_on_transient_errors
async def _generate(messages) -> str:
    reply = await llm.ainvoke(messages)
    return _text(reply)


async def answer(question: str, docs: list[Document]) -> str:
    if not docs:
        return NOT_FOUND_MESSAGE
    messages = await prompt.ainvoke({"context": _build_context(docs), "question": question})
    return await _generate(messages)


async def stream_answer(question: str, docs: list[Document]) -> AsyncIterator[str]:
    if not docs:
        yield NOT_FOUND_MESSAGE
        return
    messages = await prompt.ainvoke({"context": _build_context(docs), "question": question})
    try:
        async for chunk in llm.astream(messages):
            token = _text(chunk)
            if token:
                yield token
    except (ServiceUnavailable, ResourceExhausted):
        # Streaming can't be retried mid-stream once tokens have already been
        # sent, so fall back to one non-streamed attempt (which does retry).
        yield await _generate(messages)