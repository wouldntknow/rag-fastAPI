import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

# Models. Google retires models often, so these are overridable from .env
# without touching the code. Run `python check_setup.py` to verify them.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# Vector store. NOTE: the directory and the collection name are two different things.
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "documents"

# Chunking / retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Uploads
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Optional API-key auth. If API_KEY is empty, auth is disabled (fine for local dev).
API_KEY = os.getenv("API_KEY", "")

# When true, 500/502 responses include the real error message (dev only).
DEBUG = os.getenv("DEBUG", "false").lower() == "true"