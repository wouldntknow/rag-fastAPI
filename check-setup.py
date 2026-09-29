"""Run this BEFORE starting the server: python check_setup.py

Makes one real call to the embedding model and one to the chat model, so a
retired or misspelled model name shows up here instead of as a 500 error.
"""
from google import genai

import config

client = genai.Client(api_key=config.GEMINI_API_KEY)


def bare(name: str) -> str:
    return name.removeprefix("models/")


print(f"Embedding model: {config.EMBEDDING_MODEL}")
try:
    result = client.models.embed_content(model=bare(config.EMBEDDING_MODEL), contents="hello")
    print(f"  OK, vector size {len(result.embeddings[0].values)}")
except Exception as e:
    print(f"  FAILED: {e}")

print(f"Chat model: {config.LLM_MODEL}")
try:
    result = client.models.generate_content(model=bare(config.LLM_MODEL), contents="Say hi in 3 words.")
    print(f"  OK: {result.text.strip()}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nModels your key can use (embedding + flash):")
for m in client.models.list():
    if "embed" in m.name or "flash" in m.name:
        print("  ", m.name)