"""Shared topic names and the folder containing our saved articles."""

from pathlib import Path


# Manual labels and later topic-based retrieval use this fixed order.
TOPICS = (
    "technology",
    "science",
    "health_and_medicine",
    "sports",
    "politics",
)

# Identify our project; callers may supply a different User-Agent/contact.
USER_AGENT = (
    "Group17InfoRetrievalCrawler/0.1 "
    "(educational project; https://github.com/gurtcoder123/search_engine_ir)"
)

# Downloaded JSON stays local and is Git-ignored.
PREPROCESSING_DIR = Path(__file__).resolve().parent / "Pre-Processing"
DOCUMENTS_DIR = PREPROCESSING_DIR / "Documents"
COLLECTION_FILE = PREPROCESSING_DIR / "Document_Collection.json"
