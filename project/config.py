"""Project-wide settings shared by every stage of the pipeline.

This module is the single authoritative definition of the topic list, the
document-ID format and the on-disk layout. Every other module imports them
from here so the stages cannot drift apart.
"""

import hashlib
from pathlib import Path


# Fixed topic names, in a fixed order. Every per-topic artifact (manual
# labels, topic counts, teleport vectors, topic-rank rows) uses this order.
TOPICS = (
    "technology",
    "science",
    "health_and_medicine",
    "sports",
    "politics",
)

# Wikipedia asks for a descriptive User-Agent naming the project and a way to
# reach whoever is running it. Put a real contact address here before crawling.
CONTACT = "add your contact address here"
USER_AGENT = f"Group17InfoRetrievalCrawler/0.1 (educational Wikipedia corpus; {CONTACT})"

# Canonical data layout. project/data/ is Git-ignored: collected corpora and
# built artifacts stay local and are shared separately.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CANDIDATES_DIR = DATA_DIR / "candidates"
REVIEWED_DIR = DATA_DIR / "reviewed"
INDEX_DIR = DATA_DIR / "index"


def doc_id_for(page_id: int) -> str:
    """Our stable identifier for one article.

    Treat the result as an opaque string. It is NOT a matrix row number: use
    doc_to_row (exported by the graph and index manifests) to find a row. The
    "doc-" namespace is deliberately ours rather than Wikipedia's, so that
    later synthetic documents (the Part D spam farm) can take IDs that no
    Wikipedia page_id could collide with.
    """
    return f"doc-{page_id}"


def corpus_build_id(doc_ids) -> str:
    """Short fingerprint of one corpus selection.

    Every artifact built from the same documents records the same value, so a
    stale graph or text index can be spotted instead of quietly mismatching.
    """
    return hashlib.sha1(",".join(doc_ids).encode("utf-8")).hexdigest()[:12]
