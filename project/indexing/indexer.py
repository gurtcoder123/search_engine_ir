"""Build a plain inverted index from saved tokens. No TF-IDF or ranking yet."""

import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project.config import PREPROCESSING_DIR


def load_tokens(data_dir=PREPROCESSING_DIR):
    """Load only token files whose collection records are currently approved."""
    data_dir = Path(data_dir)
    records = json.loads((data_dir / "Document_Collection.json").read_text(encoding="utf-8"))
    tokens_by_document = {}
    for record in records:
        if record.get("approved") is not True:
            continue
        doc_id = record["doc_id"]
        saved = json.loads((data_dir / "Tokens" / f"{doc_id}.json").read_text(encoding="utf-8"))
        tokens_by_document[doc_id] = saved["tokens"]
    return tokens_by_document


def build_inverted_index(tokens_by_document):
    """Return postings, document frequencies and per-document raw term counts.

    Each posting is (doc_id, raw count), once per document containing the term.
    Even an approved document with no tokens remains in term_counts.
    """
    inverted_index = {}
    term_counts = {}
    for doc_id, tokens in tokens_by_document.items():
        counts = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        term_counts[doc_id] = counts
        for term, count in counts.items():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append((doc_id, count))

    df = {}
    for term, postings in inverted_index.items():
        df[term] = len(postings)
    return inverted_index, df, term_counts


def save_index(inverted_index, df, term_counts, output_dir):
    """Save readable JSON. Original documents and token files are untouched."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "inverted_index.json": inverted_index,
        "df.json": df,
        "term_counts.json": term_counts,
        "doc_ids.json": list(term_counts),
    }
    for filename, contents in files.items():
        (output_dir / filename).write_text(
            json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    tokens_by_document = load_tokens()
    inverted_index, df, term_counts = build_inverted_index(tokens_by_document)
    output_dir = PREPROCESSING_DIR / "Index"
    save_index(inverted_index, df, term_counts, output_dir)
    print(f"Indexed {len(term_counts)} approved documents and {len(inverted_index)} terms.")
    print(f"Saved to {output_dir}")
