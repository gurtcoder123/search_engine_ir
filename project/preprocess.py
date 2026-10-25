"""After manual review, rebuild all search artifacts in Pre-Processing/Index/."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.config import PREPROCESSING_DIR
from project.collection.crawler import Document, DocumentCollection
from project.indexing.normalize import load_document
from project.indexing.indexer import build_inverted_index, save_index
from project.indexing.tfidf import build_tfidf_matrix, save_tfidf_matrix
from project.ranking.topic_model import build_topic_model
from project.ranking.tspr import build_rankings


def build_search_data(data_dir=PREPROCESSING_DIR):
    """Reuse the small existing modules; never crawl or change review decisions."""
    data_dir = Path(data_dir)
    records = json.loads((data_dir / "Document_Collection.json").read_text(encoding="utf-8"))
    documents = {r["doc_id"]: r for r in records if r.get("approved") is True}
    if not documents:
        raise ValueError("No approved documents. Run reviewer.py first.")
    collection = DocumentCollection()
    for record in documents.values():
        collection.add_document(Document(**record))
    print(f"Tokenizing {len(documents)} approved documents...", flush=True)
    tokens = load_document(collection, data_dir / "Tokens")
    postings, df, counts = build_inverted_index(tokens)
    doc_ids = list(counts)
    matrix, vocabulary, df_vector, idf = build_tfidf_matrix(counts, df, doc_ids)
    print("Building unigram topic model and topic-sensitive PageRank...", flush=True)
    log_probs, model = build_topic_model(documents, counts, vocabulary)
    topic_ranks, global_rank, ranking = build_rankings(documents, doc_ids)
    # Do not save partial new ranking artifacts if an iteration did not converge.
    infos = [ranking["global"], *ranking["per_topic"].values()]
    if not all(info["converged"] for info in infos):
        raise ValueError("PageRank did not converge within 200 iterations; inspect the corpus/settings.")
    index_dir = data_dir / "Index"
    save_index(postings, df, counts, index_dir)
    save_tfidf_matrix(matrix, vocabulary, df_vector, idf, doc_ids, index_dir)
    np.save(index_dir / "topic_log_probs.npy", log_probs)
    np.save(index_dir / "topic_pagerank.npy", topic_ranks)
    np.save(index_dir / "global_pagerank.npy", global_rank)
    for name, contents in (("topic_model", model), ("ranking", ranking)):
        (index_dir / f"{name}.json").write_text(
            json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved all indexes/models to {index_dir}", flush=True)
    print(f"Internal links: {ranking['internal_links']}; dangling documents: {ranking['dangling_documents']}")
    return ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSING_DIR)
    args = parser.parse_args()
    try:
        build_search_data(args.data_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Cannot preprocess: {error}\n")
