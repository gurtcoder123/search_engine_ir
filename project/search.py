"""Search with TF-IDF, global PageRank or query-weighted topic PageRank."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix, load_npz

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.config import PREPROCESSING_DIR
from project.indexing.normalize import tokenize
from project.ranking.topic_model import query_topic_weights


def load_data(data_dir=PREPROCESSING_DIR, mode="tfidf"):
    """Load saved metadata and indexes once, not on every query."""
    data_dir = Path(data_dir)
    records = json.loads((data_dir / "Document_Collection.json").read_text(encoding="utf-8"))
    documents = {r["doc_id"]: r for r in records if r.get("approved") is True}
    if not documents:
        raise ValueError("No approved documents. Run reviewer.py first.")
    index_dir = data_dir / "Index"
    data = {"documents": documents}
    for name in ("doc_ids", "term_to_col", "inverted_index", "term_counts"):
        data[name] = json.loads((index_dir / f"{name}.json").read_text(encoding="utf-8"))
    if set(data["doc_ids"]) != set(documents):
        raise ValueError("Approved selection changed. Rerun preprocess.py.")
    data["doc_to_row"] = {doc_id: row for row, doc_id in enumerate(data["doc_ids"])}
    data["idf"] = np.load(index_dir / "idf.npy", allow_pickle=False)
    data["tfidf_matrix"] = load_npz(index_dir / "tfidf_matrix.npz")
    shape = (len(data["doc_ids"]), len(data["term_to_col"]))
    if data["tfidf_matrix"].shape != shape or data["idf"].shape != (shape[1],):
        raise ValueError("Index dimensions disagree. Rebuild the index and TF-IDF matrix.")
    if mode not in {"tfidf", "pagerank", "tspr"}:
        raise ValueError("Mode must be tfidf, pagerank or tspr.")
    if mode != "tfidf":
        ranking = json.loads((index_dir / "ranking.json").read_text(encoding="utf-8"))
        labels = [documents[d].get("topic") for d in data["doc_ids"]]
        if ranking["doc_ids"] != data["doc_ids"] or ranking["labels"] != labels:
            raise ValueError("Rank order or labels changed. Rerun preprocess.py.")
        if not all(i["converged"] for i in [ranking["global"], *ranking["per_topic"].values()]):
            raise ValueError("Saved PageRank did not converge. Rerun preprocess.py.")
        data["global_pagerank"] = np.load(index_dir / "global_pagerank.npy", allow_pickle=False)
        data["topic_pagerank"] = np.load(index_dir / "topic_pagerank.npy", allow_pickle=False)
        data["topic_log_probs"] = np.load(index_dir / "topic_log_probs.npy", allow_pickle=False)
        data["topic_model"] = json.loads((index_dir / "topic_model.json").read_text(encoding="utf-8"))
        k = len(ranking["topics"])
        if (data["global_pagerank"].shape != (shape[0],)
                or data["topic_pagerank"].shape != (k, shape[0])
                or data["topic_log_probs"].shape != (k, shape[1])
                or data["topic_model"]["topics"] != ranking["topics"]):
            raise ValueError("Topic/rank dimensions disagree. Rerun preprocess.py.")
    return data


def make_query_vector(query, term_to_col, idf):
    """Use saved corpus IDF and the same log-TF/L2 rules as documents."""
    counts = {}
    for term in tokenize(query):
        if term in term_to_col:
            counts[term] = counts.get(term, 0) + 1
    columns, weights = [], []
    for term, count in counts.items():
        column = term_to_col[term]
        weight = (1 + np.log10(count)) * idf[column]
        if weight != 0:
            columns.append(column)
            weights.append(weight)
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights = np.array(weights) / norm
    vector = csr_matrix((weights, ([0] * len(columns), columns)),
                        shape=(1, len(term_to_col)), dtype=float)
    return vector, list(counts)


def search(query, data, top_k=10, mode="tfidf"):
    """Find lexical candidates, then rank by cosine OR precomputed link scores.

    TSPR's query weights mix all topic vectors. No PageRank iteration happens
    at query time, and no arbitrary TF-IDF/link fusion weight is introduced.
    """
    if mode not in {"tfidf", "pagerank", "tspr"}:
        raise ValueError("Mode must be tfidf, pagerank or tspr.")
    query_vector, query_terms = make_query_vector(query, data["term_to_col"], data["idf"])
    if query_vector.nnz == 0 or top_k < 1:
        return []
    candidates = set()
    for term in query_terms:
        for doc_id, count in data["inverted_index"].get(term, []):
            candidates.add(doc_id)
    rows = sorted(data["doc_to_row"][doc_id] for doc_id in candidates)
    if not rows:
        return []
    # Only the candidate score column becomes dense, never the document-term matrix.
    cosine = (data["tfidf_matrix"][rows] @ query_vector.T).toarray().ravel()
    if mode == "tspr":
        model = data["topic_model"]
        weights = query_topic_weights(query, data["term_to_col"], data["topic_log_probs"], model["priors"])
        scores = weights @ data["topic_pagerank"][:, rows]
    elif mode == "pagerank":
        scores = data["global_pagerank"][rows]
    else:
        scores = cosine
    # A zero topic-rank is legitimate; only zero lexical matches are filtered.
    ranked = sorted(((row, score, text_score) for row, score, text_score in zip(rows, scores, cosine)
                     if text_score > 0), key=lambda item: item[1], reverse=True)
    results = []
    for row, score, text_score in ranked[:top_k]:
        doc_id = data["doc_ids"][row]
        document = data["documents"][doc_id]
        results.append({"doc_id": doc_id, "score": float(score), "cosine": float(text_score),
                        "title": document["title"], "preview": document["preview"],
                        "url": document["url"], "topic": document.get("topic")})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=("tfidf", "pagerank", "tspr"), default="tfidf")
    args = parser.parse_args()
    try:
        data = load_data(args.data_dir, args.mode)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Cannot start search: {error}\nReview and build the offline indexes first.\n")
    print(f"Loaded {len(data['doc_ids'])} documents. Ranking mode: {args.mode}.")
    while True:
        try:
            query = input("\nSearch (q to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if query.lower() in {"q", "quit", "exit"}:
            return
        if not query:
            continue
        if args.mode == "tspr":
            model = data["topic_model"]
            weights = query_topic_weights(query, data["term_to_col"], data["topic_log_probs"], model["priors"])
            print("Query topics: " + ", ".join(f"{t}={w:.1%}" for t, w in zip(model["topics"], weights)))
        results = search(query, data, args.top_k, args.mode)
        if not results:
            print("No matching documents.")
        for position, result in enumerate(results, start=1):
            label = "cosine" if args.mode == "tfidf" else args.mode
            print(f"\n{position}. {result['title']} | {label}={result['score']:.4f}")
            print(result["url"])
            print(result["preview"])


if __name__ == "__main__":
    main()
