"""Evaluate saved query-specific relevance judgments against three rankings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import platform
import time
import tracemalloc

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.config import PREPROCESSING_DIR
from project.search import load_data, search


def metrics_at_k(retrieved, relevant, k=10):
    """P@K uses K even if fewer than K results exist; recall uses all qrels."""
    if k < 1:
        raise ValueError("K must be positive.")
    relevant = set(relevant)
    hits = len(set(retrieved[:k]) & relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": hits}


def evaluate(data_dir, judgments_file, output_dir, k=10, repeats=20):
    """Read explicit qrels, never infer relevance from document topic labels.

    Timings use warmed in-memory search, excluding input/printing and loading.
    Tracemalloc measures peak traced allocations during load plus one query
    pass, NOT process RSS. Memory and latency are measured separately.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    judgments = json.loads(Path(judgments_file).read_text(encoding="utf-8"))
    queries = judgments["queries"]
    if not queries or k < 1 or repeats < 1:
        raise ValueError("Provide queries and positive K/repeats.")
    query_ids = [q["id"] for q in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Query IDs must be unique.")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "corpus_sha256": hashlib.sha256((data_dir / "Document_Collection.json").read_bytes()).hexdigest(),
              "judgment_provenance": judgments["provenance"], "k": k, "queries": len(queries),
              "repeats": repeats, "python": platform.python_version(),
              "platform": platform.platform(), "processor": platform.processor(), "systems": {}}
    for mode in ("tfidf", "pagerank", "tspr"):
        tracemalloc.start()
        started = time.perf_counter()
        data = load_data(data_dir, mode=mode)
        load_seconds = time.perf_counter() - started
        for query in queries:
            if not query["relevant"] or not set(query["relevant"]).issubset(data["documents"]):
                raise ValueError(f"{query['id']}: qrels must contain known approved IDs and a positive example.")
            search(query["query"], data, top_k=k, mode=mode)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples, rows = [], []
        for query in queries:
            result = search(query["query"], data, top_k=k, mode=mode)
            retrieved = [record["doc_id"] for record in result]
            values = metrics_at_k(retrieved, query["relevant"], k)
            rows.append({"id": query["id"], "query": query["query"], "relevant_count": len(query["relevant"]),
                         "returned_count": len(retrieved), "retrieved": retrieved, **values})
            for _ in range(repeats):
                started = time.perf_counter()
                search(query["query"], data, top_k=k, mode=mode)
                samples.append((time.perf_counter() - started) * 1000)
        report["systems"][mode] = {
            "macro_precision": float(np.mean([r["precision"] for r in rows])),
            "macro_recall": float(np.mean([r["recall"] for r in rows])),
            "macro_f1": float(np.mean([r["f1"] for r in rows])),
            "mean_query_ms": float(np.mean(samples)), "median_query_ms": float(np.median(samples)),
            "p95_query_ms": float(np.percentile(samples, 95)),
            "peak_traced_mib": peak_bytes / 1024**2, "instrumented_load_seconds": load_seconds,
            "per_query": rows,
        }
        print(f"{mode}: P@{k}={report['systems'][mode]['macro_precision']:.3f}, "
              f"R@{k}={report['systems'][mode]['macro_recall']:.3f}, "
              f"F1@{k}={report['systems'][mode]['macro_f1']:.3f}", flush=True)
        del data
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.data_dir, args.judgments, args.output_dir, args.k, args.repeats)
