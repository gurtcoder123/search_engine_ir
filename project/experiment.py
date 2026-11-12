"""Rebuild clean/control/spam-farm copies and measure the ranking changes."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project.config import PREPROCESSING_DIR
from project.evaluate import evaluate, metrics_at_k, ndcg_at_k
from project.preprocess import build_search_data
from project.search import DEFAULT_LEXICAL_WEIGHT, MODES, load_data, search
from project.indexing.normalize import tokenize


def save_snapshot(records, destination):
    """Write a self-contained experimental copy, never the source collection."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["file_path"] = f"Documents/{record['topic']}/{record['doc_id']}.json"
        path = destination / record["file_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "Document_Collection.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def target_positions(data, queries, target_id, k, lexical_weight):
    """Track the target even when it is below the displayed top K."""
    rows = {}
    for mode in MODES:
        rows[mode] = []
        for query in queries:
            results = search(query["query"], data, top_k=len(data["doc_ids"]),
                             mode=mode, lexical_weight=lexical_weight)
            position = next((i for i, r in enumerate(results, 1) if r["doc_id"] == target_id), None)
            record = results[position - 1] if position is not None else None
            rows[mode].append({"query_id": query["id"], "rank": position,
                               "in_top_k": position is not None and position <= k,
                               "relevant": target_id in query["relevant"],
                               "score": record["score"] if record else None,
                               "cosine": record["cosine"] if record else None,
                               "link_score": record["link_score"] if record else None})
    return rows


def weight_sweep(data, queries, k):
    """Descriptive development-set sensitivity, not held-out model selection."""
    rows = []
    for weight in (0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0):
        values = []
        for query in queries:
            retrieved = [r["doc_id"] for r in search(query["query"], data, top_k=k,
                                                   mode="hybrid", lexical_weight=weight)]
            values.append({**metrics_at_k(retrieved, query["relevant"], k),
                           "ndcg": ndcg_at_k(retrieved, query["relevant"], k)})
        rows.append({"lexical_weight": weight,
                     **{f"macro_{metric}": float(np.mean([v[metric] for v in values]))
                        for metric in ("precision", "recall", "f1", "ndcg")}})
    return rows


def run_experiment(data_dir, judgments_file, output_dir, target_id,
                   farm_size=10, k=10, repeats=20, lexical_weight=DEFAULT_LEXICAL_WEIGHT):
    """Only the farm's outgoing links differ between control and attacked.

    Farm text duplicates the target's saved text, with the same manual topic.
    Farm pages are synthetic nonrelevant duplicates, never new relevant items.
    No real Wikipedia page is edited and no network request is made.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if data_dir == output_dir or data_dir in output_dir.parents or output_dir in data_dir.parents:
        raise ValueError("Experiment output and source data must be separate directory trees.")
    if farm_size < 1 or k < 1 or repeats < 1 or not 0 <= lexical_weight <= 1:
        raise ValueError("Use positive farm size/K/repeats and lexical weight in [0, 1].")
    source = data_dir / "Document_Collection.json"
    original = source.read_bytes()
    approved = [r for r in json.loads(original) if r.get("approved") is True]
    documents = {r["doc_id"]: r for r in approved}
    if target_id not in documents:
        raise ValueError("Target must be an approved document ID.")
    target = documents[target_id]
    judgments_file = Path(judgments_file)
    judgments = json.loads(judgments_file.read_text(encoding="utf-8"))
    queries = judgments["queries"]
    if (not queries or len({q["id"] for q in queries}) != len(queries)
            or any(not q["relevant"] or not set(q["relevant"]).issubset(documents) for q in queries)):
        raise ValueError("Provide unique queries with nonempty qrels from the clean corpus.")
    # Choose/record an irrelevant target with a lexical match before measuring attack outcomes.
    target_terms = set(tokenize(target["text"]))
    affected = [q["id"] for q in queries if target_id not in q["relevant"]
                and target_terms.intersection(tokenize(q["query"]))]
    if not affected:
        raise ValueError("Target needs a lexical match to at least one query for which it is nonrelevant.")
    farm = []
    for number in range(1, farm_size + 1):
        spam = deepcopy(target)
        spam.update(page_id=-number, doc_id=f"spam-{number:03}",
                    title=f"Synthetic spam farm page {number:03}",
                    url=f"https://spam.invalid/{number:03}", revision_id=None,
                    preview="Synthetic duplicate used only for the local attack experiment.",
                    outgoing_links=[])
        if spam["doc_id"] in documents or any(r["title"] == spam["title"] for r in approved):
            raise ValueError("Synthetic ID/title collides with an existing document.")
        farm.append(spam)
    control = deepcopy(approved) + farm
    attacked = deepcopy(control)
    for spam in attacked[len(approved):]:
        spam["outgoing_links"] = [target["title"]]
    conditions = {"clean": deepcopy(approved), "control": control, "attacked": attacked}
    output_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the supplied judgments; never add the target/farm to relevant sets.
    frozen_qrels = output_dir / "judgments.json"
    frozen_qrels.write_bytes(judgments_file.read_bytes())
    summary = {"created_at": datetime.now(timezone.utc).isoformat(),
               "source_corpus_sha256": hashlib.sha256(original).hexdigest(),
               "judgments_sha256": hashlib.sha256(frozen_qrels.read_bytes()).hexdigest(),
               "judgment_provenance": judgments["provenance"],
               "target": {"doc_id": target_id, "title": target["title"], "topic": target["topic"]},
               "farm_size": farm_size, "k": k, "lexical_weight": lexical_weight,
               "target_nonrelevant_matching_queries": affected,
               "attack": "Same-topic duplicate-text farm; each synthetic page links only to the target.",
               "control": "Identical farm text/labels/IDs and corpus size, but farm pages have no outgoing links.",
               "caveat": "Clean-to-attacked includes corpus/IDF/topic-model changes; control-to-attacked isolates the added links.",
               "weight_sweep_scope": "Descriptive development-set sensitivity; default weight fixed at 0.9 before measurements.",
               "conditions": {}}
    for name, records in conditions.items():
        print(f"\n--- {name}: {len(records)} documents ---", flush=True)
        snapshot = output_dir / name
        save_snapshot(records, snapshot)
        started = time.perf_counter()
        ranking = build_search_data(snapshot)
        build_seconds = time.perf_counter() - started
        metrics = evaluate(snapshot, frozen_qrels, snapshot / "evaluation", k, repeats, lexical_weight)
        data = load_data(snapshot, mode="hybrid")
        row = data["doc_to_row"][target_id]
        topic_row = data["topic_model"]["topics"].index(target["topic"])
        summary["conditions"][name] = {
            "documents": len(records), "internal_links": ranking["internal_links"],
            "vocabulary": len(data["term_to_col"]), "tfidf_nonzeros": data["tfidf_matrix"].nnz,
            "build_seconds_excluding_imports": build_seconds,
            "convergence": {"global": ranking["global"], "topics": ranking["per_topic"]},
            "global_target_score": float(data["global_pagerank"][row]),
            "topic_target_score": float(data["topic_pagerank"][topic_row, row]),
            "systems": metrics["systems"],
            "target_positions": target_positions(data, queries, target_id, k, lexical_weight),
            "weight_sweep": weight_sweep(data, queries, k)}
    summary["changes"] = {}
    for before, after in (("clean", "attacked"), ("control", "attacked")):
        summary["changes"][f"{before}_to_{after}"] = {
            mode: {metric: summary["conditions"][after]["systems"][mode][metric]
                   - summary["conditions"][before]["systems"][mode][metric]
                   for metric in ("macro_precision", "macro_recall", "macro_f1", "macro_ndcg")}
            for mode in MODES}
    summary["source_collection_unchanged"] = source.read_bytes() == original
    if not summary["source_collection_unchanged"]:
        raise RuntimeError("Source collection changed during the experiment.")
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved experiment summary to {path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PREPROCESSING_DIR)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/attack"))
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--farm-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--lexical-weight", type=float, default=DEFAULT_LEXICAL_WEIGHT)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.judgments, args.output_dir, args.target_id,
                   args.farm_size, args.k, args.repeats, args.lexical_weight)
