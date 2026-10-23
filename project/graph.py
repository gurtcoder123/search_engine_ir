"""Build the final sparse link matrix after document review."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

if __package__:
    from .config import REVIEWED_DIR, TOPICS
    from .crawler import Document
else:
    from config import REVIEWED_DIR, TOPICS
    from crawler import Document


def build_incidence_matrix(documents, adjacency_lists):
    """Return binary adjacency: row=source, column=destination.

    Row/column i corresponds to documents[i]. Unknown targets and self-links
    are excluded; documents with no remaining links retain zero rows.
    """
    doc_ids = [document.doc_id for document in documents]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Document IDs must be unique")
    doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
    rows, columns = [], []
    for source in doc_ids:
        for target in sorted(set(adjacency_lists.get(source, []))):
            if target in doc_to_row and target != source:
                rows.append(doc_to_row[source])
                columns.append(doc_to_row[target])
    return csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(documents), len(documents)), dtype=np.uint8,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REVIEWED_DIR)
    args = parser.parse_args()
    with (args.input_dir / "documents.jsonl").open(encoding="utf-8") as stream:
        documents = [Document(**json.loads(line)) for line in stream if line.strip()]
    if not documents or any(document.topic not in TOPICS for document in documents):
        parser.error("Run reviewer.py first: the corpus must be nonempty and every document must have a topic")
    adjacency_lists = json.loads((args.input_dir / "adjacency_lists.json").read_text(encoding="utf-8"))
    incidence_matrix = build_incidence_matrix(documents, adjacency_lists)
    save_npz(args.input_dir / "incidence_matrix.npz", incidence_matrix)
    doc_ids = [document.doc_id for document in documents]
    (args.input_dir / "graph_manifest.json").write_text(json.dumps({
        "matrix_kind": "directed_binary_adjacency",
        "orientation": "row=source, column=destination; self-links excluded",
        "doc_ids": doc_ids,
        "doc_to_row": {doc_id: row for row, doc_id in enumerate(doc_ids)},
    }, indent=2), encoding="utf-8")
    outgoing = np.asarray(incidence_matrix.sum(axis=1)).ravel()
    incoming = np.asarray(incidence_matrix.sum(axis=0)).ravel()
    print(f"Documents: {len(documents)}; matrix: {incidence_matrix.shape}; edges: {incidence_matrix.nnz}")
    print(f"Dangling pages: {int((outgoing == 0).sum())}; isolated pages: {int(((outgoing == 0) & (incoming == 0)).sum())}")
    print(f"Saved matrix and row mapping to {args.input_dir.resolve()}")


if __name__ == "__main__":
    main()
