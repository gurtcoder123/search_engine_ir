"""Sparse PageRank power iteration and one teleport distribution per topic."""

import numpy as np
from scipy.sparse import csr_matrix

from project.config import TOPICS
from project.collection.crawler import normalize_title


def build_link_matrix(documents, doc_ids):
    """Internal binary links: row=source, column=destination, in text row order.

    Match saved Wikipedia titles; ignore outside/unresolved targets and self
    links. Record every matching edge, not just the links followed by BFS.
    """
    title_to_row = {normalize_title(documents[d]["title"]): i for i, d in enumerate(doc_ids)}
    rows, columns = [], []
    for row, doc_id in enumerate(doc_ids):
        destinations = set()
        for title in documents[doc_id].get("outgoing_links", []):
            destination = title_to_row.get(normalize_title(title))
            if destination is not None and destination != row:
                destinations.add(destination)
        for destination in sorted(destinations):
            rows.append(row)
            columns.append(destination)
    return csr_matrix((np.ones(len(rows)), (rows, columns)),
                      shape=(len(doc_ids), len(doc_ids)))


def pagerank(links, teleport, damping=0.85, tolerance=1e-8, max_iterations=200):
    """Return rank plus convergence details, without a dense N-by-N matrix.

    Dangling mass goes to the SAME teleport distribution. Topic scores may
    legitimately be zero for nodes unreachable from that topic's seed set.
    """
    teleport = np.asarray(teleport, dtype=float)
    if not 0 <= damping < 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Use 0 <= damping < 1, positive tolerance and at least one iteration.")
    if (links.shape != (teleport.size, teleport.size) or teleport.size == 0
            or not np.isfinite(teleport).all() or np.any(teleport < 0)
            or not np.isclose(teleport.sum(), 1)):
        raise ValueError("Teleport must be a probability vector matching the link matrix.")
    out_degree = np.asarray(links.sum(axis=1)).ravel()
    dangling = out_degree == 0
    rank = teleport.copy()
    for iteration in range(1, max_iterations + 1):
        share = np.divide(rank, out_degree, out=np.zeros_like(rank), where=~dangling)
        new_rank = damping * (links.T @ share + rank[dangling].sum() * teleport)
        new_rank += (1 - damping) * teleport
        change = float(np.max(np.abs(new_rank - rank)))
        rank = new_rank
        if change < tolerance:
            return rank, {"iterations": iteration, "max_change": change, "converged": True}
    return rank, {"iterations": iteration, "max_change": change, "converged": False}


def build_rankings(documents, doc_ids, topics=TOPICS, damping=0.85,
                   tolerance=1e-8, max_iterations=200):
    """Compute global baseline and K topic vectors; all share doc_ids order."""
    if not doc_ids:
        raise ValueError("No approved documents. Run reviewer.py first.")
    labels = [documents[d].get("topic") for d in doc_ids]
    if any(label not in topics for label in labels):
        raise ValueError("Every approved document needs a valid manual topic label.")
    links = build_link_matrix(documents, doc_ids)
    settings = dict(damping=damping, tolerance=tolerance, max_iterations=max_iterations)
    global_rank, global_info = pagerank(links, np.full(len(doc_ids), 1 / len(doc_ids)), **settings)
    topic_ranks, topic_info = [], {}
    for topic in topics:
        teleport = np.array([label == topic for label in labels], dtype=float)
        if teleport.sum() == 0:
            raise ValueError(f"Approve at least one document for topic: {topic}")
        teleport /= teleport.sum()
        rank, info = pagerank(links, teleport, **settings)
        topic_ranks.append(rank)
        topic_info[topic] = info
    metadata = {"topics": list(topics), "doc_ids": list(doc_ids), "labels": labels,
                **settings, "internal_links": links.nnz,
                "dangling_documents": int(np.count_nonzero(np.asarray(links.sum(axis=1)).ravel() == 0)),
                "global": global_info, "per_topic": topic_info,
                "dangling_policy": "redistribute to the run's teleport vector"}
    return np.array(topic_ranks), global_rank, metadata
