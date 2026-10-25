"""Query-topic probabilities from manually labelled articles, not embeddings."""

import numpy as np

from project.config import TOPICS
from project.indexing.normalize import tokenize


def build_topic_model(documents, term_counts, term_to_col, topics=TOPICS, alpha=1.0):
    """Aggregate RAW counts and add alpha to every vocabulary term per topic.

    documents is an approved doc_id -> record mapping. Rows follow topics;
    columns follow the existing text vocabulary. Priors are uniform.
    """
    if alpha <= 0:
        raise ValueError("Topic smoothing alpha must be positive.")
    counts = np.zeros((len(topics), len(term_to_col)), dtype=float)
    doc_counts = np.zeros(len(topics), dtype=int)
    for doc_id, terms in term_counts.items():
        topic = documents[doc_id].get("topic")
        if topic not in topics:
            raise ValueError(f"{doc_id} needs a valid manual topic label.")
        row = topics.index(topic)
        doc_counts[row] += 1
        for term, count in terms.items():
            counts[row, term_to_col[term]] += count
    missing = [topic for topic, count in zip(topics, doc_counts) if count == 0]
    if missing:
        raise ValueError(f"Approve at least one document per topic. Missing: {', '.join(missing)}")
    if not term_to_col:
        raise ValueError("The approved corpus has no searchable words.")
    totals = counts.sum(axis=1)
    probabilities = (counts + alpha) / (totals[:, None] + alpha * len(term_to_col))
    model = {"topics": list(topics), "alpha": alpha,
             "priors": [1 / len(topics)] * len(topics),
             "word_totals": totals.astype(int).tolist(), "doc_counts": doc_counts.tolist()}
    return np.log(probabilities), model


def query_topic_weights(query, term_to_col, log_probs, priors):
    """P(topic|query) proportional to prior * product P(query word|topic).

    Repeated words count repeatedly; unknown words are ignored. Calculate in
    log space to avoid underflow. A query with no known terms gets the priors.
    """
    columns = [term_to_col[t] for t in tokenize(query) if t in term_to_col]
    scores = np.log(priors) + log_probs[:, columns].sum(axis=1)
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()
