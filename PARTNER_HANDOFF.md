# Dependencies for query retrieval and PageRank/TSPR

This is the shared implementation agreement between Vignesh (A: collection,
preprocessing and indexing; D: evaluation and attacks) and the partner
implementing B (query retrieval/topic estimation) and C (link analysis/ranking).
Integration, checks and submission work are shared.

**This file describes the agreed handoff, not a completed index.** The repo
currently contains the collector, manual reviewer, graph builder and tests.
The shared preprocessing function and text-index artifacts are still to be
implemented by Vignesh. No completed final corpus or retrieval results are
implied by the presence of this document.

## 1. What the partner receives

| Input | Shared name/type in memory | Used for | Current status |
| --- | --- | --- | --- |
| Final reviewed article records | `documents: list[Document]` | Metadata, topics, result display | Collector/reviewer exist; approved corpus still needs preparation |
| Document ordering | `doc_ids: list[str]`, `doc_to_row: dict[str, int]` | Align every document-based array | Graph builder exports these for a reviewed corpus |
| Topic order and labels | `topics: list[str]`; `document.topic` | Topic estimation and teleport vectors | Topic names fixed; labels require manual review |
| Shared text processing | `preprocess(text: str) -> list[str]` | Query normalization | Pending implementation |
| Vocabulary | `term_to_col: dict[str, int]` | Query vector columns | Pending indexing |
| Processed raw term counts | `term_counts: dict[str, dict[str, int]]` | Topic language model | Pending preprocessing |
| Inverted index | `inverted_index: dict[str, list[tuple[str, int]]]` | Candidate generation | Pending indexing |
| DF and IDF | NumPy arrays `df`, `idf`, each shape `(V,)` | Query TF-IDF | Pending indexing |
| Normalized document TF-IDF | `tfidf_matrix`, SciPy CSR, shape `(N, V)` | Cosine similarity | Pending indexing |
| Binary link matrix | `adjacency_matrix`, SciPy CSR, shape `(N, N)` | Global PageRank and TSPR | `project/graph.py` builds it after review |

Here, **N is the final reviewed document count**, and V is vocabulary size.
Do not hard-code N as 100 or 200. We collect up to 40 candidates from each of
five seed groups, then manually choose and label the final corpus.

The partner can implement B/C against these names, types and shapes before
the real artifacts arrive, using small hand-written examples for development.

## 2. Documents, topics and ordering

The existing `Document` dataclass is importable from the repository root:

```python
from project.crawler import Document, TOPICS
```

Relevant fields are:

```text
doc_id: str                  stable project ID, e.g. doc-14343887
page_id: int                 Wikipedia page ID
title: str                   canonical article title
url: str                     result link
text: str                    full saved plain article text
preview: str                 first paragraph for display
topic: str | None            actual manually assigned topic
discovered_topic: str | None seed provenance, not an actual label
revision_id: int | None      revision metadata recorded at collection
collected_at: str            collection timestamp
outgoing_links: list[str]    original outgoing article titles
```

Treat `doc_id` as an opaque string. It is deliberately our own namespace rather
than Wikipedia's, so later synthetic documents (the Part D spam farm) can take
IDs no real page could collide with. Never parse it and never use it as a row
index: `doc_to_row` is the only way to find a document's row.

During crawling, `topic` is always None. For the final corpus used by B/C,
each accepted document has exactly one topic from this fixed order:

```python
topics = [
    "technology",
    "science",
    "health_and_medicine",
    "sports",
    "politics",
]
```

Use `document.topic`, never `discovered_topic`, to train the topic model or
choose teleport seeds.

All document rows follow the final reviewed document list:

```python
doc_ids = [document.doc_id for document in documents]
doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
```

For row i, the document TF-IDF vector, graph row/column, global PageRank entry
and each topic-rank entry must all refer to `documents[i]`. A page ID is an
identifier, not an array index. Do not sort/reorder one artifact independently.

`WebCrawler` now takes a shared `DocumentCollection` in its constructor and
exposes `crawl()`, which returns `(documents, adjacency_lists)`. B/C should
still depend on the saved artifacts below rather than on the crawler itself.

## 3. One shared preprocessing function

Vignesh will provide `preprocess(text: str) -> list[str]`, planned in
`project/preprocessing.py`. That module does not exist yet. The partner should
reuse this function when available instead of maintaining a second tokenizer.

The agreed processing sequence is:

1. Unicode normalization and consistent apostrophe handling.
2. Lowercasing.
3. Possessive removal.
4. Word/number tokenization; most punctuation and hyphens are boundaries.
5. Removal using one fixed English stop-word list.
6. Porter stemming.

Keep numbers and do not automatically discard short or rare terms. Spelling
correction, synonym expansion and positional/phrase retrieval are deferred.
The exact tokenizer implementation and stop-word file will be delivered with
the shared function; they should not be guessed independently.

Stemming occurs before vocabulary construction, TF counting and DF counting.
For example, `computer`, `computers` and `computing` map to the term `comput`.
That stem has one vocabulary column. Display titles/previews remain readable
original text; they do not display the stemmed tokens.

## 4. Term counts, postings and vocabulary

Illustrative data (IDs and values here are examples, not measured artifacts):

```python
term_to_col = {"chip": 0, "comput": 1, "softwar": 2}

term_counts = {
    "doc-101": {"chip": 3, "comput": 5},
    "doc-202": {"comput": 2, "softwar": 4},
}

inverted_index = {
    "chip": [("doc-101", 3)],
    "comput": [("doc-101", 5), ("doc-202", 2)],
    "softwar": [("doc-202", 4)],
}
```

Counts are **raw integer occurrences after preprocessing**, not log-scaled TF,
TF-IDF weights or normalized values. Each posting list contains a document
once. DF is the number of documents containing the processed term, not its
total number of occurrences.

`df[j]`, `idf[j]` and column j of `tfidf_matrix` all refer to the term whose
`term_to_col` value is j. The same vocabulary is used for document and query
vectors. JSON stores tuples as lists; a posting loaded as `[doc_id, count]`
has the same two fields and can be unpacked normally.

## 5. Exact TF-IDF and cosine agreement

For a raw count c:

```text
TF(c) = 1 + log10(c), when c > 0
TF(0) = 0

IDF(t) = log10(N / DF(t))
weight(t, document) = TF(count(t, document)) * IDF(t)
```

Use the same formula for query weights, with counts from the query and IDF
from the saved document collection. We chose base 10 and the textbook
`1 + log10(c)` formulation, not `log(1 + c)` or smoothed IDF.

L2-normalize document vectors offline:

```text
norm(d) = sqrt(sum_j weight(j, d)^2)
unit_d = d / norm(d)
```

**The supplied `tfidf_matrix` is already L2-normalized.** Build and normalize
the query vector once, then cosine similarity is `unit_q dot unit_d`.
Do not divide an already-normalized document row by its original norm again.
Original norms are optional metadata, not a required query-time input.

Zero rows stay zero. Ignore terms outside the vocabulary for now. If the query
has no nonzero weights, return no lexical results instead of dividing by zero.
Unsmoothed IDF also gives weight zero to terms appearing in every document.

Generate candidates using the **union** of postings for query terms. Score
candidate rows; a document need not contain every query term. PageRank does
not generate lexical candidates or replace the inverted index.

## 6. Inputs to the query-topic model

The partner receives processed `term_counts` and the final manual labels.
Aggregate the counts of all documents assigned to a topic:

```text
Two Technology documents contain comput 5 times and 2 times
    -> Technology topic count for comput = 7
```

Use those counts to build the smoothed unigram topic model discussed for the
Haveliwala-style replication. This produces five weights P(topic | query),
in the fixed `topics` order. Smoothing and prior choices belong to the partner's
topic-model implementation and must be documented. No alpha/prior value is
fixed by this handoff.

The model uses occurrence counts, not document TF-IDF weights or averages of
normalized document vectors. A TF-IDF centroid model would be a separate
alternative; it is not the agreed primary query-topic method.

## 7. Inputs to PageRank and TSPR

The graph builder receives only the accepted documents and filtered adjacency
lists. It excludes duplicate edges, self-links and destinations outside the
final corpus. Isolated/dangling documents remain present as zero rows.

The graph is a binary CSR matrix A with:

```text
A[i, j] = 1 if document i links to document j; otherwise 0
```

The file is currently named `incidence_matrix.npz` for compatibility with our
discussion, but the mathematical object is an adjacency matrix. Load it into
the shared variable `adjacency_matrix`:

```python
from scipy.sparse import load_npz

adjacency_matrix = load_npz("project/data/reviewed/incidence_matrix.npz")
```

The supplied matrix is **not row-normalized**. The partner handles floating
point transition probabilities, dangling mass and ranking iterations:

- Global PageRank: uniform teleportation and uniform dangling redistribution.
- TSPR: one teleport vector per topic, uniform over that topic's reviewed
  documents; redistribute dangling mass using that same topic vector.
- Start with damping d = 0.85, maximum absolute change below 1e-8, and at most
  200 iterations. Report if the iteration limit is reached without convergence.
- Compute scores for every document offline. Query time selects/mixes existing
  scores for lexical candidates; it does not rerun PageRank.

Suggested outputs from the partner:

```text
global_pagerank: shape (N,)
topic_pagerank:  shape (5, N)
query_topics:   shape (5,)
```

Rows of `topic_pagerank` follow `topics`; columns follow `doc_ids`.
The query-specific link score for document i is:

```text
sum_k query_topics[k] * topic_pagerank[k, i]
```

A topic vector has a coordinate per document; a TF-IDF vector has a coordinate
per vocabulary term. They are mixed through ranking scores, not by calculating
cosine similarity between a TF-IDF query and a PageRank vector.

## 8. Saved files and loading

Existing graph/review exports are under `project/data/reviewed/`:

- `documents.jsonl`: one accepted Document record per line, including full text,
  manual topic and preview.
- `adjacency_lists.json`: source doc IDs mapped to destination doc IDs.
- `incidence_matrix.npz`: binary CSR graph, built by `project/graph.py`.
- `graph_manifest.json`: final `doc_ids`, `doc_to_row` and graph orientation.

The following text-index layout is the planned export under
`project/data/index/`; it has not been generated yet:

- `manifest.json`: matching `doc_ids`, `doc_to_row` and `topics`; weighting and
  normalization settings.
- `term_to_col.json`, `term_counts.json`, `inverted_index.json`.
- `df.npy`, `idf.npy`.
- `tfidf_matrix.npz`: L2-normalized document rows.

Use JSON for mappings/counts/postings and NumPy/SciPy formats for numerical
artifacts. Load once at application startup, not for each query. All artifacts
must come from the same frozen reviewed corpus. A change in corpus selection
requires rebuilding the graph, vocabulary/IDF/index and rank vectors together.

Full text is retained on disk for reproducibility and rebuilding. Results can
display the saved title, first-paragraph preview and URL without new API calls.
`project/data/` is Git-ignored: the code repository does not automatically
contain downloaded corpora or built artifacts. Those files must be shared
separately when available; a loader alone does not provide the real data.

## 9. What can start now, and what is still open

The partner can implement query vector construction, candidate union, sparse
cosine scoring, topic modelling and PageRank/TSPR against small examples using
this contract. Real corpus testing waits for the reviewed data and index.

Vignesh will provide the preprocessing implementation and example input/output
tokens before final query integration. A shared `contracts.py` may be extracted
later; it does not exist today. Do not import pending modules as if they already
exist.

Final text-score/TSPR-score normalization and their fusion weight are still
undecided. In particular, 70/30 is not a fixed literature-prescribed choice.
Keep component scores available so fusion can be settled during integration.
HITS is not part of the required pipeline.

Evaluation queries, relevance judgements and spam-farm inputs will be prepared
with the D work after the clean B/C pipeline is connected.
