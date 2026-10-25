# Partner handoff — revised collection/review contract

This file records shared interfaces for the full project. The earlier A/B/C/D
partner ownership split is no longer assumed; implementation continues here
through retrieval, topic-sensitive ranking and attack evaluation.

## 1. Scope change: no graph implementation in A

A supplies saved article records and manual approval/topic decisions.

It does NOT supply a graph, adjacency lists/matrix, graph loader, resolved link
IDs, graph row labels or corpus fingerprints. graph.py and graph-only tests have
been removed. Ignore the previous graph export/load instructions.

outgoing_links holds raw article titles. Any link resolution and matrix/row
preparation required for ranking comes later, not in this crawler/reviewer.
There is no discovered_topic. Seed groups control crawl quotas, not labels.

## 2. Files and authoritative selection

Default collection root: project/Pre-Processing/ (Git-ignored).

- Document_Collection.json: a JSON ARRAY of ALL records, including rejected and
  unreviewed articles. It is the reference for which documents are included.
- Documents/doc-<page_id>.json: pending/rejected article records.
- Documents/<topic>/doc-<page_id>.json: approved, manually labelled records.

~~~text
page_id: int
doc_id: str
title: str
url: str
collected_at: str
revision_id: int | None
preview: str
text: str
outgoing_links: list[str]
topic: str | None
approved: bool | None
file_path: str                       relative to the collection root
~~~

**Include a document only when approved is True.** File existence, folder and
topic alone are not approval. Null means unreviewed; false means rejected.
Both stay in the array, with their article files retained for later re-review.

The current doc_id is doc-123, an opaque filename/record ID, not a matrix row.
Preserve saved IDs; use the text-index row mapping when accessing its matrix.

~~~python
import json
from pathlib import Path
from project.collection.crawler import Document

root = Path("project/Pre-Processing")
records = json.loads((root / "Document_Collection.json").read_text(encoding="utf-8"))
documents = [Document(**r) for r in records if r.get("approved") is True]
~~~

Use root / document.file_path for the article's actual location.
Review updates both JSON copies after each decision and sorts files into
topic folders. A topic change moves the same file; rejecting it moves it back
to Documents/. No document record is deleted.

Review may be incomplete. Check pending decisions, accepted counts and topic
coverage before treating the dataset as final. The course corpus needs at least
100 approved articles. Forty candidates per seed group is a crawl ceiling, not
a guarantee of 40 final articles for each manual topic.

This replaces the old data/candidates and data/reviewed JSONL interface.
Old local data is not automatically deleted or migrated.

## 3. Topic labels, text and raw links

Fixed topic order in project/config.py:

~~~python
topics = ["technology", "science", "health_and_medicine", "sports", "politics"]
~~~

A candidate starts with topic=None and approved=None. Only human approval
assigns exactly one topic from this list.

Default text AND preview are the first three nonempty paragraphs, not the
whole article. max_paragraphs/max_chars are configurable. Future indexing uses
the text actually saved; do not silently fetch a different article version.

outgoing_links retains all namespace-0 link titles returned by the API,
including continuation batches. It includes visited/unfollowed targets,
self-links and destinations outside the eventual approved corpus. It is NOT
a prefiltered internal graph, and BFS parent-child paths do not replace it.

Collected pages are not revisited/re-expanded. An unknown redirect alias may
need one request before its page ID reveals the duplicate.

The reviewer displays known incoming links by matching saved outgoing titles
against canonical titles. It makes no API backlink calls. Uncollected sources
and unresolved redirect aliases are not known incoming links.

## 4. Shared tokenization and stemming — implemented

The shared entry point now lives in project/indexing/normalize.py, not the previously
proposed preprocessing.py. Queries must import the same function:

~~~python
from project.indexing.normalize import tokenize
query_tokens = tokenize(query_text)
~~~

Rules currently implemented:

1. Unicode NFKC normalization and consistent apostrophes.
2. Lowercase the entire text; remove possessive 's.
3. Collapse dots in initial-style abbreviations (U.S.A. -> usa), without expansion.
4. Extract word/number tokens. Hyphenated terms stay together (covid-19);
   other punctuation/underscores are boundaries. Decimal dots are boundaries.
5. Remove words from the small project-defined project/indexing/stopwords.txt list.
6. Stem plain ASCII alphabetic words with NLTK's PorterStemmer in
   NLTK_EXTENSIONS mode. For example, computer/computers/computing -> comput.

Keep numbers and short/rare terms. Token order and repetitions are retained.
Stop-word removal happens before stemming. Hyphenated compounds, numbers,
mixed letter/number terms and non-ASCII words are not stemmed. NLTK is used
only for PorterStemmer, not tokenization or corpora; no resource downloads are
needed. Install the current project/requirements.txt. Recreate old unstemmed
token files before building the shared vocabulary/index.
Spelling correction, synonym/abbreviation expansion, embeddings and phrase
retrieval are deferred.

load_document(documentcollection) processes only approved documents and returns
doc_id -> tokens. It writes Pre-Processing/Tokens/<doc_id>.json containing doc_id
and tokens, without changing original article records or review decisions.
Read only token files for currently approved IDs; old files are not an approval
reference. No indexing is done in this module.

## 5. Inverted index and document TF-IDF — implemented

Run project/indexing/indexer.py after review and normalize.py. It reads current approval
flags, then loads token files only for those approved IDs. Missing approved
token files are an error; stale files for rejected/unreviewed articles are ignored.

Implemented outputs under Pre-Processing/Index/:

- inverted_index.json: term -> list of [doc_id, raw count]. Each document appears
  once per term. In memory build_inverted_index() uses (doc_id, count) tuples;
  JSON loads these as two-element lists, which can be unpacked the same way.
- df.json: term -> number of documents containing that term, a plain dictionary.
- term_counts.json: doc_id -> {term: raw integer count}, for topic models too.
- doc_ids.json: the indexed approved document order, including empty documents.

The indexer itself stores raw counts, with no positional postings. The separate
project/indexing/tfidf.py step creates log-TF/IDF weights and the sparse text matrix.

~~~python
import json
from pathlib import Path

index_dir = Path("project/Pre-Processing/Index")
inverted_index = json.loads((index_dir / "inverted_index.json").read_text(encoding="utf-8"))
df = json.loads((index_dir / "df.json").read_text(encoding="utf-8"))
term_counts = json.loads((index_dir / "term_counts.json").read_text(encoding="utf-8"))
doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
~~~

Run python project/indexing/tfidf.py AFTER indexer.py. Additional outputs in the same
Index/ folder are:

- term_to_col.json: processed term -> column, using a sorted vocabulary.
- doc_to_row.json: doc_id -> row, following the existing doc_ids.json order.
- df.npy and idf.npy: numerical arrays aligned with vocabulary columns.
- tfidf_matrix.npz: L2-normalized SciPy CSR, shape (N, V).

~~~python
import numpy as np
from scipy.sparse import load_npz

term_to_col = json.loads((index_dir / "term_to_col.json").read_text(encoding="utf-8"))
doc_to_row = json.loads((index_dir / "doc_to_row.json").read_text(encoding="utf-8"))
df_vector = np.load(index_dir / "df.npy", allow_pickle=False)
idf = np.load(index_dir / "idf.npy", allow_pickle=False)
tfidf_matrix = load_npz(index_dir / "tfidf_matrix.npz")
~~~

Use these arrays/mappings together, loading once at query-program startup.
df.json is still a term-keyed dictionary, while df.npy uses vocabulary order.
These are text-index mappings, not a graph/adjacency export from the crawler.

N is the APPROVED document count, not the length of Document_Collection.json.
DF counts documents, not occurrences. Derive N from len(doc_ids), keeping
approved empty documents in the total. Use that same order for future text
and rank vectors. No row mapping is supplied by the crawler; tfidf.py exports
the text-matrix mapping. No graph artifacts are built here.

Exact agreed formulas:

~~~text
TF(c) = 1 + log10(c) if c > 0, otherwise 0
IDF(t) = log10(N / DF(t))
weight(t, document) = TF(count(t, document)) * IDF(t)

norm(d) = sqrt(sum_j weight(j, d)^2)
unit_d = d / norm(d), leaving zero vectors zero
~~~

Do not substitute log(1+c), smoothed IDF or BM25.
Queries use saved corpus IDF, the same TF formula and L2 normalization.
Cosine is the dot product of normalized query/document vectors.
Candidates use the UNION of postings; search.py now implements this baseline.
The supplied matrix is ALREADY normalized; do not normalize it again using
the original document norms. Zero rows stay zero, and IDF=0 for a term present
in every document. Unknown query terms are ignored; all-zero queries should
return no lexical results. Rebuild normalize/indexer/tfidf outputs together
after corpus or preprocessing changes; artifact compatibility is not automatic.

Query-topic estimation uses processed RAW counts aggregated by manual labels,
not TF-IDF centroids. It is now implemented in topic_model.py: add-one smoothing,
uniform topic priors and log-space query likelihoods. Details follow below.

## 6. Current versus pending deliverables

Available now: collector/reviewer code, tokenization/token saving, inverted
indexing with raw per-document counts and DF, vocabulary/row mappings,
DF/IDF arrays and the normalized sparse TF-IDF text matrix,
Document/DocumentCollection definitions and offline tests. A real corpus is available only after an actual crawl and
human review. Generated data must be shared separately; Git ignores it.

search.py now supplies load_data(), make_query_vector() and search(). Its loader
returns approved metadata, postings, raw term counts, vocabulary, ordered IDs,
row mapping, IDF and the sparse matrix in one dictionary. Load it once and pass
it to repeated search calls:

~~~python
from project.search import load_data, search

data = load_data()
results = search("computer software", data, top_k=10)
~~~

Results are dictionaries with doc_id, score (cosine), title, preview, url and
topic. Unknown/zero-vector queries return an empty list. The loader rejects
an empty approved corpus, changed approval selection or incompatible array
dimensions. It does not fingerprint text revisions; rebuild after text changes.
The default CLI mode is the TF-IDF baseline; --mode tspr loads topic/rank data.

Topic estimation, PageRank/TSPR and query-weighted rank aggregation are now
implemented. No real retrieval quality scores are claimed from tests.

## 7. Complete offline build and ranking contract

Run python project/preprocess.py after review. It reuses normalization, indexing
and TF-IDF functions, then builds the unigram topic model and the rank vectors.
All five manual topics must be represented. No auto-labeling is done.
Source JSON stays unchanged. Run this command again after text/link/label changes.

The same Index/ folder also contains:

- topic_log_probs.npy, shape (K, V), natural-log smoothed P(term|topic).
- topic_model.json with topics, alpha=1.0, uniform priors, word_totals and doc_counts.
- topic_pagerank.npy, shape (K, N), each row sums to one.
- global_pagerank.npy, shape (N,), uniform-teleport PageRank baseline.
- ranking.json with doc_ids, topics, labels, settings, link/dangling counts
  and convergence details.

Rows follow doc_ids.json; vocabulary columns follow term_to_col.json. Topic rows
follow TOPICS. Topic model probabilities are (raw count + 1)/(topic word total + V).
query_topic_weights() uses the shared tokenize() and returns normalized Bayes
weights across all five topics; repeated words count repeatedly, unknowns are
ignored, and no known words returns the priors. This is supervised word-based
classification, not latent topics or embeddings. Smoothing is our adaptation,
not a claim of exactly reproducing Haveliwala's maximum-likelihood estimates.

tspr.py internally resolves canonical title matches to these text rows and
builds a binary CSR link matrix. No graph.py or standalone graph export is
reintroduced; the crawler still only saves raw titles. Ignore external targets,
self-links and unresolved aliases; collapse repeated edges. Every internal
saved edge is retained, not just BFS discovery edges. Dangling pages are those
with no outgoing link WITHIN this approved sample.

PageRank uses damping 0.85, max absolute change <1e-8, cap 200 iterations.
Each topic's teleport and dangling redistribution are uniform over documents
with that manual label. Links are not restricted to the topic. Global PageRank
uses a uniform vector across all approved documents. The builder refuses
non-converged results rather than calling them finished.

~~~python
from project.search import load_data, search
data = load_data(mode="tspr")
results = search("computer chip", data, top_k=10, mode="tspr")
~~~

Modes: tfidf (default), pagerank, tspr. Shared postings produce lexical candidates;
zero-cosine matches are excluded. TSPR sorts candidates by the dot product of
query-topic weights and their topic rank scores. PageRank sorts by global
rank; TF-IDF sorts by cosine. Results also include cosine alongside score.
No new PageRank run occurs online. A fixed TF-IDF/link blending weight has NOT
been chosen or introduced. Vector mixing is used as a ranking score; because
dangling redistribution depends on the topic, it is not claimed to equal a
single fresh run with mixed teleportation.

HITS is not required. Relevance judging, Precision/Recall/F1, measurement and
the report remain. The spam-farm experiment is deferred by user request; it
still remains an explicit course requirement for before/after evaluation.
