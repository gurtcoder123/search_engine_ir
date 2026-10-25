# Implementation and running guide

Run commands from the repository root. Python 3.10+ is required.

## 1. Provide seeds and crawl

~~~powershell
python -m pip install -r project/requirements.txt
Copy-Item project/collection/seeds.example.json project/collection/seeds.json
~~~

Fill project/collection/seeds.json with plain Wikipedia article titles, not URLs.
For the actual corpus provide seeds for all five groups: technology, science,
health_and_medicine, sports and politics. Multiple seeds per group can improve
coverage.

~~~powershell
python project/collection/crawler.py
~~~

The short entry block at the bottom of crawler.py loads that file and uses
constructor defaults. To configure the crawler directly:

~~~python
from project.collection.crawler import DocumentCollection, WebCrawler

collection = DocumentCollection()
crawler = WebCrawler(
    document_collection=collection,
    topics=["technology", "sports"],
    seed_topics=[["Computer", "Software"], ["Association football"]],
    depth=3,
    max_pages_per_topic=40,
    max_links_per_page=5,
    max_paragraphs=3,
    max_chars=None,
    min_words=100,
    request_delay=1.0,
)
documents = crawler.crawl()
crawler.save()  # Or save("tmp/my_crawl") for a separate collection
~~~

The same collection instance is populated in place. Real defaults preserve
intentional zeros: depth=0 fetches only seeds; request_delay=0 stays zero.
Keep a nonzero delay for live collection.

### Four crawler methods

- request(title): fetch article metadata, plain text and ALL link batches.
  Follow API continuation tokens; skip missing/disambiguation pages and
  ordinary request failures.
- crawl(): BFS per seed group and return the document list. Do not revisit or
  re-expand stored pages. Exclude visited/queued titles before selecting the
  limited number of new links to follow. Retain all raw outgoing article titles.
- save(output_dir=...): write Document_Collection.json plus one article JSON.
  Refuse an existing collection so a new crawl cannot overwrite review decisions.
- reset(): clear the in-memory collection/counts only. Saved files are untouched.
  Use a fresh output directory for a new saved crawl.

Depth defaults to 3, the discovery limit to 5, and the cap to 40 NEW pages per
seed group. Forty is a ceiling, not a guarantee; exhausted frontiers are logged.
Sampling has a fixed random seed. max_links_per_page=None follows all unvisited
links. Several seed groups may reach the same page; it is not expanded twice.

All outgoing article titles stay saved, including cross-links, backlinks,
self-links and uncollected targets. No graph, adjacency export, link-ID
resolution or matrix is constructed here.

### Text selection

- max_paragraphs=3: save the first three nonempty paragraphs in text AND preview.
- max_paragraphs=None: no paragraph limit.
- max_chars=2000: additionally truncate to 2,000 characters.
- max_chars=None: no character limit.
- min_words=100: skip pages whose SELECTED text has fewer than 100 words.

When both limits are set, apply paragraphs first, then characters. Small
excerpts can fall below min_words; adjust the threshold for small tests.
The saved excerpt, not an unsaved full article, is the planned indexing input.
Link collection does not depend on the text limit.

### Simple request handling

Requests are sequential and each HTTP call gets the configured delay, including
pagination and pages that fail. A network failure, HTTP error (including 429),
invalid JSON or API error skips that article. There are no retries, backoff or
crawl-wide cooldown handling. A failed pagination batch skips the article rather
than saving an incomplete link list.

This is deliberately minimal, not comprehensive rate-limit protection. Keep a
nonzero delay for live crawling. The User-Agent identifies our project repository;
callers can supply their own identifier/contact.

## 2. Review, shortlist and label

~~~powershell
python project/collection/reviewer.py
# Or: python -m project.collection.reviewer
# Another collection:
python project/collection/reviewer.py --data-dir tmp/my_crawl
~~~

The reviewer loads Document_Collection.json and shows title, URL, saved preview,
current decision, incoming title matches within the collection, and the first
20 outgoing titles.

- 1–5: approve and label with that topic.
- r: reject; approved=false and topic=null.
- s: skip or leave the current decision unchanged.
- v: view saved text.
- l: view the complete outgoing link list.
- q: quit. All completed decisions are already saved.

approved=null is unreviewed. approved=true is the ONLY final-corpus inclusion
rule. Rejected/unreviewed entries remain in the array. Do not infer approval
from a file's existence or topic folder alone.

Each choice updates the in-memory Document, article file and collection array.
Approved files move to Documents/<topic>/. Changing a topic moves the file
again. Rejecting a previously approved article moves it back to Documents/.
No article is deleted or copied into multiple topic folders. Existing unrelated
destination files are not overwritten. Topic folders may already exist; missing
ones are created as needed.

~~~powershell
python project/collection/reviewer.py --review-again
~~~

Normal runs skip completed decisions; --review-again allows changing them.
The summary displays approved/rejected/unreviewed counts and counts per topic.
The intended course corpus needs at least 100 approved documents.

The reviewer is offline. Incoming links are title matches from saved articles,
not a Wikipedia-wide backlink lookup. Uncollected sources and unresolved
redirect aliases are unknown.

## Files and scope

~~~text
project/Pre-Processing/
  Document_Collection.json
  Documents/
    doc-123.json                    # pending/rejected
    technology/doc-456.json          # approved
    science/
    health_and_medicine/
    sports/
    politics/
~~~

Every record has page_id, title, url, doc_id, collected_at, preview, text,
revision_id, outgoing_links, topic, approved and file_path.
file_path is relative to Pre-Processing/. doc_id is an opaque file/record ID,
not a row label for a matrix.

This replaces the earlier candidate/reviewed JSONL interface. Existing old data
is not deleted or automatically migrated. There is no graph.py, graph loader,
adjacency export, fingerprint machinery or discovered_topic field.

## 3. Tokenize approved articles

~~~powershell
python project/indexing/normalize.py
~~~

This reads the collection array, processes only approved=true records and
writes separate token files under Pre-Processing/Tokens/doc-123.json.
Each contains doc_id and its ordered tokens list, including repetitions.
Article text, previews, labels, paths and the collection JSON are not changed.
This tokenization step does not build the index or TF-IDF matrix.

Queries must reuse the same function:

~~~python
from project.indexing.normalize import tokenize
tokens = tokenize("The U.S.A. uses COVID-19 tests")
~~~

Rules: Unicode NFKC normalization, lowercase the whole text, remove possessive
's, remove stop words using the local stopwords.txt, and preserve numbers and
short/rare terms. Hyphenated terms stay together (covid-19). Other punctuation
and underscores separate tokens. Dotted initials lose dots (U.S.A. -> usa),
but abbreviations are not expanded. Decimal points are ordinary punctuation.
The stop-word list is a small project-defined list, not an external corpus.

Plain English alphabetic words use NLTK's PorterStemmer (NLTK_EXTENSIONS mode):
computer/computers/computing become comput. Stop words are removed BEFORE
stemming so words such as was/this do not leak through as stems. Numbers,
mixed letter/number tokens, hyphenated compounds and non-ASCII words stay
unchanged after lowercase/Unicode normalization. The same rules apply to queries.

Install the updated requirements before running this module. No NLTK tokenizer,
stop-word corpus or nltk.download() call is needed; only PorterStemmer is used.
Regenerate any token files produced before stemming was added:

~~~powershell
python -m pip install -r project/requirements.txt
python project/indexing/normalize.py
~~~

After changing review decisions, consume tokens only for currently approved
record IDs; a leftover token file is not evidence of approval.

## 4. Build the inverted index

After review and tokenization:

~~~powershell
python project/indexing/indexer.py
~~~

This reads only token files for records currently marked approved=true in
Document_Collection.json. Old/rejected token files are ignored. If an approved
article has no token file, run normalize.py first; it is not silently omitted.

Outputs under Pre-Processing/Index/:

- inverted_index.json: term -> list of [doc_id, raw occurrence count].
- df.json: term -> number of documents containing that term.
- term_counts.json: doc_id -> {term: raw occurrence count}.
- doc_ids.json: approved document IDs, including documents with empty token lists.

Each document appears once per term posting. A term occurring 100 times in one
document contributes TF count=100 but only 1 to DF. Counts are raw integers,
not log-scaled TF or TF-IDF. Original documents and saved tokens are untouched.
doc_ids is the indexed document order, not a numeric row assignment or graph.
Rerun after changed approval decisions or regenerated tokens.

The module has three small functions: load_tokens(), build_inverted_index()
and save_index(). No positional postings, vocabulary columns, IDF, numerical
matrix or ranking is added yet.

## 5. Build the TF-IDF matrix

After indexer.py:

~~~powershell
python project/indexing/tfidf.py
~~~

It reads term_counts.json, df.json and doc_ids.json from Pre-Processing/Index/.
It writes an L2-normalized SciPy CSR document-term matrix to tfidf_matrix.npz,
plus term_to_col.json, doc_to_row.json, df.npy and idf.npy. doc_ids.json is
preserved in the same order. This is a text matrix, not a link/graph matrix.

Rows follow doc_ids; columns follow the sorted vocabulary in term_to_col.
The numerical DF/IDF arrays use that column order. df.json remains the readable
term-to-DF mapping; df.npy stores the same counts in vocabulary-column order.

Weights use TF=1+log10(count) and IDF=log10(N/DF), followed by L2 normalization
per document. N includes approved documents with no tokens. Empty rows stay
zero; terms appearing in every document have IDF=0, as expected. The matrix
contains no stored zero weights and is never converted to dense form in the
implementation. A one-document corpus will therefore have an all-zero matrix.

To load the numerical artifacts:

~~~python
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

index_dir = Path("project/Pre-Processing/Index")
tfidf_matrix = load_npz(index_dir / "tfidf_matrix.npz")
idf = np.load(index_dir / "idf.npy", allow_pickle=False)
term_to_col = json.loads((index_dir / "term_to_col.json").read_text(encoding="utf-8"))
doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
~~~

After changing text, tokenization or review selection, rerun normalize.py,
indexer.py and tfidf.py in that order. Query vectors must use the same saved
IDF/vocabulary and be L2-normalized; document rows are already normalized.
Basic query retrieval and topic-sensitive ranking are implemented below.

## 6. Search the TF-IDF baseline

After review and the normalize/indexer/tfidf steps:

~~~powershell
python project/search.py
# Optional:
python project/search.py --top-k 5
python project/search.py --data-dir tmp/my_corpus
~~~

The index and approved metadata load once at startup. Type repeated queries;
q/quit/exit ends the session. Results show title, cosine score, saved preview
and URL, without crawling or making API calls.

The same tokenize() function handles documents and queries. The query uses
saved corpus IDF, log-scaled TF and L2 normalization. Candidates are the UNION
of term postings, not their intersection. Sparse dot products against the
already-normalized document rows give cosine scores. Only positive-score
matches are returned, up to top_k; ties keep indexed document order.

Unknown words are ignored. Empty, stopword-only and all-zero-IDF queries return
no lexical results. A corpus with no approved articles cannot start search.
If the current approval set or artifact dimensions disagree with the saved
index, startup asks for a rebuild. Same-ID text changes are not detected
automatically: regenerate the offline artifacts after editing source text.

The four functions in search.py are load_data(), make_query_vector(), search()
and the small interactive main(). The default is still the TF-IDF baseline.

## 7. Build the topic model and TSPR (all offline steps in one command)

After manual review, run:

~~~powershell
python project/preprocess.py
# Or use another collection root:
python project/preprocess.py --data-dir tmp/my_corpus
~~~

This replaces running normalize/indexer/tfidf separately when building the full
system. It calls their existing functions, then builds the topic model and
rank vectors. It never crawls, approves or labels articles. It rebuilds derived
artifacts from the CURRENT approved selection; original article JSON is unchanged.
Each of the five topics must have at least one approved article. The course
corpus target is at least 100 approved articles; tiny test fixtures are not that
dataset. Rebuild after editing text, links, labels or preprocessing settings.

### Word-based topic matching: topic_model.py

For each topic, add up RAW processed word counts from its approved articles.
The K-by-V array stores log probabilities, with the existing vocabulary columns:

~~~text
P(word | topic) = (count(word, topic) + 1) / (total_words(topic) + V)
P(topic) = 1 / 5
log_score(topic, query) = log P(topic) + sum over query tokens of log P(token | topic)
P(topic | query) = exp(log_score - max_log_score), divided by the sum across topics
~~~

Repeated query words contribute repeatedly; unknown words are ignored. No known
words means uniform priors, but search still returns no results for a zero
TF-IDF query. Query and document words share tokenize(). This is a unigram
classifier, NOT TF-IDF centroids, semantic embeddings or unsupervised topic modelling.
Add-one smoothing is our project adaptation; Haveliwala uses maximum-likelihood
unigram estimates and uniform priors in his query-topic calculation.

### Link ranking: tspr.py

Match saved outgoing titles to approved canonical article titles, using the
crawler's underscore/whitespace/first-letter normalization. Keep all matching
internal links, including backlinks/cross-links, not just the BFS discovery
edges. Repeated links become one binary edge; self-links and outside/unresolved
targets are ignored. Unknown redirect aliases are not resolved. This is the
induced graph of our sampled approved corpus, not the full Wikipedia graph.
The internal CSR matrix is built here, not by the crawler, and is not exported.

Row=source and column=destination; both use doc_ids.json order. Divide each
source's rank by its internal outgoing degree and multiply by the transpose:

~~~text
r_new = 0.85 * (A.T @ (r / out_degree) + dangling_mass * v) + 0.15 * v
~~~

Only non-dangling sources take part in r/out_degree. v is uniform over the
approved articles labelled with the chosen topic, zero elsewhere. Links may
lead OUT of the topic; only teleportation is topic restricted. Dangling mass
uses the same v. Each topic run starts at v. Stop when the maximum absolute
score change is below 1e-8, with a 200-iteration cap. The full build refuses to
save new index/rank outputs on non-convergence. Global PageRank uses uniform v
over all approved articles. A topic rank may legitimately be zero for an
unreachable page. No dense N-by-N transition matrix is constructed.

Additional files in the SAME Pre-Processing/Index/ directory:

- topic_log_probs.npy: K-by-V log P(word|topic).
- topic_model.json: topic order, uniform priors, smoothing and raw count totals.
- topic_pagerank.npy: K-by-N rank vectors.
- global_pagerank.npy: N global baseline scores.
- ranking.json: topic/document order, labels, damping/tolerance, link/dangling
  counts and convergence information for all runs.

Article metadata still comes from Document_Collection.json. No single text
file needs to hold numerical matrices. Tokens are separate intermediate files.

## 8. Search using TSPR or the baselines

~~~powershell
python project/search.py --mode tspr
python project/search.py --mode tfidf
python project/search.py --mode pagerank
~~~

All modes use the same UNION of postings and retain only positive TF-IDF cosine
matches. TF-IDF mode sorts by cosine. PageRank mode sorts these candidates by
their global rank. TSPR estimates all five query-topic probabilities and sorts
the candidates by sum_topic P(topic|query) * topic_rank(topic, document).
The CLI shows the topic probabilities. Rank ties retain indexed document order.
An article with zero link score can still be a valid lexical result.

This implements query-weighted rank aggregation, not a chosen 70/30-style
TF-IDF/link blend; cosine filters lexical matches but is not added to the link
score. Additional fusion is deferred. With topic-dependent dangling handling,
we do not claim that a mixture of saved vectors is exactly the solution of one
new PageRank run with a mixed teleport vector; we use it as the query importance
score. No iteration or topic training happens at query time.

~~~python
from project.search import load_data, search
data = load_data(mode="tspr")
results = search("computer software", data, mode="tspr", top_k=10)
~~~

Load once and reuse in memory. Results contain doc_id, score (the selected
ranking score), cosine, title, preview, url and topic. Index order/dimension or
label changes detected at load time require a rebuild; same-ID text/link edits
are not automatically detected. Use preprocess.py after any source edit.

### Literature and remaining work

- [Haveliwala, Topic-Sensitive PageRank (WWW 2002)](https://snap.stanford.edu/class/cs224w-readings/Haveliwala02Topicsenitive.pdf),
  Sections 3.2-3.3: topic teleport vectors, unigram query-topic classification
  and probability-weighted rank aggregation. We use five manually labelled
  Wikipedia topics instead of 16 ODP categories and add smoothing.
- [Manning et al., Topic-specific PageRank](https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html).

Evaluation needs manually judged query-document relevance labels, not just
topic labels. Precision/Recall/F1, timing/memory measurements and the report
can be computed with evaluate.py. Tests below verify code, not real-corpus relevance quality.
The spam-farm experiment is deferred by request but remains explicitly required
by the course; a clean-only evaluation is not the full Category 1 experiment.

## 9. Preliminary evaluation

The display scopes are now technology AND business, and science AND mathematics;
stored topic keys remain technology/science so existing interfaces do not change.
The reviewed snapshot has 162 approved articles from 200 candidates. The first
58 review decisions were made by the user and preserved; the remaining 142 were
assistant-reviewed at the user's request. output/review/ holds an article-level
audit and a backup of the collection before the assisted review. Source records
and per-article JSON were updated through the existing reviewer workflow.

~~~powershell
python project/evaluate.py --judgments output/evaluation/judgments.json --repeats 30
~~~

The supplied judgments file contains provenance and a queries array. Each query
has an id, query string, intent and relevant list of approved doc IDs. Relevance
is query-specific; topic membership alone is NOT treated as relevance. All
unlisted approved documents are treated as nonrelevant for that query.
The current ten-query, corpus-informed development set uses provisional
assistant-generated content judgments, not independent human-validated qrels.
Do not describe these results as a held-out benchmark or a human user study.

evaluate.py writes output/evaluation/results.json with per-query rankings,
macro Precision@10/Recall@10/F1@10, latency and peak traced allocations. P@10
always divides by ten even for short result lists; Recall@10 divides by that
query's relevant-set size. Macro F1 is the mean of per-query F1 values, not
the harmonic mean of macro precision and recall. Query timing excludes loading,
printing and interactive input. Memory uses tracemalloc peak during loading
plus a query pass; it is NOT process RSS. Run timing without concurrent tests.
Generated data/results remain local and Git-ignored. The evaluation module and
metric arithmetic tests are ordinary implementation code.

## Offline tests

~~~powershell
python -m pip install pytest
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/simple_workflow_tests
~~~

Tests cover BFS/non-revisits, link retention and pagination, text limits,
skip-on-error/no-retry behaviour, save/load, review resume, topic-folder moves and the
mocked crawl -> save -> review -> reload workflow.
