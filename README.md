# Wikipedia Search Engine — Group 17

Information Retrieval course project using Topic-Sensitive PageRank.
TF-IDF retrieval, unigram query-topic matching and TSPR are implemented.
Clean-data preliminary evaluation is implemented; adversarial testing remains.

## Project layout

~~~text
project/
  collection/       # Wikipedia crawler, reviewer and seed example
  indexing/         # Tokenization, stop words, inverted index and TF-IDF
  ranking/          # Unigram topic model and sparse PageRank/TSPR
  tests/            # Offline unit and integration tests
  config.py         # Shared topics and local data paths
  preprocess.py     # Build all offline search artifacts
  search.py         # Interactive search
  evaluate.py       # Evaluate query-specific relevance judgments
  requirements.txt
docs/
  IMPLEMENTATION.md
  PARTNER_HANDOFF.md
~~~

## Quick start

Run these commands from the repository root:

~~~powershell
python -m pip install -r project/requirements.txt
Copy-Item project/collection/seeds.example.json project/collection/seeds.json
python -m project.collection.crawler
python -m project.collection.reviewer
python -m project.preprocess
python -m project.search --mode tspr
~~~

Edit the local seeds.json before crawling. Do not copy the example over an
existing seed file unless you intend to replace it. seeds.json is ignored;
the example is versioned. Review/approve articles before preprocessing.
The existing direct entry points also work, such as python project/search.py.
After this reorganization, the crawler/reviewer are inside collection/ rather
than directly inside project/.

For redirected output on Windows, use UTF-8 mode to display article text with
Greek or other non-ASCII characters: python -X utf8 -m project.search --mode tspr.

## Current implementation

- project/collection/crawler.py: Document, a shared DocumentCollection, and WebCrawler.
  The crawler has request(), crawl(), reset() and save().
- project/collection/reviewer.py: inspect saved previews and known incoming/outgoing links;
  manually approve/reject articles and assign topics.
- project/config.py: topic names and the local storage location.
- project/indexing/normalize.py: simple shared tokenization and separate token JSON files
  for approved articles, using NLTK's PorterStemmer for English words.
- project/indexing/indexer.py: inverted postings, raw term counts per document and
  document frequencies, saved as plain JSON.
- project/indexing/tfidf.py: log-TF/IDF weighting and an L2-normalized SciPy CSR
  document-term matrix, with saved vocabulary and document order.
- project/ranking/topic_model.py: aggregate raw word counts by manual topic and estimate
  query-topic probabilities with a smoothed unigram model.
- project/ranking/tspr.py: sparse global PageRank and five topic-biased rank vectors.
- project/preprocess.py: run all offline steps after review; save into Index/.
- project/search.py: load artifacts once, union postings and rank results with
  TF-IDF, global PageRank or query-weighted TSPR.
- project/evaluate.py: Precision@K, Recall@K, F1@K, query latency and peak traced
  allocations from explicit query-specific relevance judgments.
- project/tests/: offline collection, persistence and reviewer tests.

The crawler/reviewer do not construct a graph. tspr.py builds its own internal
sparse link matrix from approved articles' saved outgoing titles; graph.py and
the earlier graph exports remain removed. No NetworkX or Scikit-Learn is used.

## Workflow

1. Supply Wikipedia seed titles for the five topic groups.
2. Crawl up to 40 new pages per seed group with bounded BFS.
3. Save each article as JSON, plus Document_Collection.json.
4. Manually review articles; save approval/topic in both copies.
5. Normalize and index only records with approved exactly true.
6. Run python -m project.preprocess to build the index, TF-IDF, topic model and ranks.
7. Run python -m project.search --mode tspr for repeated queries.
   Use --mode tfidf or --mode pagerank for baselines.

All five topics need at least one reviewed, approved article. The course corpus
requires at least 100 approved articles, not merely 100 collected candidates.

By default BFS follows at most five new links per page to depth three (seed
depth is zero). It saves **all** outgoing article titles returned by the API,
including visited and unfollowed targets. Already-collected pages are not
expanded again. Forty is a ceiling, not a guaranteed final topic count.

There is no discovered_topic field. Seed groups control collection quotas;
they do not label documents. A new document's topic and approved are null.

## Saved data and review

Default local root: project/Pre-Processing/ (Git-ignored).

- Document_Collection.json: the authoritative array of ALL article records.
- Documents/doc-123.json: pending or rejected article.
- Documents/technology/doc-456.json: manually approved/labelled article.

approved=null means unreviewed, true means accepted, false means rejected.
Rejected records and files stay on disk. Review updates the in-memory document,
collection entry and article JSON after each choice. Approved files move into
the selected topic folder; changing a decision moves them as needed.
file_path is relative to the collection root.

The default saved text and preview are the first three nonempty paragraphs.
max_paragraphs and max_chars are configurable. Links are collected independently
of excerpt length. Incoming links shown by the reviewer are local title matches,
not Wikipedia-wide backlinks; unresolved redirect aliases may be missed.

See [the implementation guide](docs/IMPLEMENTATION.md) for commands and
[docs/PARTNER_HANDOFF.md](docs/PARTNER_HANDOFF.md) for the revised interfaces.

## Offline tests

Install project/requirements.txt and pytest, then run from the repository root:

~~~powershell
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/simple_workflow_tests
~~~

Tests use small offline fixtures; they are not evidence of real retrieval
quality. Importing the modules does not crawl or delete saved data.

The preliminary local evaluation uses assistant-generated relevance judgments,
not independently human-validated labels or a held-out benchmark. Generated
data and results are not included in Git. Supply a reviewed collection and
query-specific relevance judgments to reproduce an evaluation.

The spam-farm experiment remains planned work for the required before/after
adversarial evaluation; clean-data metrics alone do not complete that study.
