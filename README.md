# Wikipedia Search Engine - Group 17

Information Retrieval: Category 1, Classic Paper Replication & Stress Testing.
Topic-Sensitive PageRank (TSPR) is implemented with NumPy/SciPy sparse matrix
operations. No NetworkX or Scikit-Learn is used. NLTK supplies only Porter stemming.

Members: Vignesh K S (2024A7PS0571P) and Thanishka Thatikonda (2024ADPS0041P).

## Setup and search

Use Python 3.10 or newer. Run commands from the repository root:

~~~powershell
git clone https://github.com/gurtcoder123/search_engine_ir.git
cd search_engine_ir
python -m pip install -r project/requirements.txt
python -X utf8 -m project.search
~~~

The reviewed corpus and precomputed artifacts are included in
project/Pre-Processing/. After installing dependencies, you can search directly
without crawling or rebuilding. Run preprocess to reproduce the offline build.
Use python -X utf8 -m project.preprocess from the repository root for that rebuild.
The corpus contains saved Wikipedia extracts, source URLs, revision IDs and
outgoing link titles. No API calls occur during search or evaluation.

Search now defaults to **hybrid**: 90% TF-IDF cosine and 10% scaled TSPR.
Type q, quit or exit to stop. To choose another mode:

~~~powershell
python -X utf8 -m project.search --mode hybrid --lexical-weight 0.9
python -X utf8 -m project.search --mode tspr
python -X utf8 -m project.search --mode pagerank
python -X utf8 -m project.search --mode tfidf
~~~

Pure TSPR is the replication mode. Hybrid is a separate retrieval extension;
its weight is a fixed design choice, not a validated optimal value.

## Components

| File | Responsibility |
| --- | --- |
| project/collection/crawler.py | MediaWiki API, bounded BFS, article records and raw outgoing titles |
| project/collection/reviewer.py | Approval, manual topic labels and approved-contents JSON |
| project/indexing/normalize.py | Shared document/query tokenization and stemming |
| project/indexing/indexer.py | Inverted postings, term counts and document frequencies |
| project/indexing/tfidf.py | Log-TF/IDF weighting and L2-normalized CSR matrix |
| project/ranking/topic_model.py | Smoothed unigram query-topic probabilities |
| project/ranking/tspr.py | Internal link matrix, global PR and five TSPR vectors |
| project/preprocess.py | Build all offline artifacts in one command |
| project/search.py | In-memory candidate retrieval and four ranking modes |
| project/evaluate.py | Precision@K, Recall@K, F1@K, latency and traced memory |
| project/experiment.py | Clean/control/attacked copies and before/after analysis |
| project/evaluation_judgments.json | Frozen development queries and provisional relevance labels |

## Pipeline and scoring

Offline: crawl -> review -> tokens -> inverted index + TF-IDF -> topic model +
link graph -> saved Index/ artifacts. Five topics are technology (including
business), science (including mathematics), health and medicine, sports, politics.

Query time: same tokenizer -> union of postings -> cosine > 0 candidates ->
selected ranking -> titles, saved previews and source links.

- TF(count) = 1 + log10(count) for positive counts; zero otherwise.
- IDF(term) = log10(N / DF(term)); shared corpus IDF for documents and queries.
- Document and query vectors are L2-normalized; their dot product is cosine.
- P(term|topic) = (count(term,topic)+1)/(total_words(topic)+V).
- Query-topic weights use log likelihoods and uniform priors.
- importance = query_topic_weights @ topic_pagerank.
- Hybrid: 0.9*cosine + 0.1*(importance/max(importance)). The maximum is over
  the entire indexed corpus for that query, not only matching documents.

TSPR follows ordinary links, including links outside the chosen topic. Only
teleportation is topic restricted. Dangling mass uses the same teleport vector.
Power iteration uses damping 0.85, maximum absolute change below 1e-8, and a
200-iteration cap. Pure TSPR/global modes order positive-cosine candidates by
their link scores; hybrid retains lexical strength in the final ordering.

## Evaluation and attack

~~~powershell
python -X utf8 -m project.evaluate --judgments project/evaluation_judgments.json
python -X utf8 -m project.experiment --judgments project/evaluation_judgments.json --target-id doc-31524
~~~

The supplied attack target is Triathlon, which is nonrelevant to the World Cup
query intent but has a weak lexical match. Its selection is explicit; the
experiment does not search for the target causing the worst measured damage.

The experiment rebuilds three separate copies under output/attack/:

1. clean/: original approved articles.
2. control/: the same articles plus ten same-topic synthetic pages, with text
   duplicated from the target and no outgoing links.
3. attacked/: identical to control, except each synthetic page links to the target.

The original collection and indexes stay untouched. Original incoming/outgoing
links are preserved; no live page is edited. The attack assumes the synthetic
pages have entered the indexed corpus and obtained the target's topic label.
Synthetic duplicates are nonrelevant; existing relevant sets remain fixed.

Clean-to-attacked measures the whole injection, including corpus IDF and topic
model changes. Control-to-attacked isolates the added links: text features and
topic probabilities are identical between those two conditions. summary.json
records metrics, changes, target scores/positions, corpus/judgment hashes and
a descriptive lexical-weight sweep. The sweep is development-set analysis,
not held-out tuning. Re-running writes only the experiment outputs. The saved
reference run is versioned in results/attack_summary.json.

Precision uses hits/K even for short lists; recall uses the fixed relevant set.
Reported F1 is the mean of per-query F1. Binary NDCG also measures result order.
Latency excludes loading/printing;
tracemalloc measures traced allocation peak, not total process RSS.

The relevance labels are assistant-generated, corpus-informed development
judgments. They are **not independently human-validated**. Results demonstrate
this local experiment and do not establish broad search quality or spam resistance.

## Collect a new corpus

For a fresh collection, create project/collection/seeds.json:

~~~json
{
  "technology": ["Computer"],
  "science": ["Neutrino"],
  "health_and_medicine": ["Vaccine"],
  "sports": ["FIFA World Cup"],
  "politics": ["Democracy"]
}
~~~

~~~powershell
python -X utf8 -m project.collection.crawler
python -X utf8 -m project.collection.reviewer
python -X utf8 -m project.preprocess
~~~

The crawler refuses to overwrite an existing collection. Use a separate data
directory through the crawler class for a new run; do not delete the demo corpus.
Defaults: depth 3, at most 40 new pages per seed group, five followed links per
page, saved first three paragraph blocks. All API-returned outgoing titles are
retained, including links not followed. Seed groups do not assign final labels.
Review updates approval/topic in the authoritative collection and article JSONs;
approved files move into topic folders. Rejected documents are retained.
Approved_Contents.json lists approved titles and document IDs by topic.

## Tests and further documentation

~~~powershell
python -m pip install pytest
python -X utf8 -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/final_tests
~~~

Tests use local fixtures rather than live crawling. See docs/IMPLEMENTATION.md
for algorithm details, docs/PARTNER_HANDOFF.md for data contracts,
docs/RESULTS.md for measured results and docs/DEMO.md for the presentation.

## Literature

- T. H. Haveliwala, [Topic-Sensitive PageRank (WWW 2002)](https://snap.stanford.edu/class/cs224w-readings/Haveliwala02Topicsenitive.pdf).
  We replicate topic-biased offline vectors and query-weighted importance.
  Five manual Wikipedia labels, add-one smoothing, the explicit dangling policy
  and lexical fusion are documented project adaptations.
- Manning, Raghavan and Schutze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/):
  Chapters 6, 8 and 21 for vector scoring, evaluation and link analysis.

Known limitations: sampled graph, unresolved redirect aliases, short extracts,
coarse topics, provisional judgments and no validated anti-spam defense.
HITS, dense retrieval, a graphical interface and TrustRank are outside this implementation.
