# Shared implementation contract - Group 17

Updated for the completed clean/control/attack workflow. This supersedes the
earlier planned A/B/C/D division; the modules below are integrated.

## Main entry points

Run from the repository root:

~~~powershell
python -X utf8 -m project.preprocess
python -X utf8 -m project.search --mode hybrid
python -X utf8 -m project.evaluate --judgments project/evaluation_judgments.json
python -X utf8 -m project.experiment --judgments project/evaluation_judgments.json --target-id doc-31524
~~~

For new collections: project.collection.crawler, then project.collection.reviewer.
The reviewed dataset, token files and offline artifacts are versioned under
project/Pre-Processing/. Clone the repository to get the exact saved corpus.
Generated experiment directories, temporary files and PDFs remain Git-ignored.

## Document contract

Document is defined in project/collection/crawler.py. Its fields are page_id,
title, url, doc_id, collected_at, preview, topic, text, revision_id,
outgoing_links, approved, file_path.

- page_id is Wikipedia's integer ID; doc_id is an opaque string such as doc-1164.
- outgoing_links contains article TITLES, never doc IDs or BFS parent pointers.
- approved is null/true/false. Only exactly true enters preprocessing.
- topic is one of the five project.config.TOPICS keys after approval.
- file_path is relative to the collection root.
- Document_Collection.json is the authoritative array; article files mirror it.
- Reviewer saves both after each decision and exports Approved_Contents.json.

The shared DocumentCollection holds Document objects and deduplicates page IDs.
Seed group controls crawl quota, not final labels. Reviewer owns final labels.
No standalone graph.py is needed.

## Offline artifacts

project/preprocess.py rebuilds all derived files under Pre-Processing/Index/.
Rerun it after text, link, approval or label changes. It does not recrawl.

| Artifact | Structure |
| --- | --- |
| inverted_index.json | term -> list of [doc_id, raw count] |
| term_counts.json | doc_id -> term -> raw count |
| df.json | term -> document frequency |
| doc_ids.json | ordered list of opaque document ID strings |
| doc_to_row.json | doc_id -> zero-based matrix row |
| term_to_col.json | normalized term -> zero-based vocabulary column |
| tfidf_matrix.npz | SciPy CSR, N documents by V vocabulary terms |
| df.npy / idf.npy | length V, aligned with vocabulary columns |
| topic_log_probs.npy | K by V, natural log P(term given topic) |
| topic_model.json | topic order, priors, smoothing, word/doc totals |
| topic_pagerank.npy | K by N, one converged topic rank per row |
| global_pagerank.npy | length N, global baseline |
| ranking.json | document/topic order, labels, settings, graph and convergence info |

TOPICS order: technology, science, health_and_medicine, sports, politics.
Technology includes business; science includes mathematics.

## Text and topic scoring

Use project.indexing.normalize.tokenize for both documents and queries.
Rules: Unicode NFKC, lowercase, simple apostrophe/initial handling, local stop
words, NLTK PorterStemmer for plain ASCII English words. No NLTK corpus download.
Numbers and hyphenated compounds are retained. No spelling prediction.

TF(c) = 1+log10(c) for c>0, otherwise 0. IDF(t) = log10(N/DF(t)).
Use the saved corpus IDF for queries too. L2-normalize nonzero document and
query vectors; zero vectors stay zero. L2 is Euclidean length, not softmax.

Topic model aggregates raw token counts by manual topic:
P(term|topic) = (count+1)/(topic_word_total+V), with uniform priors.
Query log-score = log prior + sum log probabilities over known repeated tokens.
Exponentiate after subtracting the maximum and normalize across topics.
Unknown words are ignored. This is a unigram classifier, not TF-IDF centroids.

## Graph and PageRank

tspr.py builds internal binary source-row/destination-column links from saved
outgoing titles. Keep cross-links/backlinks; deduplicate and omit self-links,
outside targets and unresolved aliases. All numerical rows follow doc_ids.

r_new = 0.85*(A.T @ share + dangling_mass*v) + 0.15*v
share = r/outdegree for non-dangling pages; zero otherwise.
v is uniform over the topic's labelled pages, or all pages for global PR.
Updates are simultaneous; new rank is used only on the next iteration.
Stop at maximum absolute change <1e-8, cap 200; reject a non-converged build.
Each vector sums to one. A topic vector can legitimately contain zeros.

## Online API and ranking

~~~python
from project.search import load_data, search

data = load_data(mode="hybrid")          # once
results = search("computer software", data, mode="hybrid", lexical_weight=0.9)
~~~

The Python functions retain mode="tfidf" defaults for compatibility. The CLI
defaults to hybrid. Load a link-aware mode before searching with any link mode.

Candidates = union of postings; require positive cosine.
- tfidf: cosine ranking.
- pagerank: global PageRank ranking.
- tspr: importance = query topic weights @ topic_pagerank.
- hybrid: w*cosine + (1-w)*(importance/max(importance)); default w=0.9.
  The maximum covers the whole corpus, not just candidates.

Results include doc_id, score, cosine, link_score, scaled_link_score, title,
preview, url and topic. The scaled component is supplied for TSPR/hybrid modes.
No PageRank iteration or network request happens during search.
The mixture is an importance score; topic-dependent dangling handling means we
do not assert equivalence to a fresh mixed-teleport PageRank run.

## Evaluation and attack contract

The frozen project/evaluation_judgments.json contains query-specific relevant
IDs, not topic-based inferred relevance. Labels are assistant-generated and
await independent human validation. They are development judgments, not held out.

evaluate.py evaluates four modes with macro P@K, R@K, F1@K and binary NDCG@K.
P uses K even if fewer results are returned; F1 is averaged per query.
Latency excludes loading/printing; traced memory is not process RSS.

experiment.py builds separate clean/control/attacked directories. Ten farm
pages duplicate the target's text and label. Control has no farm outlinks;
attacked adds exactly one target link per farm page. No legitimate page is
altered and no real Wikipedia page is edited. The target is explicitly supplied.
Farm pages are nonrelevant duplicates; existing qrels remain unchanged.

Clean-to-attacked measures whole injection. Control-to-attacked isolates links
while keeping text, IDF, topic model and document order identical.
summary.json holds comparisons, target positions, hashes and a development-only
weight sweep. The lexical weight is a fixed design setting, not an optimal value.

The core attack experiment is implemented. A graphical UI, TrustRank defense,
hard-versus-soft topic selection and damping experiments remain optional stretch
work, not necessary to execute this delivered pipeline.
