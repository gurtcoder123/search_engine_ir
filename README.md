# Wikipedia Search Engine — Group 17

Information Retrieval course project: implement Topic-Sensitive PageRank from
first principles and evaluate how a synthetic spam farm affects retrieval.
Core numerical work uses NumPy and SciPy; NetworkX and Scikit-Learn are not used.

## Current implementation

- `project/crawler.py`: MediaWiki JSON API collection and adjacency lists.
- `project/reviewer.py`: console-based acceptance, rejection and manual topic labels.
- `project/graph.py`: sparse adjacency matrix for the reviewed corpus.
- `project/tests/`: offline API fixtures and review-to-graph checks.

Preprocessing, TF-IDF indexing, query processing, PageRank/TSPR and attack
evaluation are still upcoming. This repository does not yet contain a complete
search engine or measured retrieval results.

The intended data flow is:

```text
Seed titles grouped by topic
    -> BFS collection: documents + adjacency lists
    -> manual review and topic assignment
    -> final corpus + filtered adjacency lists
    -> sparse graph and text indexes
    -> ranking, retrieval and attack evaluation
```

See [the implementation guide](project/README.md) for the commands supported
by the current code.

## Shared document classes

The current definitions are in `project/crawler.py` and are also imported by
the reviewer and graph scripts.

### Document

One record represents one article:

| Field | Type | Meaning |
| --- | --- | --- |
| `page_id` | `int` | Wikipedia's page identifier |
| `doc_id` | `str` | Our stable ID, currently `wiki_<page_id>` |
| `title` | `str` | Canonical title returned by the API |
| `url` | `str` | Link to the article |
| `collected_at` | `str` | UTC collection timestamp |
| `revision_id` | `int` or `None` | Revision metadata recorded at collection time |
| `text` | `str` | Full plain text returned by the API |
| `preview` | `str` | First paragraph used for review/results |
| `outgoing_links` | `list[str]` | Outgoing article titles returned by the API |
| `topic` | `str` or `None` | Actual manual label; always `None` for new candidates |
| `discovered_topic` | `str` or `None` | Seed-group provenance, not a manual label |

### DocumentCollection

Stores the shared `documents: list[Document]` and a set of known Wikipedia
page IDs for deduplication. `add_document(document)` inserts a unique page;
`get_documents()` returns the stored list.

### WebCrawler

Collects article records and preserves their outgoing links. The existing
implementation owns its collection and exposes `crawl(...)`. The simpler
shared-instance design agreed below is **planned, not implemented yet**.

## Agreed next simplification (to be implemented by Vignesh)

Retain the three classes above. Give `WebCrawler` the existing collection as
a constructor argument, so other components can refer to the same object:

```python
# Proposed interface; the current crawler still uses crawl(...).
collection = DocumentCollection()
crawler = WebCrawler(collection, seed_titles)
documents, adjacency_lists = crawler.run(depth=2, max_pages_per_topic=40)
```

`seed_titles` is a dictionary with these five groups:

```python
seed_titles = {
    "technology": [...],
    "science": [...],
    "health_and_medicine": [...],
    "sports": [...],
    "politics": [...],
}
```

The planned `run()` method:

1. Picks one seed group at a time and starts a BFS queue with its titles.
2. Requests plain article text, metadata and outgoing titles from the API.
3. Checks Wikipedia page IDs to avoid storing the same article twice.
4. Constructs new documents with `topic=None`, recording the discovery group
   separately, and appends them to the supplied collection.
5. Enqueues outgoing titles when below the depth limit. Links are recorded
   whether or not BFS eventually follows them.
6. Stops a group after 40 new unique candidates, or when its queue is exhausted.
7. Matches saved links to the collected documents and returns documents plus
   adjacency lists. The sparse matrix is built only after manual review.

The crawler must keep the supplied collection object instead of replacing or
silently clearing it. Use a fresh collection to start a separate experiment.
Use a per-group visited set to avoid BFS cycles, and the collection's page-ID
set to prevent duplicates across groups. A shared page does not fill a second
group's new-document quota, but its outgoing links may still be explored.

## Shared graph conventions

- An adjacency list maps a source `doc_id` to destination `doc_id`s.
- Duplicate edges and self-links are excluded; a dangling page has an empty list.
- Rejecting an article removes both its list and references to it in other lists.
- The final matrix is SciPy CSR. Entry `[i, j]` is 1 if document i links to j.
- Matrix rows/columns follow the final `documents` list. The same ordering must
  be used for document TF-IDF vectors and PageRank/TSPR arrays.
- Although the code calls it an incidence matrix, it is mathematically a
  directed, binary adjacency matrix.

## Install and check

Run from the repository root with Python 3.10 or newer:

```powershell
python -m pip install -r project/requirements.txt
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/project_tests
```

Install `pytest` separately to run the tests. Crawling uses the public MediaWiki
Action API; keep request delays and follow API pagination to retain all links.

## Local files

`.gitignore` excludes local `tmp/`, `output/`, downloaded data, Python caches,
Word/PDF documents, and proposal-generation utilities. These remain available
locally and are not part of the code repository.
