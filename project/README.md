# Wikipedia collection, review and graph

The workflow has three simple scripts:

1. `crawler.py` gathers candidates and their adjacency lists.
2. `reviewer.py` lets you accept/reject articles and assign their actual topics.
3. `graph.py` builds a sparse matrix from the reviewed corpus.

Run the following from the workspace root. Python 3.10+ is required.

## 1. Provide your seeds and crawl

Install the dependencies and copy the empty seeds template:

```powershell
python -m pip install -r project/requirements.txt
Copy-Item project/seeds.example.json project/seeds.json
```

Edit `project/seeds.json` and put your Wikipedia titles/URLs in each topic's
list. You can provide several starting points per topic. All five lists in
the template must be filled before using it.

```powershell
python project/crawler.py --seeds-file project/seeds.json --depth 2 --max-pages-per-topic 40
```

Depth defaults to 2; seeds are depth 0. The limit defaults to 40 **new,
globally unique articles per seed group**, or at most 200 across five groups.
Groups can collect fewer if the frontier runs out. Repeated pages are stored
once and do not fill another group's quota. An already-collected page can
still be followed as a bridge when exploring a later group's seeds.

Every document has `topic=None`. The seed group is recorded separately in
`discovered_topic` and is only a suggestion, not a topic assignment.

For a small single-group check:

```powershell
python project/crawler.py --seed "Information retrieval" --topic technology --depth 1 --max-pages-per-topic 3 --output-dir tmp/crawl_check
```

Use `--user-agent` to provide a descriptive identifier including your contact.
Requests are sequential, with a configurable delay and bounded retries that
respect server backoff. `--max-links-per-page` defaults to 30 and only limits
which links BFS follows, not which outgoing links are recorded. Set it to 0
to follow all eligible links.

The crawler returns these two objects:

```python
from project.crawler import WebCrawler

crawler = WebCrawler(seed_urls={
    "technology": ["your first seed title", "your second seed URL"],
    "science": ["your science seed"],
})
documents, adjacency_lists = crawler.crawl(depth=2, max_pages_per_topic=40)
crawler.save("project/data/candidates")
```

`documents` is a list of Document objects. Each includes the full article
text, first-paragraph preview, title, URL, revision ID and raw outgoing titles.

`adjacency_lists` maps source document IDs to destination document IDs:

```python
{
    "wiki_101": ["wiki_202", "wiki_303"],
    "wiki_202": [],
    "wiki_303": ["wiki_101"],
}
```

The lists include all API-reported eligible links whose destinations are
collected, even links BFS did not follow. Uncollected destinations do not
become graph nodes. Redirect aliases are resolved before producing these
lists. There is no sparse matrix at the crawling stage.

By default, outputs go to `project/data/candidates/`:

- `documents.jsonl`: complete candidate records.
- `adjacency_lists.json`: unique internal document-to-document links.
- `manifest.json`: candidate order, seed-group counts and review status.
- `crawl_log.json`: collection/skip/error records.
- `snapshots/wiki_<page_id>.json`: per-article checkpoints, enabled by the CLI.

The crawler reads structured JSON fields directly: `extract` is plain article
text (`explaintext=1`), `links` contains outgoing titles, and the other fields
provide page metadata. It follows link continuation tokens to collect every
batch. There is no BeautifulSoup, HTML parsing or `action=parse` request.
The first paragraph of the plain text becomes the preview.

Links are the API's main-namespace page links, which can include links supplied
by templates/navigation; we no longer apply a custom article-body HTML filter.
`revision_id` records the revision metadata returned at collection time, but
this simpler current-page API mode is not an exact revision-pinned fetch.
The saved local text and links are the frozen experimental snapshot.
Tokenization and stemming for retrieval happen after corpus review. Checkpoints
preserve downloaded records on later failure; they are not an automatic
resume cache. Use a fresh output directory for separate experiments.

## 2. Review and label

```powershell
python project/reviewer.py
```

Each article shows its title, preview, URL and seed provenance. Choose:

- 1-5: accept and assign one of the five topics.
- r: reject.
- s: skip for later, or leave an existing decision unchanged.
- v: show the saved full article text.
- q: quit; decisions already made remain saved.

Decisions are written immediately to
`project/data/candidates/review_decisions.json`. Run the command again to
resume. Use `--review-again` to revisit earlier decisions, including rejections.

Only after all articles have an accept/reject decision does the script export
the reviewed corpus to `project/data/reviewed/`. Skipped articles remain
pending. The original candidate documents and lists are not overwritten.

Final outputs are `documents.jsonl`, `adjacency_lists.json` and
`manifest.json`. Rejected documents are removed as sources **and** destinations.
Accepted documents keep their original document IDs and order; pages with no
remaining links keep empty lists. Topic counts are displayed so you can aim
for a balanced final corpus (approximately 20 articles per topic).

For other directories, use `--input-dir` and `--output-dir`.
Programmatically, `finalize_documents(documents, adjacency_lists, decisions)`
returns the accepted documents and filtered lists without modifying the raw
objects.

## 3. Build the final matrix

```powershell
python project/graph.py
```

This reads `project/data/reviewed/` and saves:

- `incidence_matrix.npz`: SciPy CSR matrix.
- `graph_manifest.json`: final `doc_ids` and `doc_to_row`.

The project calls it an incidence matrix; mathematically it is a directed,
binary adjacency matrix. Entry [i, j] is 1 when final document i links to
final document j. Self-links, duplicate edges and external destinations are
excluded. Isolated/dangling pages remain present. The script prints edge,
dangling-page and isolated-page counts for a quick connectivity check.

```python
from project.graph import build_incidence_matrix

incidence_matrix = build_incidence_matrix(final_documents, final_adjacency_lists)
doc_ids = [document.doc_id for document in final_documents]
doc_to_row = {doc_id: row for row, doc_id in enumerate(doc_ids)}
```

Build the text vocabulary, IDF and TF-IDF matrix from this same reviewed
document collection and order. If review decisions change later, rerun graph
construction and rebuild the text indexes/rank vectors too.

## Offline tests

With pytest installed:

```powershell
python -m pytest project/tests -q -p no:cacheprovider --basetemp=tmp/review_workflow_tests
```

These tests do not access Wikipedia. They cover per-group quotas, shared-page
deduplication, full outgoing links despite discovery limits, redirect handling,
review/resume, rejection filtering, and the final matrix/save workflow.
