# Presentation and demo guide

Group 17: Vignesh K S and Thanishka Thatikonda.

## Before presenting

Use the supplied saved corpus; do not start a fresh crawl during the demo.
From the repository root:

```powershell
python -X utf8 -m project.search --mode hybrid
```

Try `turing test in robotics`, then `neutrino detector experiment`, then `q`.
Hybrid prints cosine and scaled TSPR separately. These are illustrative queries,
not a separate held-out evaluation. The first query is an earlier diagnosed case.

## A 10-15 minute presentation

1. **Problem and scope (1 minute).** Build a small Wikipedia search engine and
   reproduce Topic-Sensitive PageRank; test a local ten-page link-spam farm.
2. **Data and offline pipeline (2 minutes).** Show the approved-contents JSON.
   Explain crawling, manual labels, shared normalization, postings, sparse
   TF-IDF and the five precomputed rank vectors. Save all artifacts offline.
3. **TSPR (2 minutes).** A link shares the source's old score divided by its
   outgoing degree. Teleportation goes to the chosen topic, dangling mass follows
   the same distribution, updates are simultaneous, and iteration converges.
4. **Query processing and live search (2 minutes).** Union postings, cosine
   eligibility, unigram topic probabilities, weighted topic scores, hybrid
   lexical/link combination. There is no PageRank iteration during a query.
5. **Attack design (2 minutes).** Show output/attack/summary.json and the report.
   Explain clean, identical-text no-link control, and attacked farm. All ten
   synthetic pages link to Triathlon. No legitimate page is modified.
6. **Results and limitations (2-3 minutes).** Explain why an irrelevant target
   can gain rank without changing P@10; NDCG and target positions reveal ordering.
   Hybrid helps against this weak lexical target but is not a proven anti-spam
   defense. Discuss sampled graph, coarse topics and provisional judgments.
7. **Close (1 minute).** Replication and stress testing are the main contribution.
   The observed ranking weakness matters more than claiming universal quality.

## Compare the earlier ranking problem

```powershell
python -X utf8 -m project.search --mode tspr
```

Enter `turing test in robotics`. In the saved clean corpus, Turing test ranks
below Artificial intelligence in pure TSPR because the final ordering uses only
importance. Run the same query in hybrid: Turing test ranks first because its
stronger textual match contributes to the final score.

## Show attacked search without changing the clean data

```powershell
python -X utf8 -m project.search --data-dir output/attack/clean --mode tspr
python -X utf8 -m project.search --data-dir output/attack/attacked --mode tspr
python -X utf8 -m project.search --data-dir output/attack/attacked --mode hybrid
```

Enter `FIFA World Cup tournament seeding` each time. Triathlon is an unrelated
sport for this intent. Explain its rank and the two score components rather
than calling its topic label wrong. It really belongs to the broad sports topic.

Re-run the full experiment ahead of time if needed:

```powershell
python -X utf8 -m project.experiment --judgments project/evaluation_judgments.json --target-id doc-31524 --repeats 30
```

## Files to explain, in order

config.py -> collection/crawler.py -> collection/reviewer.py ->
indexing/normalize.py -> indexing/indexer.py -> indexing/tfidf.py ->
ranking/topic_model.py -> ranking/tspr.py -> preprocess.py -> search.py ->
evaluate.py -> experiment.py.

## Likely questions

**Why not just TF-IDF?** Textual relevance is essential; TSPR adds a different
notion of graph importance. Our measured lexical baseline remains strong, and
adding a link signal is not guaranteed to improve every collection.

**Is this an exact recreation of the original paper?** The central topic-biased
vectors and query-weighted importance are replicated. Five manually labelled
Wikipedia topics, add-one smoothing, our explicit dangling policy and the hybrid
score are project adaptations. Hybrid is evaluated separately from pure TSPR.

**Why can spam pages help without real backlinks?** In this test they are
admitted to the topic's teleport set, so each receives restart probability and
passes link-following mass to the target. The experiment assumes admission; it
does not demonstrate bypassing Wikipedia moderation or our manual reviewer.

**Why use a control?** Adding documents also changes IDF and topic statistics.
The control has the exact same documents as the attack, so their difference
isolates the farm's outgoing links.

**Why can P/R/F1 stay equal?** At a fixed K they depend on the number of relevant
hits, not their order. A nonrelevant result moving within the top ten may change
NDCG while leaving all three unchanged.

**What still needs scientific validation?** Independent human relevance review,
more queries/targets, a more representative graph, held-out parameter selection
and stronger adaptive attacks. These are limitations, not missing core modules.

**Would finer topics fix everything?** No. They may improve query-topic matching,
but link importance still differs from relevance to an exact question. Keeping
the lexical signal is necessary; topic specificity alone is no guarantee.

**What are possible next steps?** Title/phrase signals, redirect resolution,
duplicate detection, carefully vetted teleport seeds and evaluated trust-based
defenses. None should be described as implemented here.
