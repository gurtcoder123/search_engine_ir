# Measured results - 12 November 2026

Run: project.experiment, target doc-31524 (Triathlon), farm size 10, K=10,
30 timing repetitions per query. Versioned source: results/attack_summary.json
(also generated as output/attack/summary.json). Labels are
provisional assistant-generated development judgments, not independently human
validated. Relevant sets and query strings match the earlier evaluation.

## Before and after the attack

Each cell is clean / attacked. NDCG has binary relevance and log2 discount.

| Ranking | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.600 / 0.600 | 0.923 / 0.923 | 0.698 / 0.698 | 0.930 / 0.930 |
| Global PageRank | 0.220 / 0.220 | 0.294 / 0.294 | 0.242 / 0.242 | 0.261 / 0.261 |
| Pure TSPR | 0.360 / 0.360 | 0.531 / 0.531 | 0.411 / 0.411 | 0.454 / 0.452 |
| Hybrid, lexical weight 0.9 | 0.560 / 0.560 | 0.875 / 0.875 | 0.655 / 0.655 | 0.879 / 0.879 |

P/R/F1 changes are **0.000** in every mode. Control-to-attacked differences are
also zero for those three metrics. TSPR macro NDCG falls by 0.0018836484.
NDCG was added as an ordering diagnostic after observing the unchanged hit counts;
it is supplementary, not a preregistered primary endpoint.

Stable P/R/F1 does not mean stable ranks. The target is nonrelevant to the World
Cup intent; its weak cosine match allows it into the candidate pool.

| Target position | Clean | No-link control | Attacked |
| --- | --- | --- | --- |
| TF-IDF | 56 | 45 | 45 |
| Global PageRank | 15 | 15 | 5 |
| Pure TSPR | 5 | 5 | 3 |
| Hybrid | 35 | 35 | 28 |

The target's sports-topic score is 0.03381029 clean, 0.03232035 in the control,
and 0.06621586 attacked. The attack increases it by 95.8% relative to clean and
104.9% relative to control. These are topic-vector probabilities, not relevance
probabilities. Global target PageRank rises from 0.01029833 clean to 0.02178762.

## Interpretation

TSPR is better than global link-only ranking on this development evaluation.
Hybrid is better than pure TSPR and keeps the weak lexical target outside the
top ten. TF-IDF remains the strongest measured overall ranker. Do not claim
the hybrid universally improves on lexical retrieval or prevents link spam.

The identical-text no-link control is important: both control and attacked have
172 documents, identical IDF/text matrices and identical topic probabilities.
Only their farm outlinks differ. The whole clean-to-attacked injection also
changes corpus statistics and the topic teleport support; it does not isolate
link propagation by itself.

The farm is a local simulation of pages already admitted to the corpus and the
target's topic. It does not test bypassing manual approval or Wikipedia moderation.
No legitimate document's links are edited and no external request is made.

## Weight sensitivity on the same development judgments

The 0.9 default was fixed before the sweep, not selected as a measured optimum.

| Lexical weight | Clean F1@10 | Attacked F1@10 |
| --- | --- | --- |
| 0.00 | 0.411 | 0.411 |
| 0.50 | 0.554 | 0.543 |
| 0.70 | 0.596 | 0.596 |
| 0.85 | 0.643 | 0.643 |
| 0.90 | 0.655 | 0.655 |
| 0.95 | 0.688 | 0.677 |
| 1.00 | 0.698 | 0.698 |

This is descriptive sensitivity, not held-out validation. Greater lexical
weight works better in this corpus; a more representative graph and independent
test judgments are needed before choosing a deployment setting.

## Performance and verification

Clean snapshot: 162 documents, 8,703 terms, 28,119 nonzero TF-IDF weights,
715 internal links and 17 dangling pages. Clean/control/attacked builds all
converged. The clean build took 1.88 seconds excluding imports on this machine.

| Clean ranker | Mean query ms | Peak traced MiB |
| --- | --- | --- |
| TF-IDF | 0.667 | 15.34 |
| Global PageRank | 0.611 | 15.34 |
| Pure TSPR | 0.697 | 15.34 |
| Hybrid | 0.753 | 15.34 |

Timing excludes loading, printing and user input. Peak traced allocations cover
load plus one query pass and are not total process RSS. Measurements use the
approved-only experimental snapshot, not the original 200-record review store.

114 local tests pass, including sparse PageRank against an independent dense
solve, hybrid formula/endpoints, CLI search, metric arithmetic, preservation of
source review records, and a real clean/control/attack rebuild on a fixture.
No new live crawl or fresh-machine dependency installation was performed.

Independent human relevance review, more targets/queries and adaptive attacks
remain scientific validation needs. UI and trust-based defenses remain optional.
