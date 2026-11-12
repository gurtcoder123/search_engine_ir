# Reference experiment results

attack_summary.json is the saved clean/control/attacked experiment described
in docs/RESULTS.md. It includes per-query results, all four ranking modes,
target positions, metric changes, weight sensitivity, convergence details,
and source corpus/judgment hashes.

Reproduce with:

```powershell
python -X utf8 -m project.experiment --judgments project/evaluation_judgments.json --target-id doc-31524 --repeats 30
```

The rebuilt summary is output/attack/summary.json. Timing and allocation
measurements vary between runs and machines. On the same corpus and settings,
the deterministic retrieval rankings and effectiveness metrics should match.

Relevance judgments are provisional assistant-generated development labels,
not independently human-validated. The summary records that provenance.
