# v0.2 robustness results

This aggregate sensitivity package reuses the local DreamViews TSV from Zenodo [10.5281/zenodo.19161757](https://doi.org/10.5281/zenodo.19161757), licensed CC BY-NC 4.0. Derived aggregates are attributed to the dataset creator and shared under CC BY-NC 4.0. The source dataset, report-level predictions, raw duplicate pairs, author IDs, titles, and tag values are not included.

Reproduce with:

```bash
python -m lucidbench.robustness --config configs/v0.2-robustness.yaml
```

The package contains 30 repeated author-disjoint split metrics and summaries, ten capped-author split metrics and summaries, author-weighted metrics, a text/metadata artifact audit, a structural-only negative-control classifier, and a cross-author exact/near-duplicate audit with a post-removal baseline rerun. Seed-level repeated and capped metrics are provided in addition to distribution summaries. No individual report or author identifiers are exported.

The structural-only negative control excludes all lexical tokens, including forum abbreviations. The abbreviation counts are reported separately in the descriptive audit. Near-duplicate detection uses an approximate SimHash candidate search, so the reported count is not guaranteed exhaustive. See `docs/RESEARCH_PROTOCOL.md` for exact definitions and limitations.
