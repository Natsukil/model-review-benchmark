# Benchmark data layout

The repository tracks the small, curated selections and manifests needed to
describe a benchmark run:

- `manifest.json`: source identifiers and checksums.
- `selections/`: deterministic benchmark profiles, including `review-hour`.

The following directories are intentionally ignored because they contain
downloaded or generated data:

- `raw/`: Hugging Face/Martian source snapshots.
- `cache/`: downloaded PR diffs and other runtime caches.
- `workspaces/`: temporary agent workspaces.
- `../outputs/`: per-run JSONL results and reports.
- `../reports/`: cross-model aggregate files.

Prepare the review data locally with:

```bash
python -m coder_review_benchmark prepare-review --profile review-hour
```
