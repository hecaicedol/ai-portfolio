# Evaluation harness

A small synthetic corpus + benchmark runner used to populate the metrics
table in the project root README.

## Layout

```
eval/
├── corpus/                 # 15 hand-written synthetic docs (4 types × 3 difficulties)
│   ├── invoice_*.json
│   ├── receipt_*.json
│   └── contract_*.json
├── benchmark.py            # runner + grader + aggregator
└── results/                # generated — git-ignored
    ├── <label>_results.json   # per-doc raw output
    ├── <label>_summary.json   # aggregate stats
    └── <label>_summary.md     # rendered for the README
```

## Corpus format

Each document is a single JSON file with the same schema:

```json
{
  "id": "invoice_001_clean",
  "document_type": "invoice",
  "difficulty": "easy",
  "notes": "free-text explanation of what makes this doc easy/medium/hard",
  "content": "the raw document text the pipeline ingests",
  "ground_truth": {
    "<required_field>": "expected value"
  }
}
```

`document_type` must be one of: `invoice`, `receipt`, `contract`, `purchase_order`, `generic`.
The `ground_truth` keys are the fields the validator considers required for
that document type (see `agents/validator.py`).

## Running

From `backend/`:

```bash
# Dry-run with a deterministic stub — verifies the harness without spending
# money. Every doc passes with accuracy=1.0 by construction.
.venv/Scripts/python -m eval.benchmark --mode dry-run

# Real run against Claude — needs ANTHROPIC_API_KEY in the environment.
$env:ANTHROPIC_API_KEY = "sk-ant-..."
.venv/Scripts/python -m eval.benchmark --mode real --label baseline-2026-05
```

Results land in `eval/results/<label>_*`. The markdown file is what gets
pasted into the root `README.md` once we're happy with the numbers.

## How accuracy is graded

`benchmark.field_match` does lenient equality, per field type:

| Type | Match rule |
|---|---|
| numeric | absolute diff < 0.01 |
| string  | case-insensitive after `strip()` |
| list    | element-wise field_match (order-sensitive) |
| dict    | each key in `expected` matches in `actual` |

Per-doc accuracy = matched fields / required fields. Overall accuracy is the
unweighted mean across docs.

## Cost estimate (real mode)

15 docs × ~2 LLM calls per pass (extract + critic), plus up to 2 reflection
cycles for failing docs. Worst case: 15 × 6 = 90 calls. At Claude Sonnet 4.5
list pricing (≈$3/MTok in, $15/MTok out), assuming ~2k in + 500 out per call,
that's roughly **$0.80–1.20** for a full run.
