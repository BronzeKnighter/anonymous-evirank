# Data Format

EviRank expects a query set, a reviewer set, and graded reviewer relevance labels.

## Processed Query File

`queries_for_inference.json`

```json
[
  {
    "id": "query-paper-id",
    "title": "Query paper title",
    "abstract": "Query paper abstract"
  }
]
```

## Processed Reviewer File

`specter_reviewers.json`

```json
[
  {
    "id": "reviewer-id",
    "papers": [
      {
        "id": "reviewer-paper-id",
        "title": "Reviewer paper title",
        "abstract": "Reviewer paper abstract"
      }
    ]
  }
]
```

## Gold Labels

`gold_standard.json`

```json
{
  "query-paper-id": {
    "reviewer-id-1": 3,
    "reviewer-id-2": 2,
    "reviewer-id-3": 0
  }
}
```

The default training code treats label `3` as a hard positive and label `2` as a soft positive. Label `0` is non-relevant within the judged candidate pool.

## Raw Qrel-Style Input

If a benchmark provides paper, reviewer, and qrel JSONL files, use:

```bash
python tools/preprocess_paper_data.py \
  --papers data/raw/KDD_papers_test.json \
  --reviewers data/raw/KDD_reviewers_test.json \
  --qrels_soft data/raw/KDD_queries_test_soft.json \
  --qrels_hard data/raw/KDD_queries_test_hard.json \
  --out_dir data/processed/kdd \
  --gold_mode per_query
```

The preprocessing utility is included only to prepare inputs for EviRank.
