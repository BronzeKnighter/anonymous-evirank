# Model Card: EviRank

## Overview

EviRank is an evidence-grounded retrieve-and-rerank model for reviewer recommendation. It first retrieves candidate reviewers using SPECTER2-style scientific document embeddings and then reranks candidates with query-specific evidence features derived from the candidate reviewer's publication history.

The repository also provides locked configurations, evaluation scripts,
lightweight result summaries, and paper-aligned supplementary analyses.

## Intended Use

EviRank is intended for research on reviewer recommendation, evidence-aware ranking, and hard-negative evaluation. It is designed to support shortlist refinement and diagnostic evaluation, not automatic reviewer assignment.

## Inputs

The model expects:

- A query paper with title and abstract.
- A reviewer profile represented as a list of past papers with titles and abstracts.
- Optional graded qrels for training and evaluation.

## Outputs

The standard inference output is a ranked list of reviewer identifiers and optional scores:

```json
{"query_id": "q1", "recommend_reviewer": ["r1", "r2"], "recommend_score": [0.81, 0.42]}
```

## Method Summary

EviRank uses:

- Stage-1 semantic retrieval with SPECTER2-style embeddings.
- Reviewer-level proximity aggregation over publication embeddings.
- Query-specific evidence selection from reviewer publications.
- Dense interaction, lexical, profile, and attention/statistical features.
- A dual-head MLP reranker.
- Optional score fusion with the Stage-1 anchor score.

The locked configuration is documented in `configs/evirank_locked_standard.json`.

## Limitations

- The release does not include raw datasets or trained checkpoints.
- Reviewer recommendation is a decision-support task. Conflict-of-interest filtering, workload balancing, availability, diversity policy, and final editorial judgment remain external constraints.
- Hard-negative diagnostic results should not be interpreted as complete real-world assignment utility.
- Proxy diagnostics are auditing signals, not fairness guarantees.

## Ethical Considerations

Models that rank experts can affect visibility and opportunity. Users should audit exposure patterns, avoid automatic assignment, disclose evidence used for recommendations, and keep editors or program chairs in the decision loop.
