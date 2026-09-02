# S7. Baseline Comparability

Standard benchmark results use each benchmark's reported judged candidate pool.
`Shared evaluated pool` means an external method was adapted to score that same
pool; it does not imply that every method shares an identical native
candidate-generation procedure.

- SPECTER2-PRX aggregates publication-level proximity; SPECTER2-CLS uses a
  pooled reviewer-profile representation.
- SciRepEval-CTRL and RATE-8B are adapted to the reported evaluated pools rather
  than evaluated through their native end-to-end pipelines.
- RATE-8B receives aggregated title-abstract reviewer histories rather than its
  native keyword-profile input.
- The SIGIR BM25-evidence top-50 run does not cover the complete canonical
  189-candidate judged pool and is treated as a cautionary, non-fully-comparable
  result.
- ConfusEval within-pipeline gains compare each second stage with its own first
  stage and should not be interpreted as absolute cross-model dominance.

These protocol qualifications define the scope of comparative claims in the
paper.
