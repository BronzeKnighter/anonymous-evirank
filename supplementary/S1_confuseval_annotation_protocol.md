# S1. ConfusEval Annotation Protocol

## Scope

ConfusEval tests query-specific reviewer discrimination within a plausible
shortlist. Stage 1 retrieves ten high-scoring non-assigned candidates per query
from the reviewer universe. Non-assignment alone is never treated as evidence
of unsuitability.

## Labels

- `hard_negative`: retrievable and superficially plausible, but the candidate's
  publications do not provide sufficient evidence for the query's specific
  task, method, setting, organism, disease, or modality.
- `ambiguous`: available evidence is insufficient for a safe negative judgment.
- `possible_positive`: the candidate is not an original assigned reviewer but
  has credible query-specific expertise evidence.

Annotators first identify the query's expertise constraints, inspect the
candidate's highest-ranked evidence papers, and prioritize specific scientific
alignment over retrieval score or broad field overlap. Uncertain cases default
to `ambiguous`; stage-1 rank never overrides semantic mismatch or match.

## Quality Control and Adjudication

ConfusEval contains 5,100 manually reviewed non-assigned candidates for 510
queries: 2,821 hard negatives, 2,138 ambiguous cases, and 141 possible
positives. Of these, 3,249 instances (63.71%) were double reviewed. Raw
agreement is 0.8119 and Cohen's kappa is 0.623.

Disagreements involving `hard_negative` versus `possible_positive` are escalated
for adjudication. A `hard_negative`/`ambiguous` disagreement defaults to
`ambiguous` unless the evidence supports a confident adjudicated decision. The
same conservative rule applies to `ambiguous`/`possible_positive` disagreements.

## Released Qrels

The diagnostic qrels contain 618 strict-positive pairs and 2,821 confirmed hard
negatives. Ambiguous and possible-positive candidates remain in the extended
annotations but are excluded from the negative class. This policy favors label
precision over coverage and does not claim that every unassigned reviewer is
unsuitable.
