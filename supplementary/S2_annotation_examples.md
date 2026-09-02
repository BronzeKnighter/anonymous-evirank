# S2. Annotation Examples

The examples below illustrate decision patterns without reproducing restricted
benchmark text.

## Hard negative

The query and candidate share a broad area such as neuroimaging, but the query
requires a specific signal modality and prediction task that are absent from
the candidate's evidence papers. Broad semantic proximity is insufficient.

## Ambiguous

One evidence paper appears related to the query, while the remaining history is
in a neighboring subfield. The available titles do not support either a safe
rejection or a defensible positive judgment.

## Possible positive

Multiple evidence papers match a rare method, dataset, organism, or disease
named by the query. Although the candidate was not originally assigned, the
publication evidence supports a credible reviewer recommendation.

## Borderline rule

When topical overlap is strong but query-specific expertise remains uncertain,
the candidate is retained as `ambiguous`, not forced into the hard-negative
class. Annotation notes record the concrete evidence used for each decision.
