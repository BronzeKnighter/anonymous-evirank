# S3. Calibration Sensitivity

We vary the reranker coefficient `lambda` in

`final = lambda * reranker + (1 - lambda) * Stage1 anchor`

after per-query score normalization, while keeping the trained reranker fixed.
`lambda=0` is anchor-only and `lambda=1` is reranker-only.

The `lambda=0` and reported `lambda=0.2` rows use the locked Retriever-only and
EviRank runs from Table III, respectively. This keeps both endpoints aligned
with the accepted main comparison and avoids changes caused only by re-exporting
tied candidate scores.

The preferred coefficient is dataset-dependent. Larger reranker weights improve
KDD and SIGIR, whereas NeurIPS and SciRepEval favor conservative interpolation.
The paper therefore uses `lambda=0.2` as a fixed cross-dataset operating point,
not as a claim that it is optimal for every benchmark.

Exact values for NDCG, MAP, and precision are in
[`calibration_sensitivity.csv`](calibration_sensitivity.csv).
