# S5. Runtime and Scalability

For `K` candidate reviewers, average publication-history length `M`, embedding
dimension `d`, and retained evidence budget `L`, evidence scoring costs
`O(KMd)` before retaining the top `L` papers. Expensive query-publication
matching is restricted to the Stage-1 candidate pool rather than the complete
reviewer universe. Reranker inference is linear in the number of resulting
candidate feature vectors.

The reported measurements cover cached Stage-2 judged-pool reranking, not model
training, Stage-1 encoding, or one-time embedding construction. Across the four
benchmarks, Stage 2 takes 0.052--0.164 seconds per query. Exact measurements are
in [`runtime_stage2.csv`](runtime_stage2.csv).

Profiling used one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB), an AMD Ryzen 9
9950X CPU, and 192 GB system memory. Embeddings and trained fold checkpoints
were cached before timing.
