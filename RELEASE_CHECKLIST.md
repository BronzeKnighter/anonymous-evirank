# Release Checklist

Use this checklist before pushing the repository to GitHub.

## Safe To Upload

- `r2reviewer/`: EviRank model implementation.
- `tools/`: preprocessing and evaluation utilities.
- `scripts/`: runnable entrypoints.
- `configs/`: locked model configuration.
- `docs/`: data format, reproducibility, and anonymization notes.
- `examples/`: toy data and toy evaluation scripts.
- `results/`: lightweight EviRank-only summary tables.
- `README.md`, `LICENSE`, `CITATION.cff`, `MODEL_CARD.md`, `.gitignore`.

## Do Not Upload

- Raw benchmark datasets or private annotation files.
- SPECTER2 caches, `.npy` embedding caches, model weights, or checkpoints.
- Paper drafts, PDFs, DOCX files, review notes, or submission metadata.
- External repositories or competing baseline implementations.
- Files containing absolute local paths or local usernames.
- Files containing author names, affiliations, emails, or self-identifying comments during anonymous review.

## Pre-Push Checks

```bash
find . -type f -size +1M
rg -n "(local-user-name|absolute-local-path|private-project-name|author-email|paper-draft-file)" .
python -m py_compile $(find r2reviewer tools -name "*.py")
git status --short
```

The first two commands should return no concerning files or text. The Python compile command requires the dependencies listed in `requirements.txt`.
