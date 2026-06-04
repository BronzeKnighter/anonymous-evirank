# Anonymization Notes

This repository is prepared for anonymous review.

## Removed From This Release

- Author names, affiliations, emails, and local usernames.
- Raw datasets and manually adjudicated private annotation spreadsheets.
- Cached embeddings and trained checkpoints.
- External baseline repositories and implementations.
- Absolute local paths from experiment manifests.
- Paper drafts and PDF/DOCX files.

## Remaining Identifiers

The repository keeps method and benchmark names needed to understand the implementation:

- `EviRank`: the released model.
- `ConfusEval`: the hard-negative diagnostic benchmark format used by the evaluation script.
- `SPECTER2`: the external scientific document encoder dependency.

These names are methodological identifiers, not author identifiers.

## After Review

After the anonymous review period, update:

- `CITATION.cff` with the final paper metadata.
- `README.md` with the public repository URL and citation.
- `LICENSE` copyright holder if required by your institution.
