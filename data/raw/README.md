# data/raw — READ-ONLY

Place the downloaded, unmodified public datasets here:

| Directory          | Dataset                         | Source (to be recorded in docs/dataset_notes.md) |
|--------------------|---------------------------------|--------------------------------------------------|
| `wesad/`           | WESAD (Schmidt et al. 2018)     | official release archive, unmodified             |
| `nurse/`           | Nurse Stress (Hosseini et al. 2022) | official release, unmodified                 |
| `stress_predict/`  | Stress-Predict (Iqbal et al. 2022) | official release, unmodified                  |

Rules (research invariant #8 in `docs/PROJECT_CONTEXT.md`):

- Files in this directory are never edited, renamed, re-encoded, or partially deleted by code.
- All processing writes to `data/interim/` or `data/processed/`.
- After downloading, record SHA-256 checksums with `python -m wsr.utils.integrity snapshot data/raw/<dataset>`;
  the manifest goes to `data/manifests/raw_checksums_<dataset>.json` and is committed.
  `tests/test_raw_immutable.py` verifies the raw tree still matches the manifest.
- Nothing in this directory is committed to Git (see `.gitignore`).
