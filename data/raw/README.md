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
- Immutability is protected by a committed checksum baseline. After downloading, run once:
  `python -m wsr.utils.integrity snapshot data/raw/<dataset>`; the baseline goes to
  `data/manifests/raw_checksums_<dataset>.json` and is committed. It hashes every file in the
  dataset tree, including documentation shipped with the dataset (the only exclusion is a
  repository-owned `.gitkeep` at the dataset root).
- `python -m wsr.utils.integrity verify data/raw/<dataset>` (and `tests/test_raw_immutable.py`)
  reports added, removed and modified files against the baseline.
- `snapshot` refuses to overwrite an existing baseline, so a modified tree cannot be silently
  re-blessed. Replacing a baseline (`--replace-baseline`) is only for a deliberate new dataset
  release; it is a dataset-version change that must be reviewed and logged in `docs/decisions.md`.
- Nothing in this directory is committed to Git (see `.gitignore`).
