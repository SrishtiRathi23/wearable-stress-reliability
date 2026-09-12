# Decision Log

Format: one entry per decision. `Status` is `agreed`, `provisional` (agreed for now, expected to be revisited at a named point), or `superseded by D-xxx`. Decisions are never deleted; they are superseded.

Category tags: DESIGN CHOICE / ASSUMPTION / ENGINEERING.

---

### D-001 - Label-selection-aware reliability is the central research direction
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** The core study is how selective label observation affects apparent accuracy, calibration, model/policy ranking and abstention conclusions in wearable stress evaluation. Cross-context transfer, activity analysis, SHAP and NLP are supporting or optional.
- **Rationale:** The earlier broad proposal (transfer + activity + adaptation + calibration + abstention + SHAP + NLP + stressor prediction) overlapped substantially with existing work (Mihirette 2025, Kwon 2026, ReliaGate 2026, Farahani 2026) and had no single coherent question.
- **Alternatives rejected:** plain nurse-stress detection (already heavily researched); the broad multi-component proposal.
- **Status:** agreed

### D-002 - WESAD is the primary controlled study and the first dataset implemented
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Studies A, B, C are run first on WESAD, binary baseline vs protocol stress.
- **Rationale:** Protocol labels cover a large part of each recording, so hidden-label simulation can be compared to a fuller reference.
- **Status:** agreed. Label codes, device stream, and excluded conditions remain TODO until the audit (see `configs/wesad.yaml`).

### D-003 - Nurse Stress dataset is a bounded real-world case study, not a validation set
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Nurse analyses are limited to what validated labels support; no whole-shift accuracy claims; no clinical claims.
- **Status:** agreed

### D-004 - Stress-Predict is a later replication dataset, and is a controlled-protocol dataset
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE (with a factual correction)
- **Decision:** Used only after the WESAD experiment works, to replicate the observation-policy experiment under a different controlled protocol. It is NOT described as natural daily-life data.
- **Status:** agreed

### D-005 - Simple classical models first; no deep learning initially
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Logistic Regression, Random Forest, XGBoost, plus majority-class and (possibly) a simple EDA score. No CNN/LSTM/Transformer unless a later scientifically motivated question requires it.
- **Rationale:** The classifier is not the novelty; interpretability and reproducibility matter more; CPU-only must remain sufficient.
- **Status:** agreed

### D-006 - Participant-independent evaluation is mandatory
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Participant-level outer holdout; grouped inner splits; held-out participant influences nothing (imputation, scaling, feature selection, hyperparameters, calibration, thresholds). Random-row splits are diagnostic only.
- **Status:** agreed. LOPO vs grouped k-fold is TODO pending verified participant count.

### D-007 - Unknown Nurse periods remain unknown
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Unlabelled hospital time is `unknown`, never `non-stress`. `configs/nurse.yaml: labels.unknown_is_negative` must stay `false`.
- **Status:** agreed

### D-008 - Exact stress-cause prediction is removed from the core
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** No stressor-family / cause prediction from physiology as a core task. Free text stating a cause is never used as input to predict that cause. NLP only for organising survey text; SHAP only for feature reliance, if at all.
- **Status:** agreed

### D-009 - Mixed random + targeted audit is the candidate mitigation study (Study C)
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE (hypothesis, not assumed result)
- **Decision:** Study C compares targeted:random labelling mixtures at fixed budget. Fractions are config-driven and provisional (0/10/20/30/50 % random).
- **Open:** the estimator that uses the random component is TODO (see `research_protocol.md` Section 7).
- **Status:** agreed as a study; estimator provisional

### D-010 - 60-s non-overlapping windows as the starting point
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** 60 s / 60 s step. Sensitivity 30/60/120 s only after the central pipeline works.
- **Status:** provisional (sensitivity analysis planned)

### D-011 - Sigmoid/Platt calibration first
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** Platt-type calibration on development participants; isotonic only if enough independent development data exists (`calibration.isotonic_allowed: false` until decided).
- **Status:** provisional

### D-012 - Repository root is the working directory; package is `wsr` under `src/`
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Decision:** The Git repository root is `E:\Tinaas Minor Project` itself (no nested `wearable-stress-reliability/` folder). The Python package is `src/wsr/` (importable as `wsr`), not a top-level package literally named `src`.
- **Rationale:** A nested folder adds a level with no benefit. A package named `src` is a known anti-pattern (`import src.data...` breaks under installation and pytest rootdir conventions); the `src/<package>/` layout is standard and pip-installable (`pip install -e .`). The internal sub-package names match the brief exactly.
- **Status:** agreed

### D-013 - Python 3.12 via uv (3.11 not installed)
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Decision:** Use CPython 3.12.13 (managed by `uv`) in `.venv`. `pyproject.toml` allows `>=3.11,<3.14`.
- **Rationale:** The brief preferred 3.11; the laptop has 3.12, 3.13, 3.14 but not 3.11. 3.12 is the safest of the available versions for scipy/scikit-learn/xgboost/neurokit2 wheels; 3.14 is too new to rely on for the scientific stack. Installed versions are frozen in `requirements-lock.txt`.
- **Status:** agreed

### D-014 - Planned modules are not created as empty stubs
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Decision:** Sub-packages exist with docstrings listing the planned modules; individual modules (`load_wesad.py`, etc.) are created when implemented. Tests for not-yet-existing invariants are explicit `pytest.skip` placeholders with reasons, never silently passing.
- **Rationale:** Avoid inventing schemas for uninspected data; avoid tests that always pass.
- **Status:** agreed

### D-015 - Raw data immutability is enforced by a committed checksum baseline
- **Date:** 2026-09-12 (amended 2026-09-12 after independent review)
- **Category:** ENGINEERING
- **Decision:** After each dataset download, `python -m wsr.utils.integrity snapshot data/raw/<dataset>` writes a committed SHA-256 baseline covering every file in the dataset tree, including dataset-shipped documentation; the only exclusion is a repository-owned `.gitkeep` at the dataset root. `tests/test_raw_immutable.py` verifies added/removed/modified files against it.
- **Amendment (review finding):** ordinary `snapshot` refuses to overwrite an existing baseline, so a modified raw tree cannot be silently re-blessed. Replacing a baseline requires the explicit `--replace-baseline` flag (API: `replace=True`), is never called automatically, and represents a deliberate dataset-version change that must be reviewed and recorded here as a new decision entry (dataset, release/version, reason, date).
- **Status:** agreed

### D-016 - Seeding convention: one global seed, named child streams
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Decision:** `seed` in config; every stochastic component uses `wsr.utils.seeds.rng(seed, "<stream name>")` or `child_seed(seed, "<stream>")` for library `random_state` arguments.
- **Rationale:** Adding a new random consumer must not shift the random state of existing ones.
- **Status:** agreed

### D-017 - Dependency lock file excludes the local editable package
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Decision:** `requirements-lock.txt` is generated with `uv pip freeze --exclude-editable` and lists third-party pins only; the local package is installed separately with `-e .`.
- **Rationale:** Independent review found an absolute `-e file:///E:/...` entry that made the lock file machine-specific.
- **Status:** agreed

---

## Open decisions (TODO before the affected stage)

| ID | Decision needed | Blocks | Where |
|---|---|---|---|
| T-01 | WESAD device stream (wrist / chest / both) | feature extraction | `configs/wesad.yaml: device` |
| T-02 | WESAD raw label codes and excluded conditions (verify against readme) | labels | `configs/wesad.yaml: labels` |
| T-03 | Usable WESAD participants and any exclusions with reasons | splits | audit -> `data/manifests/` |
| T-04 | LOPO vs grouped k-fold outer split; inner fold count | baseline | `configs/base.yaml: evaluation` |
| T-05 | Episode / selection-unit definition on WESAD; candidate generation must itself be label-independent, not just the final mask function | Study A | `research_protocol.md` S5; KI-06 |
| T-06 | Budget unit and budget grid | Study A/B/C | `configs/base.yaml: observation_policies` |
| T-07 | Detector definition for `detector_triggered` | Study A | `research_protocol.md` S5 |
| T-08 | Study B specification: development roles (which participants play calibration/threshold/selection roles), detector access in development, coverage constraint or loss, tie handling, comparator; then the regret formula | Study B | `research_protocol.md` S6; KI-18 |
| T-09 | Study C specification: random sampling frame, overlap/budget accounting between targeted and random draws, 100 %-random same-budget comparator; then the estimator (naive pooled / audit-only / IPW) | Study C | `research_protocol.md` S7; KI-13 |
| T-10 | HRV feature gating threshold (beat coverage) | features | `configs/base.yaml: features.hrv` |
| T-11 | Nurse timestamp units / timezone / alignment (Gate 1) | Study D | `configs/nurse.yaml: timestamps` |
| T-12 | Nurse raw label values and binary mapping | Study D | `configs/nurse.yaml: labels` |
| T-13 | Coverage levels for tabular selective-prediction reporting | reporting | `configs/base.yaml: abstention` |
