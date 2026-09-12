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

### D-018 - WESAD release provenance and raw baseline (Phase 1)
- **Date:** 2026-09-12
- **Category:** ENGINEERING / FACT
- **Decision:** The project uses the official WESAD release downloaded from the University of Siegen public share (chain: UCI DOI 10.24432/C57K5T -> Siegen page -> sciebo `HGdUkoNlW1Ub0Gx`), `WESAD.zip` 2,249,444,501 bytes, locally computed SHA-256 `5e15d260...38fd71c` (no publisher checksum exists). Extracted unmodified to `data/raw/wesad/WESAD/`; the ZIP is retained. Checksum baseline committed as `data/manifests/raw_checksums_wesad.json` (77 files). Any future replacement of this baseline is a dataset-version change requiring a new decision entry.
- **Status:** agreed

### D-019 - WESAD factual findings adopted from the structural audit (resolves T-02; factual part of T-03)
- **Date:** 2026-09-12
- **Category:** FACT (verified; see `docs/dataset_notes.md` for provenance of each item)
- **Decision:** The following are treated as facts: 15 participants (S2-S11, S13-S17), all structurally usable, none excluded; label codes 0-7 with readme meanings, no undocumented codes; label rate 700 Hz on the RespiBAN timeline; chest 700 Hz x6 modalities, wrist ACC 32 / BVP 64 / EDA 4 / TEMP 4 Hz; the pkl chest ECG and wrist ACC each match their raw device file at a unique offset (the other modalities are inferred, not individually checked); label runs are empirically consistent with `quest.csv` intervals trimmed by ~10 s at each end (within one 700-Hz sample); exactly one contiguous baseline run (1140-1198 s) and one stress run (615-725 s) per participant. Recorded in `configs/wesad.yaml`.
- **Not decided here:** device choice (T-01), split strategy (T-04), selection unit (T-05), budget (T-06), detector (T-07), and the binary label mapping / exclusion list (proposed: positive = 2, negative = 1, excluded = {0, 3, 4, 5, 6, 7}) which remains a DESIGN proposal until approved.
- **Status:** agreed (facts); mapping provisional

### D-020 - Git history was rewritten once for author identity; frozen from here
- **Date:** 2026-09-12
- **Category:** ENGINEERING / PROVENANCE
- **What happened:** the two bootstrap commits were created with the operator's global git identity and a `Co-Authored-By: Claude` trailer. At the repository owner's explicit request (sole authorship: Srishti Rathi <srishtirathi723@gmail.com>, no AI co-author trailer), `git filter-branch` rewrote author/committer metadata and stripped the trailer **before the first push**. Tree contents were unchanged. Commit ids changed: `297872c` -> `c98193c` (bootstrap), `6e5496d` -> `3445fd0` (review fixes). Earlier review documents citing the old ids refer to the same trees.
- **Rule from this checkpoint:** no further history rewriting. Commit identities are preserved for reproducibility; corrections are made with new commits.
- **Status:** agreed

### D-021 - Phase-2 preprocessing contract (APPROVED after independent Phase-1 review)
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Primary device:** wrist Empatica E4. Rationale: keeps the main project focused; aligns the sensor family with the later Nurse case-study motivation; chest remains optional sensitivity work. Shared hardware must NOT be claimed to imply shared domain or label validity between WESAD and the Nurse data. (Resolves T-01.)
- **Primary window grid:** 60-s non-overlapping windows, anchored at synchronised pickle time t=0 for each participant, constructed from TIME ONLY. Windows are never restarted at true-label boundaries.
- **Candidate frame:** the time grid is constructed over the synchronised recording without consulting reference labels. Window existence and boundaries must not depend on stress/baseline labels. Complete provenance is preserved for every generated window (participant, window index, start/end sample and second, raw-code composition).
- **Primary binary reference:** raw code 1 = BASELINE reference; raw code 2 = PROTOCOL-STRESS reference; codes 0, 3, 4, 5, 6, 7 are NOT eligible for the primary binary target and are never silently converted to baseline/non-stress. Raw label values are preserved separately from any analysis label. Wording: "baseline versus protocol stress"; code 1 is not claimed to prove absence of psychological stress. (Resolves the mapping part of T-02.)
- **Window eligibility:** determined AFTER the time-only window exists. A window is eligible for the primary binary supervised task only when its complete 60-s interval is homogeneous raw code 1 or homogeneous raw code 2. Mixed-label / boundary windows are not majority-voted or relabelled; they are kept in provenance/audit outputs and marked ineligible.
- **Participants:** all 15 released participants stay in the primary preprocessing dataset. S6/S15 are not excluded for weak induction; S2/S17 are not excluded for chest-temperature caveats (wrist is primary). Any later exclusion needs a pre-declared quality reason and a log entry. (Resolves the remaining part of T-03.)
- **Inferential unit:** the participant. 60-s windows are not independent people.
- **Chest:** optional sensitivity analysis only; not required for minimum project completion.
- **HRV:** disabled for Phase 2 unless separately approved (T-10 stays open).
- **Status:** agreed (frozen). Reflected in `configs/base.yaml`, `configs/wesad.yaml`, `research_protocol.md` amendment A-1, `PROJECT_CONTEXT.md` invariant #19.

### D-022 - Phase-2 WESAD loader safety contract
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Contract:** the Phase-2 loader must (1) verify the committed raw checksum baseline BEFORE deserialising any pickle; (2) fail closed if verification fails or no baseline exists; (3) validate required array numeric dtypes, channel counts (`EXPECTED_CHANNELS`) and nominal rates on load; (4) never assume that a previous CLI audit run makes a later direct loader call safe - every loader entry point performs its own verification. Not implemented in this closeout.
- **Status:** agreed

### D-023 - Label-independence invariant for Study-A candidate construction
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE / INVARIANT
- **Rule:** candidate construction and window boundaries for the primary Study-A frame must be independent of held-out reference labels. Future selectors may use only explicitly permitted time-, signal- or model-derived information. The fields `raw_label`, `binary_analysis_label`, `eligibility`, and protocol-state boundaries are never selector inputs. Reference labels are used only afterwards to define the evaluation target, determine scoring eligibility, and compute full-reference comparison metrics. The selector is not implemented yet.
- **Status:** agreed (invariant #19 in PROJECT_CONTEXT.md)

### D-024 - Phase-2 feature set, processing parameters and provisional HR status
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE (signal processing; chosen without reference to labels)
- **Decision:** The canonical Phase-2 table `data/processed/wesad_windows_60s.parquet` (schema v1.0.0, `data/manifests/wesad_feature_schema.json`) carries 117 columns: 18 provenance, 8 reference-label, 24 quality, 67 numeric window features of which **59 are `model_feature: true`** (ACC 27, EDA 17, BVP raw statistics 7, TEMP 8) and **8 are `feature_provisional` (model_feature false)**: `bvp_beat_count`, `bvp_valid_ibi_count`, `bvp_beat_coverage`, `hr_mean/median/std/min/max`.
- **Processing (all deterministic, recording-level where filtering is involved, parameters in `configs/base.yaml: features`, PROVISIONAL):** ACC counts / 64 -> g, no filtering; EDA tonic = zero-phase Butterworth low-pass 0.05 Hz order 2 over the whole recording, phasic = raw - tonic, SCR-like peaks = phasic prominence >= 0.01 uS with >= 1 s spacing; BVP band-pass 0.5-8 Hz order 3, peaks with >= 0.33 s spacing and prominence >= 0.2 x MAD scale, IBIs valid in [0.33, 2.0] s, HR summaries need >= 10 valid IBIs and >= 50 % coverage; TEMP summaries only. Quality flags (provisional thresholds): ACC clip |count| >= 127, EDA < 0.01 uS, TEMP outside 20-45 degC, constant-signal eps 1e-9.
- **Why HR is provisional:** on the real release the in-house detector yields 30-45 % successive beat-to-beat HR changes > 20 bpm and `hr_max` pinned at the distance floor (174.5 bpm) in many windows (KI-21). Rather than tune a detector without a beat reference, the columns stay in the table for transparency and are excluded from `model_feature` until a validated detector or an IBI-consistency gate is approved. HRV remains disabled.
- **Not decided here:** anything in T-04..T-10; whether provisional HR columns are ever promoted.
- **Status:** agreed (feature set frozen for Phase 3 unless amended)

---

## Open decisions (TODO before the affected stage)

| ID | Decision needed | Blocks | Where |
|---|---|---|---|
| T-01 | ~~WESAD device stream~~ RESOLVED (D-021): wrist E4 primary; chest optional sensitivity | feature extraction | `configs/wesad.yaml: device` |
| T-02 | ~~Raw label codes~~ RESOLVED as FACT (D-019); ~~binary mapping~~ RESOLVED (D-021): 1 = baseline reference, 2 = protocol-stress reference, {0,3,4,5,6,7} ineligible | labels | `configs/wesad.yaml: labels` |
| T-03 | ~~Usable participants~~ RESOLVED (D-019, D-021): all 15 kept; no caveat-based exclusion | splits | `docs/dataset_notes.md` |
| T-04 | LOPO vs grouped k-fold outer split; inner fold count. NOT decided - `configs/base.yaml: evaluation.outer_split` is deliberately `null` so no executable default masquerades as an agreed choice | baseline | `configs/base.yaml: evaluation` |
| T-05 | Episode / selection-unit definition on WESAD; candidate generation must itself be label-independent, not just the final mask function. Audit CONFIRMS KI-05: exactly 1 baseline + 1 stress block per participant, so protocol-block selection would be an extremely coarse, condition-confounded annotation model (see KI-05 for the precise limitation). The primary candidate FRAME is now fixed (D-021: time-only 60-s grid at pkl t=0); what remains open is the selection UNIT/grouping on that grid, the annotation budgets, random-mask repetition count, and how budget is charged for windows outside baseline/stress. Engineer's recommendation (not adopted): fixed-length contiguous pseudo-episodes (e.g. 3-5 min) as the primary unit with detector-proposed episodes as a comparison | Study A | `research_protocol.md` S5; KI-05; KI-06 |
| T-06 | Budget unit and budget grid | Study A/B/C | `configs/base.yaml: observation_policies` |
| T-07 | Detector definition for `detector_triggered` | Study A | `research_protocol.md` S5 |
| T-08 | Study B specification: development roles (which participants play calibration/threshold/selection roles), detector access in development, coverage constraint or loss, tie handling, comparator; then the regret formula | Study B | `research_protocol.md` S6; KI-18 |
| T-09 | Study C specification: random sampling frame, overlap/budget accounting between targeted and random draws, 100 %-random same-budget comparator; then the estimator (naive pooled / audit-only / IPW) | Study C | `research_protocol.md` S7; KI-13 |
| T-10 | HRV feature gating threshold (beat coverage) | features | `configs/base.yaml: features.hrv` |
| T-11 | Nurse timestamp units / timezone / alignment (Gate 1) | Study D | `configs/nurse.yaml: timestamps` |
| T-12 | Nurse raw label values and binary mapping | Study D | `configs/nurse.yaml: labels` |
| T-13 | Coverage levels for tabular selective-prediction reporting | reporting | `configs/base.yaml: abstention` |
