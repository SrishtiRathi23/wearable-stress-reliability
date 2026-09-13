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

### D-025 - EDA derived features have recording-context dependence (ACCEPTED offline design choice)
- **Date:** 2026-09-12 (Phase-2 closeout)
- **Category:** DESIGN CHOICE
- **Decision:** The EDA tonic estimate is a zero-phase (forward-backward) Butterworth low-pass over the whole participant recording; phasic = raw - tonic; SCR-like peaks are detected on the phasic component. Consequently `eda_tonic_*`, `eda_phasic_*` and `eda_scr_*` are influenced by signal before and after the window (a 0.05-Hz filter has a long impulse response). This is accepted as an OFFLINE approximation for the primary participant-held-out analysis because it is deterministic and label-independent (it is not reference-label leakage). These features must NOT be described as strictly window-local, as causal, or as deployable real-time features; a separate causal/online implementation would be needed for that. Earlier wording ("a few seconds of neighbouring signal") was incorrect and has been removed.
- **Status:** agreed

### D-026 - SCR missingness contract
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** In a window with no detected SCR-like peak: `eda_scr_count` = 0, `eda_scr_amp_sum` = 0 (sum over an empty set), `eda_scr_amp_mean` and `eda_scr_amp_max` = NULL (amplitude is undefined when no event exists). Nulls are never replaced by zero in the canonical table. In Phase 3 these nulls are imputed only inside the training partition of the model pipeline. Ordinary zero-peak missingness (`q_eda_decomposition_ok` True) must remain distinguishable from a decomposition/extraction failure (`q_eda_decomposition_ok` False or `q_feature_error` non-empty, where all decomposition-derived EDA features are null).
- **Status:** agreed

### D-027 - BVP / HR contract
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Decision:** The 8 beat/HR-derived columns (`bvp_beat_count`, `bvp_valid_ibi_count`, `bvp_beat_coverage`, `hr_mean/median/std/min/max`) stay in the table for audit with role `feature_provisional` (model_feature false) and must not enter Phase-3 models. The pulse detector is not tuned or replaced now (KI-21). The 7 raw/simple BVP statistics remain eligible model features. `q_bvp_hr_available` means only that the detector met mechanical count/coverage criteria; it is not a claim of physiological validity.
- **Status:** agreed

### D-028 - Outer/inner evaluation design (resolves T-04)
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE (frozen)
- **Outer:** leave-one-participant-out, 15 folds. For outer participant p: p is completely untouched test data; the other 14 form the outer training/development set. No outer-test participant may influence feature imputation, scaling, feature selection, hyperparameters, classification threshold, calibration, abstention policy or model choice.
- **Inner (model selection):** 4-fold participant-grouped validation within the 14 outer-training participants using `sklearn.model_selection.StratifiedGroupKFold` (group = participant_id; stratified on the analysis label), computed from outer-training data only, with the named seed stream `inner_split`. Exact participant memberships are saved to split manifests and validated before use (participant-disjoint, every outer-training participant appears in exactly one validation fold). The SAME inner folds are reused across all candidate models.
- **Inner score:** balanced accuracy computed separately for each validation participant; score(candidate) = equal-weight mean of these participant-level values over all 14 inner-validation participant appearances. Never window-pooled; never a mean of fold means (folds hold unequal participant counts).
- **Status:** agreed. `configs/base.yaml: evaluation` now carries these values.

### D-029 - Phase-3 modelling contract (frozen before any model code)
- **Date:** 2026-09-12
- **Category:** DESIGN CHOICE
- **Purpose:** comparative baseline analysis. Candidates: prevalence/majority baseline, regularised Logistic Regression, constrained Random Forest, constrained XGBoost. No candidate is discarded for scoring lower; predictions from every candidate are saved because Study A asks whether selective label observation alters apparent performance, ranking and selection.
- **Model inputs:** the exact ordered `role == "feature"` list from `data/manifests/wesad_feature_schema.json` (currently 59), obtained through `wsr.features.schema.select_model_features`, which fails on a missing feature, a duplicate column, or any schema/table disagreement. Never "all numeric columns", never `FEATURE_NAMES` (contains provisional columns), never quality/ID/time/label/eligibility/provenance fields.
- **Supervised rows:** training and metric computation use `binary_eligible == True` rows only.
- **Threshold:** fixed 0.5 on the positive-class probability; not tuned in Phase 3 (threshold tuning belongs to later policy-selection experiments). Raw positive-class probabilities are saved wherever `predict_proba` exists. Majority baseline: predicts the training-partition majority analysis label for every window and outputs the training-partition prevalence of class 1 as its constant probability.
- **Preprocessing:** imputation (median from the training partition) and, for Logistic Regression only, standard scaling, are fitted inside EVERY training partition: inner-training participants -> applied to inner-validation participants; after hyperparameter selection, refit on all 14 outer-training participants -> applied to the untouched outer participant. Never fitted globally. Tree models are not scaled. No iterative/KNN imputation. If a model feature is entirely missing in a training partition, its imputation constant is 0.0 and the event is logged in the run manifest (the feature is then constant in that partition); the pipeline never consults validation/test data.
- **Class / sample weights:** none invented silently. `class_weight in {None, "balanced"}` may appear as a predeclared grid option for LR and RF; any XGBoost weighting is predeclared and computed from training data only. Participants are never weighted using outer-test information.
- **Grids:** small and predeclared in the Phase-3 prompt; no Optuna, Bayesian optimisation, large random search, neural networks or broad feature selection.
- **Prediction output (for Study A):** for every outer fold and every candidate, predictions/probabilities for ALL complete windows of the held-out participant (the full label-independent candidate frame), with `binary_eligible`, `analysis_label` and raw-label provenance preserved as separate columns. Phase-3 metrics are computed only where the binary reference exists.
- **Study-B warning (recorded, not solved):** global LOPO predictions pooled from "other participants" are NOT a clean calibration/development set for one outer participant, because some of those predictions come from models whose training included that participant. Study B needs outer-fold-specific development predictions with proper independence.
- **Status:** agreed

### D-030 - Phase-2 closeout engineering fixes (independent review)
- **Date:** 2026-09-12
- **Category:** ENGINEERING
- **Fixes:** (1) `VerifiedRelease` can no longer be constructed by callers - only `VerifiedRelease.open()` (which loads the committed baseline and verifies the raw tree) can create one; stored hashes are private and immutable; every load still re-hashes the pickle. (2) Signal arrays must be real-valued integer/float dtypes (complex/bool rejected); label arrays must be (n,) or (n,1), never silently flattened. (3) Config contract enforced: `params_from_config` rejects any config that contradicts what Phase 2 implements (device, 60/60 s grid, anchor, label independence, eligibility rule, mixed-window policy, feature families, HRV off, counts_per_g 64, positive/negative codes 2/1, ineligible codes, preserve_raw_label); the binary mapping and counts-per-g are read from config and validated rather than hard-coded. (4) `q_acc_clipped` renamed `q_acc_near_rail` (near the +-2 g rail, not proven clipping; informational only); the constant-signal check is now per axis over time (a repeated [0,0,64] vector is constant). Schema version 1.1.0.
- **Canonical table impact:** rebuilt; every value in every shared column is identical to the 1.0.0 table and the renamed flag column is value-identical; zero constant-flagged windows before and after. The table SHA-256 changed ONLY because of the column rename: `f786c304...` -> `ebc0ccbde2062a77...`.
- **Status:** agreed

### D-031 - Phase 3 executed; prediction artifact recorded (freeze pending independent review)
- **Date:** 2026-09-13
- **Category:** ENGINEERING / RESULT RECORD
- **What ran:** `python -m wsr.experiments.baseline` exactly as frozen in D-028/D-029: LOPO outer (15), inner 4-fold StratifiedGroupKFold, predeclared grids (LR 6, RF 8, XGB 8), participant-level equal-weight balanced-accuracy selection, threshold 0.5, training-only median imputation (+ scaling for LR), complete-frame predictions. 1380 model fits ({'majority': 15, 'logistic': 375, 'random_forest': 495, 'xgboost': 495}). No all-missing-feature events, no fit warnings.
- **Artifact:** `results/phase3/oof_predictions.parquet`, 5768 rows (1442 per family x 4), SHA-256 `34010de95b3779b652fd5c9f7d372e69b3d7d61aac020912fd871d1298022212`; manifests `data/manifests/phase3_splits.json`, `phase3_run.json`, `phase3_predictions_manifest.json` (`frozen: false` until independent approval; Study A must verify the hash before use and must never regenerate the file after seeing selective-label results).
- **Reproducibility:** a second full run in the same environment produced a byte-identical parquet, identical CSVs and identical split manifests.
- **Implementation note:** sklearn 1.9.1 deprecates the explicit `penalty="l2"` kwarg; L2 is the default and is used (liblinear), recorded in `phase3_run.json: fixed_params.logistic.penalty`. xgboost 3.4.1 added to dependencies and lock file.
- **Not decided by these results:** no model family is dropped; all four remain Study-A material (D-029).
- **Status:** agreed (record); artifact freeze awaits review

### D-032 - Phase-3 prediction artifact FROZEN; primary full-reference ranking recorded
- **Date:** 2026-09-13
- **Category:** RESULT RECORD / INVARIANT
- **Independent review:** APPROVE PHASE 3. The reviewer reran all 1380 fits and reproduced the prediction parquet and result CSVs byte-for-byte.
- **Frozen artifact:** `results/phase3/oof_predictions.parquet`, SHA-256 `34010de95b3779b652fd5c9f7d372e69b3d7d61aac020912fd871d1298022212`, 5768 rows (1442 per family; 434 eligible + 1008 ineligible per family), commit lineage 3ec6a3e -> 282b8d2. `data/manifests/phase3_predictions_manifest.json` now carries `frozen: true`. Downstream code obtains the predictions only via `wsr.experiments.phase3_artifact.load_approved_predictions`, which verifies the hash first.
- **Invariant #21:** the frozen artifact must not be regenerated or replaced based on Study-A results. A genuine software defect -> log it, create a NEW versioned artifact with its own manifest, never overwrite this one.
- **Primary full-reference model-ranking metric for Study A (fixed now, before Study A):** equal-participant-weight mean balanced accuracy across the 15 LOPO outer participants. Ordering from the approved results: 1. XGBoost 0.8848; 2. Logistic Regression 0.8559; 3. Random Forest 0.8559; 4. Majority 0.5000. Logistic and RF are separated by ~2e-5 - essentially tied on this metric. The primary criterion is not to be changed later because another metric yields a more interesting reversal. Secondary metrics (macro-F1, AUROC, average precision, class recalls) stay secondary. No significance claims are made.
- **Approved interpretation:** XGBoost has the highest mean participant balanced accuracy; no learned family clearly dominates across participants and metrics; Logistic Regression has the highest mean AUROC/AP; participant heterogeneity is substantial (per-participant BA 0.50-1.00); these results concern baseline versus protocol stress under this offline WESAD design and do not establish psychological-stress specificity or clinical generalisation. No model family is removed; all four prediction streams are Study-A inputs.
- **Closeout fix (no result change):** hard-label metrics now always use the stored hard prediction (`pred_threshold_0_5`), so the majority baseline's declared tie rule (prevalence exactly 0.5 -> 0) governs every hard-label metric. The edge did not occur in the WESAD folds (prevalences 0.349-0.352); metrics recomputed from the frozen parquet with the fixed code are identical to the committed CSVs.
- **Status:** agreed (frozen)

### D-033 - Study A protocol FROZEN before any selective-label result (resolves T-05, T-06, T-07)
- **Date:** 2026-09-13
- **Category:** DESIGN CHOICE (pre-registration)
- **Full text:** `docs/study_a_protocol.md` (authoritative). Summary:
  - **T-05 candidate unit:** one complete non-overlapping 60-s Phase-2 time-grid window; candidate frame = ALL complete windows per participant (currently 1442, recomputed, never hard-coded); Phase-2 time-only boundaries; no eligibility prefilter, no block construction, no label use. Participant remains the inferential unit.
  - **T-06 budget:** one annotation unit per selected window, consumed regardless of the revealed label (out-of-target selections consume budget and never become baseline/stress). Per participant B_p(f) = floor(f x N_p) with primary f in {0.25, 0.50, 0.75} and 1.00 as negative control; identical B_p for targeted and random; 10 % only ever exploratory.
  - **T-07 selector:** one common targeted policy shared by all evaluated models: within-participant percentile rank (`rankdata`, average ties) of the frozen `prob_positive` for logistic, random_forest and xgboost, averaged equally into a consensus score; top B_p selected; ties broken by ascending SHA-256 of `"{tie_seed}:{participant_id}:{window_id}"` with `tie_seed = child_seed(42, "study_a_selector_tie")`. Majority excluded from the selector. Label-independent, deterministic.
  - **Random comparator:** uniform without replacement over all complete windows of the participant, exactly B_p, 500 repetitions, seed stream `study_a_random:<pid>:<fraction>:<k>`; same mask for all four families within a repetition.
  - **Availability rule:** BA estimable only with >= 1 selected baseline AND >= 1 selected stress window; otherwise NA (never 0.5 / majority / borrowed / pooled); `ba_estimable` recorded as an outcome.
  - **Primary distortion:** D(p, b, policy) = mean over the three learned families of |BA_observed - BA_full|; majority excluded. Contrast C(p, b) = D_targeted - mean random D over estimable repetitions.
  - **Model selection:** frozen full-reference criterion (D-032); >= 10 of 15 evaluable participants required, else NOT ESTIMABLE; highest mean observed BA over the same evaluable participants; tie (1e-12) order logistic > random_forest > xgboost.
  - **Regret:** full_reference_mean_BA(XGBoost) - full_reference_mean_BA(selected m), using frozen Phase-3 values; `top_model_changed` recorded; LR <-> RF swaps not exaggerated.
  - **Probabilistic endpoints:** Brier and log loss on selected eligible labels vs full reference, signed and absolute; never called "calibration distortion" alone; NA with zero eligible labels; no slope/intercept/ECE confirmatory endpoints.
  - **Secondary metrics:** macro-F1, class recalls, AUROC, AP under the same availability rule.
  - **Hypotheses:** H-A1 (BA distortion), H-A2 (estimability/class coverage), H-A3 (selection change and regret), H-A4 (Brier/log-loss distortion) - targeted vs matched random; null/reversed results valid.
  - **Negative control:** at 100 % both policies recover the full frame, all 434 eligible labels, exact full-reference BA, zero distortion, full-reference ranking; failure = STOP.
  - **Mask pairing:** one mask table keyed by participant/policy/budget/repetition joined to all families; enforced by structure and tests.
  - **Immutable input:** runner verifies `frozen: true` and the exact SHA-256 through `load_approved_predictions`; fails closed; never regenerates Phase 3 (invariant #21).
- **Still open at D-033 (resolved by D-034/SA-1 as T-14 - the item was mistakenly numbered T-11, which is the Nurse timestamp item):** the Study-A inferential procedure; NA reporting format; whether the exploratory 10 % run happens. Study B/C: T-08, T-09.
- **Status:** agreed (frozen; no mask built, no outcome inspected)

### D-034 - Study A pre-execution amendment SA-1 (independent review: APPROVE WITH REQUIRED AMENDMENTS)
- **Date:** 2026-09-14
- **Category:** DESIGN CHOICE (pre-registration amendment; no mask existed)
- **Unchanged (approved core):** full 60-s candidate frame; 25/50/75 % budgets; 100 % control; one-window annotation cost; out-of-target budget accounting; common LR/RF/XGB consensus selector; uniform same-budget random comparator; 500 repetitions; fixed Phase-3 predictions.
- **Frozen by SA-1 (full text `docs/study_a_protocol.md`):**
  - Budget hierarchy: 50 % PRIMARY confirmatory; 25/75 % supporting budget-response; 100 % negative control; no promotion of 25/75 % if 50 % is unavailable; no 10 % confirmatory.
  - Random replicates: dataset-level replicate M_{b,k} = union over the 15 participants of M_{p,b,k}, k = 0..499, same k across participants for H-A3; seed string `study_a_random:{participant_id}:{fraction}:{k}` with canonical fractions "0.25"/"0.50"/"0.75"/"1.00", UTF-8, via `child_seed(42, ...)`.
  - Selector terminology: "highest consensus stress rank", not "most confident of stress"; scope = offline, self-consistent model-triggered batch acquisition; not online triggering; not label leakage; results conditional on the mechanism; independent selector optional later. Tie key: SHA-256 of UTF-8 `f"{tie_seed}:{participant_id}:{window_id}"`, digest as unsigned big-endian integer, ascending wins.
  - H-A1: conditional estimand C_BA(p,b) = D_T - mean over estimable random repetitions of D_R, defined only if targeted BA estimable and >= 1 estimable random repetition; unavailable otherwise (no penalty/substitution/redraw).
  - H-A2: primary C_E(p,b) = mean_k E_R - E_T over all 500 repetitions, defined for all 15; supporting yield/coverage/class-balance quantities kept separate.
  - H-A3: dataset-level only (no participant NHST); >= 10/15 sufficiency; primary EMPIRICAL FULL-COHORT SELECTION REGRET R_cohort = max_m F_m - F_{m_hat}; required diagnostic SAME-EVALUABLE-COHORT regret R_subset with subset_full_best_model and subset_top_model_changed; random summaries over the 500 dataset-level replicates with stated denominators; unavailable selection -> regret NA.
  - H-A4: Brier distortion C_BS primary (conditional on >= 1 eligible label), log loss C_LL secondary, no composite; numerical convention: labels {0,1}, natural log, float64, `labels=[0,1]`, clipping at float64 machine epsilon identically for observed and full reference (sklearn 1.9.1 convention), Brier unclipped.
  - Metric availability table (BA/AUROC/macro-F1/AP need both classes; recalls need their class; Brier/log loss need >= 1 eligible label; zero eligible -> all NA; no library pseudo-values).
  - Confirmatory hierarchy: H-A1 C_BA at 50 % primary; H-A2 C_E at 50 % required companion; H-A3/H-A4 supporting at 50 %; 25/75 % budget-response; 100 % control; everything else secondary/descriptive.
  - T-14 inference plan: participant-level contrasts with mean/median/IQR/range/sign counts (numerical zero 1e-12)/n_evaluable/ids; MC SD and MCSE; binomial MCSE for probabilities; no p-value threshold as primary rule; no post-hoc test choice; finite-cohort interpretation.
  - Mandatory pre-output implementation invariants (14 items) and frozen tolerance (exact for counts/ids/model; 1e-12 for values); canonicalised selected ids.
  - Claim scope statement recorded before results.
- **Documentation synchronised:** PROJECT_CONTEXT Study-A summary, research_protocol hypotheses/metrics/falsification/confirmatory sections (amendment A-5), known_issues KI-05/KI-09/KI-19/KI-25, the duplicate T-11 renamed T-14, the "Section 12" cross-reference corrected to Section 18.
- **Status:** agreed (frozen; no mask, selector code or result exists)

---

## Open decisions (TODO before the affected stage)

| ID | Decision needed | Blocks | Where |
|---|---|---|---|
| T-01 | ~~WESAD device stream~~ RESOLVED (D-021): wrist E4 primary; chest optional sensitivity | feature extraction | `configs/wesad.yaml: device` |
| T-02 | ~~Raw label codes~~ RESOLVED as FACT (D-019); ~~binary mapping~~ RESOLVED (D-021): 1 = baseline reference, 2 = protocol-stress reference, {0,3,4,5,6,7} ineligible | labels | `configs/wesad.yaml: labels` |
| T-03 | ~~Usable participants~~ RESOLVED (D-019, D-021): all 15 kept; no caveat-based exclusion | splits | `docs/dataset_notes.md` |
| T-04 | ~~Outer/inner split~~ RESOLVED (D-028): LOPO outer (15 folds); inner 4-fold StratifiedGroupKFold on the 14 outer-training participants; participant-level equal-weight balanced-accuracy selection score | baseline | `configs/base.yaml: evaluation` |
| T-05 | ~~Selection unit~~ RESOLVED (D-033): one complete 60-s time-grid window; candidate frame = all complete windows; no label use | Study A | `study_a_protocol.md` S2 |
| T-06 | ~~Budget unit and grid~~ RESOLVED for Study A (D-033): one annotation unit per selected window incl. out-of-target; B_p = floor(f x N_p), f in {0.25, 0.50, 0.75} + 1.00 control; 500 random repetitions. Study C accounting remains open (T-09) | Study A | `study_a_protocol.md` S3-S4, S7 |
| T-07 | ~~Selector~~ RESOLVED (D-033): consensus percentile-rank of frozen LR/RF/XGB `prob_positive`, hash tie-break, majority excluded | Study A | `study_a_protocol.md` S5-S6 |
| T-08 | Study B specification: development roles (which participants play calibration/threshold/selection roles), detector access in development, coverage constraint or loss, tie handling, comparator; then the regret formula | Study B | `research_protocol.md` S6; KI-18 |
| T-09 | Study C specification: random sampling frame, overlap/budget accounting between targeted and random draws, 100 %-random same-budget comparator; then the estimator (naive pooled / audit-only / IPW) | Study C | `research_protocol.md` S7; KI-13 |
| T-10 | HRV feature gating threshold (beat coverage) | features | `configs/base.yaml: features.hrv` |
| T-11 | Nurse timestamp units / timezone / alignment (Gate 1) | Study D | `configs/nurse.yaml: timestamps` |
| T-12 | Nurse raw label values and binary mapping | Study D | `configs/nurse.yaml: labels` |
| T-13 | Coverage levels for tabular selective-prediction reporting | reporting | `configs/base.yaml: abstention` |
| T-14 | ~~Study-A inferential/reporting procedure~~ RESOLVED (D-034/SA-1): participant-level contrasts with mean/median/IQR/range/sign counts and n_evaluable for H-A1/H-A2/H-A4; Monte-Carlo SD/MCSE and binomial MCSE for random quantities; dataset-level descriptive/Monte-Carlo summaries for H-A3; no p-value threshold as the primary decision rule; no post-hoc test choice | Study A analysis | `study_a_protocol.md` S20 |
