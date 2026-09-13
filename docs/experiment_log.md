# Experiment Log

Every run that produces a number that might be reported gets an entry. Entries are append-only. If a run is later found to be invalid (e.g. a leakage bug), add a follow-up entry marking it `INVALIDATED` with the reason; do not edit or delete the original.

Runners under `src/wsr/experiments/` should write a machine-readable copy of the same fields to `results/logs/<run_id>.json`; this file is the human-readable narrative.

---

## Entry template

```
### <run_id>  (e.g. 2026-09-20_wesad_baseline_lr_001)

- Date/time (local):        YYYY-MM-DD HH:MM
- Git commit:               <full sha>  (dirty tree? yes/no - if yes, say what was uncommitted)
- Dataset:                  wesad | nurse | stress_predict
- Participants used:        <list or manifest reference>
- Exclusions:               <ids + reason, or "none">
- Split manifest:           data/manifests/<file>
- Config:                   configs/<file>.yaml  (+ overrides, if any, verbatim)
- Config snapshot:          results/logs/<run_id>_config.yaml
- Random seed:              <int>  (streams used: split, model, mask, ...)
- Window length / step:     <s> / <s>
- Feature set:              <families>  (feature table: data/processed/<file>)
- Model(s):                 <family + hyperparameters or grid>
- Calibration:              none | sigmoid | isotonic  (fit on: <inner dev participants>)
- Abstention policy:        <name + threshold + how threshold was chosen>
- Observation policy:       full_reference | random | detector_triggered | duration_preferred | mixed
    - budget:               <value + unit>
    - audit random fraction:<value>
    - detector:             <definition + training participants>
    - mask file:            results/predictions/<run_id>_masks.parquet
- Confirmatory or exploratory:  confirmatory | exploratory | post-hoc
- Metrics (participant-level summary; never window-pooled only):
    - balanced accuracy:    median [IQR]
    - macro-F1:             median [IQR]
    - Brier:                median [IQR]
    - accepted error / coverage:
    - <others>
- Result artifacts:
    - predictions:          results/predictions/<file>
    - tables:               results/tables/<file>
    - figures:              results/figures/<file>
    - models:               results/models/<file>
    - log:                  results/logs/<file>
- Runtime / hardware:       <minutes, CPU/GPU>
- Notes:                    <what was being tested and why>
- Anomalies:                <warnings, failed participants, NaNs, convergence issues, anything odd>
- Interpretation:           <one or two sentences; state uncertainty; do not over-claim>
- Follow-ups:               <next steps or questions raised>
```

---

## Entries

### 2026-09-12_bootstrap

- Date/time: 2026-09-12
- Git commit: (initial commit)
- Dataset: none
- Notes: Repository bootstrap only. No data inspected, no models trained, no results produced.
- Interpretation: n/a

### 2026-09-12_wesad_structural_audit

- Date/time: 2026-09-12
- Git commit: (see commit "WESAD Phase 1: structural audit")
- Dataset: wesad (archive sha256 5e15d260...38fd71c; baseline raw_checksums_wesad.json verified unchanged before the audit)
- Participants used: S2-S11, S13-S17 (15); exclusions: none
- Config: none (audit is config-free); command: `python -m wsr.data.audit_wesad`
- Random seed: n/a (deterministic)
- Model / calibration / abstention / observation policy: none (structural audit only, no predictive number produced)
- Confirmatory or exploratory: n/a
- Result artifacts: data/manifests/wesad_audit.json (tracked); results/tables/wesad_{participant,signal,label,blocks}_audit.csv (regenerable, not tracked)
- Runtime: ~82 s CPU
- Notes: verified participants, files, signals, rates, label codes, contiguous runs, pkl-vs-raw crop for both devices, quest.csv alignment (10-s trim), device clock consistency.
- Anomalies: none structural. Documentation discrepancies listed in docs/dataset_notes.md.
- Interpretation: all 15 participants structurally usable; protocol-block selection unit confirmed degenerate (KI-05).

### 2026-09-12_wesad_phase2_feature_table

- Date/time: 2026-09-12
- Git commit: (see Phase-2 commit; manifest records the build commit and dirty flag)
- Dataset: wesad (raw baseline verified by the loader before every pickle; raw manifest sha256 438a1bb3e38b23d1...)
- Participants used: all 15; exclusions: none
- Config: configs/base.yaml (windowing, features, quality) + configs/wesad.yaml (device wrist, labels); command: `python -m wsr.features.build_features`
- Random seed: n/a (no stochastic processing)
- Model / calibration / abstention / observation policy: none - no predictive number produced
- Confirmatory or exploratory: n/a (data preparation)
- Result artifacts: data/processed/wesad_windows_60s.parquet (sha256 f786c304c5884a0de157031a95817a448809fa27a38297fded6b6d4693afcb0b, 1442 rows x 117 cols, not committed); data/manifests/wesad_feature_schema.json; data/manifests/wesad_windows_60s_manifest.json; results/tables/wesad_windows_60s_summary.csv
- Runtime: ~60 s CPU (incl. 19 GB raw-tree verification)
- Notes: time-only 60-s grid at t=0; eligibility after the grid; 434 eligible (282 + 152) recomputed, matching the review expectation.
- Anomalies: in-house pulse detector not beat-accurate (KI-21) -> HR columns provisional; ACC clipping flag 12.3 % (KI-22).
- Interpretation: feature table is model-ready for participant-independent pipelines; no outcome-driven engineering was performed.

### 2026-09-12_wesad_phase2_closeout_rebuild

- Date/time: 2026-09-12
- Git commit: (Phase-2 closeout commit)
- Dataset: wesad; raw baseline verified unchanged
- Participants used: all 15; exclusions: none
- Notes: canonical table rebuilt under schema 1.1.0 after the closeout fixes (q_acc_clipped -> q_acc_near_rail; per-axis constant check). Before/after comparison: all shared columns value-identical, renamed flag value-identical, 0 constant-flagged windows both times. Counts unchanged (1442 / 282 / 152 / 785 / 223).
- Result artifacts: data/processed/wesad_windows_60s.parquet sha256 ebc0ccbde2062a7727dcd7f2d5f3ad3588dfe32639368e4fc08ec39bdb084a32; manifests regenerated.
- Interpretation: no scientific change; hash change is purely the column rename.

### 2026-09-13_wesad_phase3_baselines

- Date/time: 2026-09-13
- Git commit: (Phase-3 commit; manifests record the build commit)
- Dataset: wesad; feature table sha256 ebc0ccbde2062a77... (verified against the Phase-2 manifest before loading); schema 1.1.0
- Participants used: all 15 (LOPO); supervised rows: 434 binary-eligible (282 baseline / 152 protocol stress); exclusions: none
- Split manifest: data/manifests/phase3_splits.json (15 outer LOPO folds; 15 inner StratifiedGroupKFold(4) splits, seed stream inner_split:<pid>)
- Config: configs/base.yaml evaluation + phase3; seed 42; model seed stream model_fit:<family>:<pid>
- Models: majority, logistic (6 candidates), random_forest (8), xgboost (8); fits {'majority': 15, 'logistic': 375, 'random_forest': 495, 'xgboost': 495} = 1380
- Calibration / abstention / observation policy: none
- Confirmatory or exploratory: baseline (pre-Study-A); comparative, all families retained
- Metrics (participant-level, 15 outer participants; SD across participants, not a window SE):

| model | BA mean | BA median | BA SD | BA min-max | macro-F1 | stress recall | baseline recall | AUROC | AP |
|---|---|---|---|---|---|---|---|---|---|
| xgboost | 0.885 | 0.944 | 0.146 | 0.500-1.000 | 0.873 | 0.848 | 0.922 | 0.961 | 0.960 |
| logistic | 0.856 | 0.894 | 0.133 | 0.579-1.000 | 0.845 | 0.849 | 0.863 | 0.970 | 0.971 |
| random_forest | 0.856 | 0.909 | 0.146 | 0.500-1.000 | 0.839 | 0.837 | 0.875 | 0.955 | 0.958 |
| majority | 0.500 | 0.500 | 0.000 | 0.500-0.500 | 0.394 | 0.000 | 1.000 | 0.500 | 0.350 |

- Result artifacts: results/phase3/oof_predictions.parquet (sha256 34010de95b3779b652fd5c9f7d372e69b3d7d61aac020912fd871d1298022212, 5768 rows), per_participant_metrics.csv, model_summary.csv, selected_hyperparameters.csv, inner_candidate_scores.csv, inner_candidate_participant_scores.csv; manifests phase3_run.json, phase3_splits.json, phase3_predictions_manifest.json
- Runtime: ~4 min 47 s CPU (single-threaded fits)
- Anomalies: none in execution (no all-missing events, no warnings). Scientific: S7 and S14 show participant-level probability offsets (KI-24); per-participant BA ranges 0.50-1.00.
- Interpretation: wrist features separate baseline from protocol stress for unseen participants well above the majority baseline (mean BA 0.86-0.89), with large between-participant variability; XGBoost is highest on mean/median BA but LR has the highest mean AUROC/AP and the narrowest spread, so no family clearly dominates across metrics. All families are retained for Study A. Reproduced byte-identically on rerun.
