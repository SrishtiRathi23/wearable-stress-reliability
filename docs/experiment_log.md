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
