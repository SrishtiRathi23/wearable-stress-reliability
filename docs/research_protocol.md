# Research Protocol (pre-registration-style DRAFT)

**Status:** DRAFT v0.1, 2026-09-12. Not yet frozen. Items marked **TODO** are unresolved decisions that must be settled and logged in `docs/decisions.md` before the affected analysis is run. Once frozen, changes to confirmatory analyses require a dated amendment entry at the bottom of this file.

**Category tags used throughout:** FACT (verified), DESIGN CHOICE (deliberate), ASSUMPTION (required, unverified), HYPOTHESIS (under test), EXPLORATORY (not confirmatory).

---

## 1. Research question

**Primary.** When a wearable stress classifier is evaluated with labels that were selectively observed (only some moments received a reference label), how much do its apparent accuracy, calibration, model ranking, and abstention-policy ranking differ from what the same frozen predictions would show under full reference labelling?

**Secondary.**
- (Q-B) Can selectively observed *development* labels cause a worse calibration/abstention policy to be selected than full development labels would?
- (Q-C) Does reserving a fraction of a fixed labelling budget for random audit reduce these distortions?
- (Q-D) Given the above, what can legitimately be claimed about the real Nurse Stress dataset, where most shift time carries no verified label?

## 2. Motivation

Wearable-stress evaluation typically reports accuracy/F1 against whatever labels exist. In real deployments (e.g. the Nurse Stress dataset), labels are obtained for detector-flagged or otherwise salient moments and retrospectively validated; the rest of the time is unlabelled. The selective-labels problem (Lakkaraju et al. 2017) predicts that metrics computed on such subsets need not reflect performance on the full population of moments. Whether this effect is material for wearable stress models, calibration, and abstention decisions - and whether it changes *which* model or policy would be chosen - has not, to our current knowledge, been examined directly. (Novelty claim pending literature review; see `docs/known_issues.md`.)

## 3. Datasets and roles

| Dataset | Role | Stage |
|---|---|---|
| WESAD (Schmidt et al. 2018) | Primary controlled benchmark; Studies A, B, C | First |
| Nurse Stress (Hosseini et al. 2022) | Bounded real-world case study; Study D | After WESAD pipeline works |
| Stress-Predict (Iqbal et al. 2022) | Replication of Study A (and B/C if time) under a different controlled protocol | Optional extension |

FACT-level details for WESAD (participants, signals, rates, label codes, alignment) were verified by the Phase-1 audit on 2026-09-12 and are recorded with provenance in `docs/dataset_notes.md`. Nurse and Stress-Predict details are not yet verified.

## 4. Primary hypotheses

All are HYPOTHESES. Null results are valid outcomes.

- **H-A1 (measurement distortion).** For fixed predictions, targeted observation policies (detector-triggered, duration-preferred) yield apparent balanced accuracy / macro-F1 that differ from the full-reference values by more than random observation at the same budget does.
- **H-A2 (calibration distortion).** Targeted observation policies distort apparent calibration (Brier score, reliability curve, slope/intercept) more than random observation at the same budget.
- **H-A3 (ranking distortion).** Under targeted observation, the rank order of candidate models and of abstention policies differs from the full-reference rank order in a non-trivial fraction of participants/budgets.
- **H-B1 (policy-selection distortion).** Policies selected on targeted-observation development labels have higher regret on the full-reference held-out set than policies selected on random-observation or full development labels.
- **H-C1 (mitigation).** Reserving a fraction of the labelling budget for random audit reduces measurement error (Study A) and regret (Study B) relative to fully targeted labelling at the same total budget.

Direction of effect is stated for clarity; the analyses are two-sided.

## 5. Study A - Measurement / evaluation distortion

**Design.** Participant-level outer holdout (Section 10). For each held-out participant, a model trained on the other participants produces frozen predictions (probabilities) for every window. The full protocol reference label is available for every window. Observation policies then produce a *mask* selecting which reference labels are visible to the evaluator. Metrics are computed on visible labels only and compared with the full-reference values.

**Invariants.** Same test recording; same trained model; same frozen predictions; only the mask changes. Test labels are never an input to any mask builder (enforced by `tests/test_label_masks.py` once implemented: permuting test labels must not change any mask).

**Observation policies (planned).**
1. `full_reference` - all windows labelled.
2. `random` - episodes sampled uniformly at random up to the budget.
3. `detector_triggered` - episodes ranked by a detector score; top-ranked labelled up to the budget. The detector is trained WITHOUT the held-out participant. **TODO:** define the detector (options: simple EDA-based score; a separate classifier trained on training participants; the evaluated model itself - each answers a different question and must be pre-declared).
4. `duration_preferred` - longer candidate episodes preferred. **TODO:** episode boundaries for this policy must come from the detector or from time structure, NOT from reference labels (otherwise this uses test labels).
5. `mixed` - targeted + random at a fixed ratio (feeds Study C).

**Candidate frame (FROZEN, amendment A-1 / D-021).** 60-s non-overlapping windows on a time-only grid anchored at synchronised pickle t=0 per participant; boundaries never depend on labels. Eligibility for the primary binary task is decided afterwards: a window is eligible only if it is entirely raw code 1 (baseline reference) or entirely raw code 2 (protocol-stress reference); mixed windows are kept in provenance and marked ineligible, never relabelled. Codes 0, 3-7 are ineligible. Primary device: wrist E4; chest is sensitivity only.

**Selection unit.** DESIGN CHOICE: selection operates on episodes/blocks, not on individual 60-s windows treated as independent. **TODO (open design issue, HIGH):** in WESAD, protocol blocks are few per participant (roughly one baseline block and one stress block per person). If "episode" = protocol block, targeted selection has almost nothing to choose between and Study A degenerates. Candidate resolutions: (a) define pseudo-episodes as fixed-length contiguous segments (e.g. 3-5 min) within the recording; (b) define episodes as detector-proposed contiguous runs; (c) window-level selection with contiguity constraints, reported as a diagnostic. Decide after the WESAD audit reports block durations.

**Budget.** **TODO:** budget unit (labelled episodes vs labelled duration vs labelled windows) and budget grid. Comparisons across policies must be at matched budget in the chosen unit; do not assume units are interchangeable.

**Outcomes (per held-out participant, per policy, per budget).** Balanced accuracy, macro-F1, per-class recall, Brier score, calibration slope/intercept (where estimable), reliability-curve distance, accepted error and coverage under each abstention policy, model rank order.

**Primary effect measure.** Difference between metric on visible labels and metric on full reference, summarised across participants (median and interquartile range; participant-level paired comparison vs `random` at the same budget). Windows are not treated as independent samples.

## 6. Study B - Policy-selection distortion

**Design.** Within the training participants, an inner grouped split yields development participants. Development labels are masked under each observation policy. Using only visible development labels: fit calibration (where the policy uses it) and choose thresholds. Lock the resulting policy. Evaluate on the held-out participant with full reference labels.

**Candidate policies.** always_predict; raw_confidence threshold; calibrated_confidence threshold; signal_quality withholding. **TODO:** selection criterion used to pick a policy in development (e.g. lowest accepted error at a target coverage; must be pre-declared).

**Policy-selection regret (definition to be finalised before implementation).** Provisional: for held-out participant *p* and observation policy *o*, let pi_o be the abstention/calibration policy chosen using labels visible under *o*, and pi_full the policy chosen using full development labels. Regret(p, o) = L(pi_o; full reference test of p) - L(pi_full; full reference test of p), where L is the pre-declared loss (**TODO:** e.g. accepted error at matched coverage, or a coverage-penalised risk). **TODO:** confirm whether the comparison is against pi_full or against the oracle-best policy on the test set (the latter is a diagnostic, not the confirmatory measure, because it uses test labels).

## 7. Study C - Mitigation by random audit

**Design.** Fixed total budget. Mixtures of targeted:random from `configs/base.yaml` (`observation_policies.audit_random_fractions`, provisional 0/10/20/30/50 % random). Re-run Study A and Study B outcomes under each mixture.

**Estimator (open design issue, HIGH). TODO:** pre-declare how visible labels are combined into a metric estimate. Options: (i) naive pooling of targeted + random labels; (ii) audit-only estimate (random subset alone); (iii) inverse-probability-weighted estimate using known selection probabilities. Under naive pooling the "mitigation" is expected to scale roughly linearly with the random fraction, which would be a near-trivial result; the scientifically interesting comparison is naive pooling vs an estimator that exploits the random component. At least (i) and (ii) should be reported; (iii) is EXPLORATORY unless pre-declared.

## 8. Study D - Nurse real-world case study

**Constraints.** Unlabelled periods are UNKNOWN. No metric that requires a label is computed on unlabelled periods. The WESAD simulation is not claimed to reproduce the Nurse label-collection process.

**Planned analyses (feasibility depends on audit).** Performance on validated events only; model output coverage over the whole shift; fraction of accepted predictions that fall in labelled periods; per-nurse variation; activity/movement strata; signal-quality strata; contextual/stressor strata where counts allow; sensitivity bounds for unknown periods (**TODO:** define the bounding assumptions explicitly, e.g. best-case/worst-case labelling of unknown windows, reported as bounds, never as point estimates).

**TODO (Gate 1):** timestamp units/timezone, survey-to-signal alignment, event boundary semantics, and the meaning of the raw label values must be verified before any Nurse modelling.

## 9. Primary metrics

Classification: balanced accuracy, macro-F1, per-class precision/recall, confusion matrix; AUROC / PR-AUC where appropriate.
Calibration: Brier score (primary), reliability diagram, calibration slope/intercept where estimable, log loss where appropriate, ECE descriptive only.
Selective prediction: accepted error, coverage, risk-coverage curve, per-class and per-participant coverage, number/fraction of participants (or events) with no accepted output. **TODO:** predefined coverage levels for tabular reporting.
Reporting unit: the participant. Windows are never treated as independent units in inferential statistics.

## 10. Split rules

- Outer: participant-level holdout. **TODO (T-04, still open after the audit):** leave-one-participant-out vs grouped k-fold with 15 participants; the config deliberately holds no default. Splits are written to `data/manifests/splits_<dataset>.json` and versioned.
- Inner: grouped (by participant) splits within the training participants for hyperparameters, calibration, and thresholds. **TODO:** number of inner folds / development participants.
- Random-row splits: diagnostic only, always labelled as such, never a headline result.

## 11. Leakage rules

1. Scalers, imputers, feature selection, hyperparameters, calibration, and all thresholds are fit on training/development participants only.
2. No test outcome is used to choose any threshold or policy.
3. No test label is used to construct any observation mask. Stronger (D-023): candidate construction and window boundaries for the primary Study-A frame are independent of held-out reference labels; selectors may use only explicitly permitted time/signal/model-derived inputs, never `raw_label`, `binary_analysis_label`, `eligibility` or protocol-state boundaries. Labels are used afterwards only for the evaluation target, scoring eligibility and full-reference metrics.
4. Any detector used for `detector_triggered` is trained without the held-out participant.
5. Transductive use of unlabelled test-participant data, if ever done, is labelled as such and reported separately.
6. Raw data is never modified (`wsr.utils.integrity`).

## 12. Unknown-label rules

- Nurse: unlabelled time is `unknown`, never negative. Config key `labels.unknown_is_negative` must remain `false`.
- WESAD: windows outside the pre-declared conditions (baseline, stress) are `excluded`, not relabelled. **TODO:** explicit list of excluded raw codes after audit.
- Label mapping tables live in `docs/decisions.md` and in config; they are not changed between experiments without an amendment entry.

## 13. Planned sensitivity analyses (EXPLORATORY unless promoted before freeze)

- Window length 30 / 60 / 120 s.
- Budget grid density.
- Detector choice for `detector_triggered`.
- Episode definition (see Study A TODO).
- Calibration method (sigmoid vs isotonic) if development data allows.
- Model family (LR / RF / XGBoost).
- Device stream in WESAD: chest as sensitivity to the wrist primary (D-021).

## 14. Failure gates

- Gate 1: timestamps/labels not justifiable -> stop modelling.
- Gate 2: participant-independent baseline nonsensical -> debug first.
- Gate 3: masks use test labels -> invalidate and rebuild.
- Gate 4: little label-selection effect -> do not manufacture; investigate, replicate, or report null.
- Gate 5: Nurse data too incomplete -> reduce the Nurse analysis.

## 15. What would falsify or weaken the hypotheses

- H-A1/H-A2: if, across participants and budgets, targeted policies produce metric deviations from full reference that are comparable to (or smaller than) random selection at matched budget, the distortion claim is not supported for this setting.
- H-A3: if model/policy rank order is preserved under targeted observation in the large majority of participant-budget cells.
- H-B1: if regret under targeted development labels is not larger than under random/full development labels.
- H-C1: if audit fractions do not reduce measurement error or regret, or if any reduction is fully explained by the random component alone (i.e. the targeted part adds nothing) - in which case the honest conclusion is "use random labelling", not "mix".
- General: effects present only for one detector definition, one budget, or one model would be reported as fragile.

## 16. Confirmatory vs exploratory

**Confirmatory (fixed at freeze):** H-A1, H-A2, H-A3, H-B1 on WESAD with the pre-declared model set, window length 60 s, pre-declared detector, budget unit, budget grid, and regret definition.

**Exploratory:** H-C1 estimator variants beyond the pre-declared one; all sensitivity analyses in Section 13; Stress-Predict replication; every Nurse analysis in Study D beyond descriptive reporting on validated events; activity-conditioned analyses; SHAP feature-reliance summaries.

Any analysis added or changed after results are seen is logged as post-hoc in `docs/experiment_log.md`.

---

## Amendments

- **A-1, 2026-09-12, Sections 5, 10, 11, 13.** Froze the Phase-2 preprocessing contract after the independent Phase-1 review (D-021, D-022, D-023): wrist E4 primary; time-only 60-s grid anchored at pkl t=0; homogeneous-code eligibility; binary reference 1 vs 2 with 0, 3-7 ineligible; all 15 participants kept; participant as inferential unit; HRV off. Reason: audit facts (dataset_notes.md) and review findings. Confirmatory analyses not yet run, so none affected. Still open: T-04 split, T-05 selection unit, T-06 budgets/repetitions/budget charging, T-07 detector, T-08, T-09, T-10.
