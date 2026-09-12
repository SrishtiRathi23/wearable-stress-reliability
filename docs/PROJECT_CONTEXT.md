# PROJECT CONTEXT

**Purpose of this file.** This is the durable statement of what the project is, why it exists, and what constraints govern it. It exists so that any engineer or coding agent can understand the project from the repository alone, without chat history. It is the authoritative specification unless `docs/decisions.md` records a later change.

Last updated: 2026-09-12 (repository bootstrap).

---

## 1. Identity

**Working title:** Label-Selection-Aware Evaluation of Calibration and Abstention in Wearable Stress Prediction

**Possible subtitle:** Controlled Label-Observation Experiments and a Nurse Stress Case Study

**Older umbrella title (historical documents only):** "From Laboratory Stress to Clinical Reality"

**Type:** Undergraduate minor project, Electrical & Electronics Engineering. Intended outputs: a defensible minor-project report and potentially a research manuscript.

**Domain:** wearable biomedical signals, physiological signal processing, machine learning, uncertainty/calibration, selective prediction/abstention, evaluation methodology, real-world nurse stress data.

**Ethics scope:** no new human participants. Public, anonymised secondary datasets only.

**Priorities, in order:** scientific validity > reproducibility > leakage prevention > traceability of decisions > honest interpretation > high accuracy numbers.

---

## 2. The research idea in plain language

A conventional wearable-stress paper does: signals -> model -> predict stress/non-stress -> compare with labels -> report accuracy/F1.

This project asks: **what if the labels used to judge the model are themselves selectively observed?**

Illustration: a model predicts 1,000 moments. If all 1,000 have reference labels, measured performance might be 80%. If only 200 "interesting-looking" moments receive labels, the *same* model with the *same* predictions might appear to score 90%+. Nothing about the model changed; only which outcomes were observed changed.

**Central question:** When a wearable stress model appears accurate and confident, is that reliability genuine, or partly an artifact of which moments received labels?

**Sub-questions:**

1. Does selective label observation distort measured accuracy / F1?
2. Does it distort calibration / confidence?
3. Does it make one model look better than another?
4. Does it make one abstention/confidence policy appear safer than another?
5. Can selective *development* labels cause us to choose the wrong operating policy?
6. Can adding a small random audit component to targeted labelling reduce this distortion?
7. What can legitimately be concluded from the real Nurse Stress dataset, where many hospital periods have no verified label?

---

## 3. What the project is NOT claiming

None of the following are our novelty: wearable stress detection; Random Forest / XGBoost; cross-dataset transfer; activity-aware stress detection; calibration; abstention; SHAP; label-selection bias in ML in general. All have prior literature.

**Candidate contribution:** a focused experimental study of how selective label observation affects apparent reliability, calibration, model/policy ranking and abstention conclusions in wearable-stress evaluation, plus a carefully bounded Nurse case study.

**Novelty is not guaranteed.** Do not write "first-ever", "novel", or similar as fact unless later supported by a proper literature review and experimental evidence.

---

## 4. History: why the direction changed

- First idea: "use wearable signals to detect nurse stress" - heavily researched already.
- Second proposal: cross-dataset transfer (WESAD -> Nurse, Stress-Predict -> Nurse), activity confounding, domain adaptation, calibration, abstention, SHAP, NLP, stressor/cause analysis - too broad and overlapping with existing work.
- Current direction: label-selection-aware reliability experiments as the core.

Role of previously planned components now:

| Component | Current role |
|---|---|
| Cross-context transfer | supporting baseline |
| Activity analysis | supporting control |
| Calibration / abstention | important evaluation targets |
| SHAP / NLP | optional supporting analysis |
| Label-selection-aware reliability experiments | **core** |

### Related literature already identified (to be verified and expanded in the literature review)

- Hosseini et al. 2022 - Nurse Stress dataset / hospital wearable recordings.
- Schmidt et al. 2018 - WESAD.
- Iqbal et al. 2022 - Stress-Predict.
- Mihirette et al. 2025 - cross-contextual stress prediction and domain adaptation.
- Kwon et al. 2026 - cross-corpus wearable stress evaluation involving healthcare workflows; close overlap with the older transfer/activity proposal.
- ReliaGate 2026 - wearable-stress reliability routing / withholding / abstention.
- Farahani et al. 2026 - classify/defer/abstain based on structural physiological ambiguity.
- Lakkaraju et al. 2017 - the selective-labels problem in ML generally.
- Ovadia et al. 2019 - predictive uncertainty under dataset shift.

---

## 5. Datasets and roles

### WESAD (Schmidt et al. 2018)
- **Role:** primary controlled benchmark and the FIRST dataset for the controlled label-observation experiment.
- **Why:** protocol-defined states cover a large part of each recording, so labels can be hidden in simulation while a fuller reference remains available for evaluation.
- **Initial task:** baseline vs protocol stress. Other states excluded or handled only under a pre-declared mapping.
- **Caveat:** protocol stress is a reference condition, not perfect psychological ground truth.

### Nurse Stress dataset (Hosseini et al. 2022)
- **Role:** primary real-world hospital case study, bounded.
- **Data:** Empatica E4-type wrist sensing of nurses during hospital shifts; signals may include EDA, BVP, HR, IBI (where usable), skin temperature, accelerometer, survey/event context.
- **Critical limitation:** no verified ground truth for every minute of a shift. Candidate stress events were detector-assisted and later retrospectively validated.
- **Invariant:** UNLABELLED HOSPITAL PERIOD != VERIFIED NON-STRESS. Unknown remains unknown. Never silently convert unlabelled time into "no stress".
- It is not proof of whole-shift accuracy and not clinical validation.

### Stress-Predict (Iqbal et al. 2022)
- **Role:** later replication/extension dataset, used only after the WESAD central experiment works.
- **Correction:** it is NOT a natural daily-life dataset; it contains controlled tasks/protocols (e.g. Stroop, TSST, hyperventilation, rest).
- Should replicate the central observation-policy experiment under a different controlled protocol.

### Optional future datasets
Not part of the minimum minor project. Do not add datasets merely because they exist.

---

## 6. Core experimental design (four studies)

Details, hypotheses and open TODOs live in `docs/research_protocol.md`. Summary:

**Study A - Measurement / evaluation distortion.** Fix the held-out recording, the trained model and its predictions. Vary only the label-observation mask (full reference; random; detector-triggered; duration-preferred; mixed targeted + random). Measure how much apparent balanced accuracy, macro-F1, calibration, accepted error, model ranking and participant-level performance change. Test labels must never be used to build the mask. Any detector proposing events must be trained without the held-out participant. Selection should operate at event/episode/block level rather than treating neighbouring 60-s windows as independent events. Annotation budgets must be compared carefully (episodes vs duration vs windows are not interchangeable).

**Study B - Policy-selection distortion.** Candidate policies: always predict; raw-confidence threshold; calibrated-confidence threshold; signal-quality withholding. Mask *development* labels under different observation policies; fit calibration and choose thresholds using only visible development labels; LOCK the policy; evaluate on the untouched full-reference held-out set. Question: can selective development labels make a policy look best in development yet perform worse on the full reference? Candidate concept: "policy-selection regret" (formula must be documented before implementation).

**Study C - Mitigation / random audit.** Fixed labelling budget; compare mixtures such as 100/0, 90/10, 80/20, 70/30, 50/50 targeted/random (provisional, config-driven). Hypothesis (not assumed result): reserving part of the budget for random auditing improves reliability estimation or reduces wrong policy selection.

**Study D - Nurse real-world case study.** Performance on validated events; model output coverage; fraction of accepted predictions that are labelled; per-nurse variation; activity/movement strata; signal-quality strata; contextual/stressor information where counts allow; sensitivity bounds for unknown periods where justified. Never invent truth for unlabelled periods. Never claim the WESAD simulation reproduces the Nurse collection process.

---

## 7. Modelling and processing constraints

**Models:** Logistic Regression, Random Forest, XGBoost; plus majority/prevalence baseline and possibly a simple EDA-based score. No deep learning (CNN/LSTM/Transformer) unless a later, scientifically motivated question requires it.

**Windows:** 60-s non-overlapping as the starting design choice (not a claim of optimality). Sensitivity later: 30/60/120 s, only after the central pipeline works.

**Feature families:** EDA (mean, variance/std, slope, tonic, phasic/SCR summaries, peak count/amplitude where valid); cardiac (reliable pulse/HR summaries, variability where valid); IBI/HRV only if beat quality and window duration support it; temperature (mean, variability, slope); ACC (magnitude, variability, movement intensity, jerk/burst where justified); signal quality (valid fraction, missingness, beat coverage, non-wear indicators). Do NOT upsample all signals to one rate just to align arrays; use modality-appropriate processing and common window-level features.

**Evaluation (non-negotiable):** participant-level outer holdout; grouped development splits within training participants. The held-out participant must not influence imputation, scaling, feature selection, hyperparameters, calibration, confidence or abstention thresholds. Any use of unlabelled test-participant data must be labelled transductive and never confused with cold-start evaluation. Random-row splits only as a clearly labelled diagnostic.

**Calibration:** sigmoid/Platt first; isotonic only with enough independent development evidence; trained on development data only. Evidence: Brier score, reliability diagrams, calibration slope/intercept where appropriate, log loss where appropriate, ECE as descriptive only (bin-sensitive in small samples).

**Abstention:** always predict; raw confidence threshold; calibrated confidence threshold; signal-quality withholding; one stronger published comparator later if reproducible. Always report error AND coverage: accepted error, coverage, risk-coverage curves, per-class coverage, per-participant coverage, number/fraction of participants or events with no accepted output.

**Metrics:** macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, AUROC/PR-AUC where appropriate; calibration metrics above; selective-prediction metrics above; participant-level summaries, participant-clustered uncertainty, paired participant-level comparisons. Thousands of windows are not thousands of independent participants.

**Activity:** supporting control only (ACC-only, physiology-only, combined, activity-conditioned). Activity matching does not prove causality.

**NLP / stressor causes / SHAP:** not core. NLP only to organise survey text/categories. Never use text that states the cause as input to predict that cause. Exact stress-cause prediction from physiology is removed from the core. SHAP, if used, answers "what did the model rely on", never "what biologically caused stress".

---

## 8. Research invariants (failure modes to actively prevent)

1. NEVER convert unknown Nurse labels into confirmed non-stress.
2. NEVER let the held-out participant influence preprocessing/model/calibration decisions.
3. NEVER fit scalers/imputers on the full dataset before splitting.
4. NEVER use test outcomes to choose thresholds.
5. NEVER use test labels to construct observation masks.
6. NEVER count windows as independent participants.
7. NEVER silently change label definitions between experiments.
8. NEVER change raw files.
9. NEVER drop problematic sessions/participants without recording why.
10. NEVER let event IDs / participant IDs / label provenance disappear during preprocessing.
11. NEVER use a random train/test row split as the main result.
12. NEVER claim simulated selection exactly reproduces the Nurse collection procedure.
13. NEVER add a model/dataset just because it might improve the paper aesthetically.
14. NEVER present high accuracy as proof of clinical usefulness.
15. NEVER claim clinical diagnosis, deployment safety, burnout diagnosis or medical validation.
16. NEVER claim novelty automatically.
17. NEVER manipulate the analysis because the expected effect did not appear.
18. NULL RESULTS ARE VALID.

Where an invariant is mechanically checkable, a test under `tests/` enforces it (see `tests/` for which are implemented and which are still explicit skips).

---

## 9. Philosophy

One strong research question + clean experiments + clear limitations + reproducible code. NOT: 15 models + 10 datasets + hundreds of experiments + no coherent story. Improvement comes from a sharper experiment, not a longer algorithm list.

Before adding any method, ask: what additional research question does it answer? "It may improve accuracy" is usually not enough.

---

## 10. Reproducibility requirements

Git from day one; recorded Python/package versions (`requirements-lock.txt`); explicit seeds (`wsr.utils.seeds`); configuration files over hard-coded parameters (`configs/`); preserved raw inputs (`wsr.utils.integrity` checksums); logged exclusions and participant splits (`data/manifests/`); saved prediction files used by final figures (`results/predictions/`); no hidden notebook-only logic (reusable logic under `src/wsr/`); tests for critical invariants; repeatable experiment commands. A final result must be recreatable from: configuration + code commit + dataset + seed/split manifest.

---

## 11. Environment

Dev laptop: HP Omen 16, AMD Ryzen 7 7000-series, NVIDIA RTX 4060. GPU is NOT required; the minimum project stays CPU-runnable. Python: the brief preferred 3.11; the laptop has 3.12/3.13/3.14, and 3.12 is used (see `docs/decisions.md` D-013). Package stack kept minimal; see `pyproject.toml`.

---

## 12. Timeline and gates

Minimum one-month study. Weeks 1-2: freeze protocol, audit datasets, validate labels/timestamps, preprocessing/features, basic baselines. Weeks 3-4: minimum label-selection replay, calibration/abstention evaluation, mitigation pilot, Nurse case study where feasible, robustness checks, report. Optional 4-8 week extension: Stress-Predict replication, stronger comparator, more policies, stronger statistics, manuscript. The deadline must not depend on optional extensions.

Failure gates:
- Gate 1: timestamps/labels cannot be justified -> STOP modelling, resolve.
- Gate 2: participant-independent baseline nonsensical -> debug before label-selection experiment.
- Gate 3: masks accidentally use test labels -> invalidate and rebuild.
- Gate 4: first label-selection pilot shows little difference -> do not manufacture an effect; investigate, replicate, or accept a negative result.
- Gate 5: Nurse data too incomplete -> reduce the Nurse analysis rather than invent labels.

---

## 13. Working style for engineers and coding agents

- Write code another engineer can understand; document non-obvious decisions in `docs/decisions.md`.
- Keep project state in the repository, not in chat memory.
- Summarise changes after every task; list assumptions and unresolved questions; run tests before claiming completion.
- Do not make important scientific decisions silently. If ambiguity could materially affect the result (label mapping, unknown-label treatment, split definition, timestamp interpretation, event boundaries, imputation, detector construction, calibration split, observation-budget definition), STOP and ask.
- Disagree explicitly using: ISSUE / WHY IT MATTERS / SEVERITY (CRITICAL, HIGH, MEDIUM, LOW) / RECOMMENDATION / STATUS (stop, proceed with warning, optional).
- Verify assumptions against the data, official documentation, source code, and the cited papers. When evidence disagrees with the plan, report the discrepancy; do not force the data to fit.
- Keep the categories separate in all documentation: **FACT** (verified), **DESIGN CHOICE** (deliberate), **ASSUMPTION** (required, unverified), **HYPOTHESIS** (under test), **EXPLORATORY IDEA** (not part of the confirmatory study).
- Protect against our own bias: do not hunt seeds, drop hard participants, move thresholds after seeing test results, redefine outcomes, or cherry-pick metrics/datasets/budgets. Post-hoc changes are logged as exploratory.
- Agreement is not a success criterion; a scientifically defensible project is.
