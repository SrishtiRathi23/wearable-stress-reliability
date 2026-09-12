# Known Issues and Risks

Living document. Each item has a status: `open`, `mitigated` (how), `accepted` (why), or `resolved` (link to decision/commit). Severity uses the same scale as the disagreement format: CRITICAL / HIGH / MEDIUM / LOW.

---

## Data and labels

### KI-01 Nurse timestamp / label alignment (Gate 1 blocker)
- **Severity:** CRITICAL for Study D
- **Issue:** The Nurse Stress dataset combines E4 signal files with survey/event records. Timestamp units, timezone, epoch conventions, and how event intervals map onto signal time have not been verified. Misalignment by even minutes would corrupt every Nurse label.
- **Plan:** `audit_nurse.py` must reconstruct alignment from official documentation and verify it empirically (e.g. event intervals fall within recording bounds; distributions look plausible). No Nurse modelling until this is documented in `dataset_notes.md`.
- **Status:** open

### KI-02 Selective hospital labels
- **Severity:** HIGH (it is also the phenomenon under study)
- **Issue:** Nurse labels exist only for detector-flagged, retrospectively validated events. Unlabelled shift time carries no information about stress status. Any metric computed over the whole shift as if unlabelled = non-stress is invalid.
- **Mitigation:** invariant #1; `labels.unknown_is_negative: false`; Study D restricted to validated events plus explicitly bounded sensitivity analyses.
- **Status:** mitigated by design; enforcement test pending Nurse loader

### KI-03 Small number of independent participants
- **Severity:** HIGH
- **Issue:** WESAD and the Nurse dataset each have on the order of 15 participants (to be verified). With participant-level holdout that is ~15 outer folds; inner development splits are even smaller. Statistical power for participant-level comparisons is low; confidence intervals will be wide.
- **Mitigation:** report participant-level medians/IQRs and paired comparisons; avoid window-pooled p-values; treat single-participant reversals honestly; consider Stress-Predict replication for robustness.
- **Status:** accepted (inherent to the data); reporting rules in protocol

### KI-04 Correlated windows
- **Severity:** HIGH
- **Issue:** Adjacent 60-s windows from the same participant are strongly autocorrelated. Window counts are not sample sizes. Calibration curves and ECE estimated from pooled windows overstate precision.
- **Mitigation:** participant is the unit of inference; window-level numbers are descriptive.
- **Status:** accepted; enforced by reporting rules

### KI-05 WESAD selection-unit degeneracy
- **Severity:** HIGH for Study A
- **Issue:** If episodes are defined as WESAD protocol blocks, there are only a handful per participant (roughly one baseline, one stress block of the conditions used). Targeted selection then has almost nothing to choose among and Study A cannot show a selection effect regardless of whether one exists.
- **Plan:** decide episode definition (T-05) after the audit reports block durations; options listed in `research_protocol.md` Section 5.
- **Status:** open

### KI-06 Duration-preferred policy could use test labels by accident
- **Severity:** HIGH (invariant #5)
- **Issue:** "Prefer longer events" is only valid if event boundaries come from a detector or time structure. If boundaries come from the protocol labels, the mask depends on test labels.
- **Plan:** mask builders receive only signals/detector outputs/time indices; the permutation test in `tests/test_label_masks.py` will catch violations.
- **Status:** open until implemented

## Modelling

### KI-07 Activity confounding
- **Severity:** MEDIUM
- **Issue:** Movement changes HR, EDA and temperature; a model may learn exertion rather than psychological stress. In WESAD the stress condition (TSST-type) involves standing/speaking; baseline is seated - activity is partially confounded with the label by protocol design.
- **Mitigation:** ACC-only vs physiology-only vs combined comparisons as a supporting control; report but do not claim causal separation.
- **Status:** open (supporting analysis planned)

### KI-08 HRV quality / window duration
- **Severity:** MEDIUM
- **Issue:** 60-s windows of wrist BVP give few beats and are motion-sensitive; many HRV statistics are unreliable at this duration. Blindly computing them adds noise and false precision.
- **Mitigation:** HRV disabled by default (`features.hrv.enabled: false`); enable only with a beat-coverage gate (T-10).
- **Status:** mitigated by default config

### KI-09 Calibration sample size
- **Severity:** MEDIUM
- **Issue:** Calibration must be fit on development participants only, leaving few windows and very few participants per calibration fit; isotonic regression will overfit; even Platt scaling will be noisy. Reliability diagrams with few bins will be unstable.
- **Mitigation:** sigmoid first; isotonic gated; Brier score as primary calibration metric; ECE descriptive only.
- **Status:** accepted with mitigation

## Scientific risk

### KI-10 Weak or null label-selection effect
- **Severity:** MEDIUM (scientifically acceptable; a risk to the narrative, not to validity)
- **Issue:** The effect of targeted observation on measured metrics may be small for this task, e.g. if the detector selects a nearly random subset, or if the model is near-perfect so subset choice barely matters.
- **Plan:** Gate 4 - do not manufacture an effect. Report the null with its uncertainty; investigate whether the detector/episode definition is too weak to select anything, but log such changes as exploratory.
- **Status:** open (hypothesis under test)

### KI-11 Simulated observation policy vs real Nurse collection
- **Severity:** MEDIUM
- **Issue:** The WESAD simulation is a controlled abstraction of plausible observation policies. The actual Nurse collection involved a specific detector, human validation, survey timing and compliance effects that we cannot reproduce.
- **Mitigation:** invariant #12; wording in all documents: "controlled simulation of plausible observation policies", never "reproduction".
- **Status:** accepted

### KI-12 Novelty not yet established
- **Severity:** MEDIUM
- **Issue:** The selective-labels problem (Lakkaraju 2017) and wearable abstention (ReliaGate 2026, Farahani 2026) exist. Whether the specific combination (selection-aware evaluation of calibration/abstention in wearable stress) is already published has not been checked systematically.
- **Plan:** literature search before the protocol is frozen; record findings in `dataset_notes.md` or a `literature.md`.
- **Status:** open

### KI-13 Study C could produce a trivial result
- **Severity:** MEDIUM
- **Issue:** If visible labels are naively pooled, adding random labels reduces bias approximately in proportion to the random fraction - an arithmetic consequence, not a finding. The interesting question is whether an estimator that uses the random component (audit-only or IPW) recovers full-reference metrics better than pooling.
- **Plan:** pre-declare estimators (T-09).
- **Status:** open

### KI-14 Timeline
- **Severity:** MEDIUM
- **Issue:** Four studies plus two datasets in ~4 weeks is ambitious. Dataset audits (especially Nurse alignment) can consume a large fraction of the time.
- **Plan:** Study A on WESAD is the minimum deliverable; B and C next; Nurse case study reduced to descriptive analysis if time is short (Gate 5).
- **Status:** accepted

## Engineering

### KI-15 Python 3.11 unavailable locally
- **Severity:** LOW
- **Issue:** Brief preferred 3.11; laptop has 3.12/3.13/3.14. Using 3.12 (D-013). Should be harmless; recorded for reproducibility.
- **Status:** resolved (D-013)

### KI-16 Windows path / newline conventions
- **Severity:** LOW
- **Issue:** Development is on Windows; collaborators or reviewers may be on Linux. Path handling goes through `pathlib` and `wsr.utils.paths`; text files are written with explicit UTF-8.
- **Status:** mitigated
