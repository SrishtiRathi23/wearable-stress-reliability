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

### KI-05 WESAD selection-unit degeneracy - CONFIRMED by audit
- **Severity:** HIGH for Study A
- **Issue:** If episodes are defined as WESAD protocol blocks, block-level selection is a poor model of annotation: only two relevant blocks per person; budget resolution is extremely coarse (0, 1 or 2 blocks); selection is confounded with condition (choosing a block chooses a class); many partial-observation scenarios are single-class, so most metrics are undefined or degenerate; and it is a poor analogue of realistic annotation budgets. A selection effect could still exist at block level, but it would be uninformative about the question the study asks.
- **Evidence (audit 2026-09-12, `data/manifests/wesad_audit.json`):** every participant has exactly **one** contiguous baseline run (1140-1198 s, 19 x 60-s windows) and exactly **one** contiguous stress run (615-725 s, 10-12 windows). Block-level selection = choosing among 2 blocks per person, 30 across the dataset. Window-level material under the approved time-only 60-s grid anchored at pkl t=0: 434 label-homogeneous windows (282 baseline, 152 stress) plus 60 boundary-straddling ineligible windows; restarting windows at run boundaries would give 445 but uses label timing and is not the primary frame (see dataset_notes.md).
- **Plan:** decide T-05 among (a) fixed-length contiguous pseudo-episodes inside each block, (b) detector-proposed episodes, (c) window-level selection with contiguity constraints as a diagnostic. Whatever is chosen, the number of selectable units per participant is small (e.g. 3-min pseudo-episodes give ~6 baseline + ~3-4 stress per person), so budget grids must be coarse and results reported per participant.
- **Status:** open (design decision pending)

### KI-06 Candidate generation and masks could use test labels by accident
- **Severity:** HIGH (invariant #5)
- **Issue:** "Prefer longer events" is only valid if event boundaries come from a detector or time structure. If boundaries come from the protocol labels, the mask depends on test labels. Independent review (2026-09-12) sharpened this: **candidate generation itself must be label-independent**. Keeping `y_test` out of the final mask function is insufficient if the candidate episodes, their boundaries, durations, or detector scores were computed upstream with any access to held-out labels.
- **Plan:** the whole candidate-construction pipeline (episode boundaries, detector scores, durations, ranking) receives only signals/features/time indices from the held-out participant and models trained without that participant. The future leakage test in `tests/test_label_masks.py` must **rebuild candidate construction from the beginning** under permuted held-out labels and assert identical candidates and masks; permuting labels only after candidate metadata exists would not detect upstream leakage.
- **Status:** open until implemented

### KI-19 WESAD code 0 (transient) is ~45 % of the recording and is not "non-stress"
- **Severity:** MEDIUM
- **Issue:** ~40 min per participant carries label 0 ("not defined / transient"), plus ~2.7 min of reading blocks (5/6/7). These periods have no verified affective state. Treating them as baseline/negative would be the WESAD analogue of the Nurse invariant violation.
- **Plan:** exclude from the binary task (proposed mapping); if ever used (e.g. as an "unlabelled pool" for a Study A variant), label them `unknown`, never negative, and pre-declare the analysis.
- **Status:** open (mapping approval pending)

### KI-20 Interpretation caveats on specific WESAD participants
- **Severity:** LOW-MEDIUM
- **Issue:** Readme notes: S6 and S15 report the TSST did not stress them / they did not believe the cover story; S2 and S17 have a loose chest temperature sensor; S5 may have slept in meditation; S3 baseline in a sunny workplace; S8/S16 felt cold during stress. None are structural defects and none justify exclusion by themselves.
- **Plan:** keep all 15 (no exclusion). If a per-participant analysis later shows these subjects as outliers, that is reported, not removed. Any chest-temperature feature must carry the S2/S17 quality flag.
- **Status:** accepted

### KI-21 In-house BVP pulse detector is not beat-accurate on wrist PPG
- **Severity:** MEDIUM (feature quality; no effect on the candidate frame or labels)
- **Issue:** The Phase-2 detector (band-pass + `find_peaks` with distance/prominence floors) produces plausible median HR (~82 bpm) but implausible within-window variability: median `hr_std` ~27 bpm, `hr_max` at the 0.33-s floor (174.5 bpm) in many windows, and 30-45 % of successive beat-to-beat HR changes > 20 bpm on S2/S10 under three parameter settings. Wrist PPG with motion is known to be hard; a validated algorithm or an IBI-consistency gate is needed.
- **Mitigation:** beat/HR columns are `feature_provisional` (model_feature false) in the schema (D-024). Raw BVP statistics remain features. Options for later (all label-free): stricter IBI-consistency gating per window, a validated PPG detector (e.g. from NeuroKit2, with version pinned), or dropping HR entirely. Any change is a schema-version bump and a decision entry.
- **Status:** open

### KI-22 ACC clipping flag fires on 12.3 % of windows
- **Severity:** LOW (informational)
- **Issue:** `q_acc_clipped` (any |count| >= 127, i.e. at the +-2 g rail) is True for 12.3 % of the 1442 windows. This is expected during vigorous movement and is not a defect; it is recorded so that activity-related analyses can stratify or check sensitivity to clipped windows. The 127-count threshold is provisional.
- **Status:** accepted (flag only)

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

### KI-13 Study C is under-specified and could produce a trivial result
- **Severity:** MEDIUM
- **Issue:** If visible labels are naively pooled, adding random labels reduces bias approximately in proportion to the random fraction - an arithmetic consequence, not a finding. The interesting question is whether an estimator that uses the random component (audit-only or IPW) recovers full-reference metrics better than pooling. Independent review (2026-09-12) added that before implementation Study C must define: (a) the **random sampling frame** (random over what population: all episodes, all windows, all time, per participant or pooled?); (b) **overlap/budget accounting** when a randomly drawn unit is also targeted (does it consume one budget unit or two, and which estimator sees it?); (c) a **100 %-random same-budget comparator**, without which "mixing helps" cannot be distinguished from "random alone is enough".
- **Plan:** pre-declare sampling frame, overlap accounting, comparator and estimators (T-09) before any Study C code.
- **Status:** open

### KI-17 Brier-score change alone is not proof of calibration distortion
- **Severity:** MEDIUM (interpretation risk)
- **Issue:** The Brier score decomposes into calibration (reliability), refinement/resolution and uncertainty terms. Selecting a label subset changes the base rate and the discrimination component, so the Brier score can move even when the reliability curve is unchanged. A later report must not describe "Brier changed under policy X" as "calibration was distorted".
- **Plan:** report calibration evidence separately from overall Brier (reliability curve distance, slope/intercept, and/or the calibration component of a Brier decomposition), and state which term moved. Exact metric set is part of the protocol freeze.
- **Status:** open (recorded from independent review, 2026-09-12)

### KI-18 Study B is under-specified
- **Severity:** HIGH for Study B
- **Issue:** Independent review (2026-09-12) listed items that must be defined before implementation: (a) **development roles** - which training participants serve model fitting vs calibration vs threshold/policy selection, and whether roles rotate; (b) **detector access** in development - whether the detector used to mask development labels may see development labels, and how it is trained relative to the roles above; (c) the **coverage constraint or loss** under which policies are compared (e.g. accepted error at fixed coverage vs a coverage-penalised risk); (d) **tie handling** when several policies are equally good on visible labels; (e) the **comparator** for regret (policy chosen on full development labels vs test-oracle policy, the latter diagnostic only).
- **Plan:** resolve as T-08 before any Study B code; record in `research_protocol.md` Section 6 as an amendment.
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
