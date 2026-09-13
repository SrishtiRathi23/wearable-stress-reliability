# Study A Protocol - FROZEN before any selective-label result

**Status:** FROZEN 2026-09-13 (D-033); **pre-execution amendment SA-1 applied 2026-09-14 (D-034)** after the independent pre-execution review (APPROVE WITH REQUIRED AMENDMENTS). No mask has been built, no selector has run, no selective-label outcome has been inspected. Further changes to confirmatory elements require a dated amendment at the bottom and a new decision entry.

**Inputs (immutable):** the independently approved Phase-3 prediction artifact `results/phase3/oof_predictions.parquet`, SHA-256 `34010de95b3779b652fd5c9f7d372e69b3d7d61aac020912fd871d1298022212` (D-032), and the Phase-2 reference annotation it carries (`binary_eligible`, `analysis_label`, raw-code provenance; schema 1.1.0). Any Study-A runner must load predictions only through `wsr.experiments.phase3_artifact.load_approved_predictions`, which verifies `frozen: true` and the hash, and must fail closed on mismatch. No model is retrained, retuned or replaced.

**Approved core (unchanged by SA-1):** full 60-s candidate frame; 25/50/75 % budgets; 100 % control; one-window annotation cost; out-of-target budget accounting; common LR/RF/XGB consensus selector; uniform same-budget random comparator; 500 random repetitions; fixed Phase-3 predictions.

---

## 1. Scientific question

When the same frozen wearable-stress predictions are evaluated under different label-observation policies, how much can apparent performance, probabilistic performance and model-selection conclusions differ from the full-reference result? Predictions and model parameters stay frozen; only the observation mask changes. This is an OFFLINE label-acquisition experiment on one fixed recording set.

## 2. Candidate / observation unit (T-05)

- **Primary candidate unit:** one complete, non-overlapping 60-s Phase-2 time-grid window.
- **Candidate frame:** ALL complete windows of each participant (currently 1442 in total; recomputed from the artifact, never hard-coded into mask logic; always reconstructed from the complete frame, never from a previously filtered binary-only table).
- **Boundaries:** exactly the Phase-2 time-only boundaries. Never restarted at reference-label boundaries; never prefiltered to `binary_eligible`; never constructed from stress/baseline blocks; protocol labels play no part in candidate generation (invariant #19, D-023).
- **Selection unit != inferential unit.** The participant remains the inferential unit.

## 3. Annotation cost and out-of-target selection

- One selected candidate window costs **one annotation unit**, regardless of the label subsequently revealed.
- A selected window that turns out to be transient (code 0), amusement (3), meditation (4), a reading block (5/6/7) or mixed-label still consumes its unit. It does NOT become baseline, does NOT become stress, and does NOT contribute to binary performance metrics; it contributes only to label-yield statistics. It is annotation budget spent outside the target population.
- **The selector never receives oracle knowledge of binary eligibility.**

## 4. Budget grid and hierarchy (T-06; SA-1)

For each participant p independently, with N_p = number of complete candidate windows: B_p(f) = floor(f x N_p). Targeted and random use EXACTLY the same B_p for a participant and fraction. Canonical fraction strings are `"0.25"`, `"0.50"`, `"0.75"`, `"1.00"`.

| Role | Fraction |
|---|---|
| **PRIMARY confirmatory budget** | **0.50** (chosen before execution as the middle predeclared budget) |
| Supporting budget-response analyses | 0.25, 0.75 |
| Negative control | 1.00 |

If the 50 % primary endpoint is unavailable or inconclusive, 25 % or 75 % are NOT promoted to replace it; that outcome is reported as is. No 10 % confirmatory analysis; 10 % is never more than an explicitly exploratory stress test.

## 5. Primary targeted selector (T-07; terminology per SA-1)

One COMMON targeted observation policy, shared by every evaluated model family, built from the frozen Phase-3 predictions of the THREE learned families (Logistic Regression, Random Forest, XGBoost; majority excluded from the selector).

For each participant p:

1. take every complete candidate window w of p;
2. for each learned family m, compute the within-participant rank of `prob_positive_m(w)` over the participant's COMPLETE frozen recording with `scipy.stats.rankdata(method="average")` (1 = lowest) and the percentile rank `r_m(w) = (rank - 1) / (N_p - 1)` (0 = lowest, 1 = highest; if N_p = 1, r = 0);
3. `consensus_score(w) = mean(r_logistic(w), r_random_forest(w), r_xgboost(w))`;
4. select the B_p windows with the **highest consensus stress rank**.

Rank aggregation (not raw-probability averaging) prevents one family's probability scale from dominating the policy. Rank aggregation does not preserve calibrated confidence magnitude, so the selected windows are NOT described as "most confident of stress"; they are the highest consensus stress rank.

**Scope statement (SA-1).** This is the PRIMARY selector for the restricted scenario *self-consistent model-triggered batch label acquisition*: an OFFLINE acquisition experiment in which percentile ranking uses the participant's complete frozen recording. It is not demonstrated prospective online triggering. Using the evaluated models inside the selector is not label leakage (no reference label enters the selector). Results are conditional on this self-referential acquisition mechanism. An externally independent selector is optional later sensitivity evidence, not required for the restricted claim.

## 6. Targeted tie-breaking (encoding fixed by SA-1)

Ties in consensus score are broken without labels and without time order:

- `tie_seed = child_seed(42, "study_a_selector_tie")` (`wsr.utils.seeds.child_seed`, global seed 42);
- `key_string = f"{tie_seed}:{participant_id}:{window_id}"` (decimal integer seed, ASCII colon separators, participant id such as `S2`, window id such as `wesad:S2:w00017`);
- `tie_key = int.from_bytes(sha256(key_string.encode("utf-8")).digest(), byteorder="big", signed=False)`;
- among tied consensus scores, the window with the **lower** `tie_key` is selected first (ascending value wins).

Same repository state + same prediction artifact => identical targeted mask. Input row order must not change the mask.

## 7. Random comparator and dataset-level replicates (SA-1 clarifies)

- Uniform sampling WITHOUT replacement of exactly B_p windows from ALL complete candidate windows of the participant (no restriction to eligible windows, no label stratification, no guaranteed class coverage, no conditioning on protocol state).
- **500 repetitions, k = 0, ..., 499.**
- **Seed serialisation (frozen):** `seed_string = f"study_a_random:{participant_id}:{fraction}:{k}"` with `fraction` one of the canonical strings `"0.25" | "0.50" | "0.75" | "1.00"` and `k` the decimal integer repetition index, e.g. `study_a_random:S2:0.50:17`; `seed = child_seed(42, seed_string)` (SHA-256 of the UTF-8 bytes of `f"{42}:{seed_string}"` per `wsr.utils.seeds`); `rng = numpy.random.default_rng(seed)`; `selected = rng.choice(window_indices_sorted_by_window_index, size=B_p, replace=False)`.
- Participant-specific draws are independent across participants because their seed strings contain the participant id.
- **Dataset-level replicate:** for budget b and repetition k, `M_{b,k} = union over all 15 participants of M_{p,b,k}` - one coherent dataset-level acquisition realisation. The same k is used across participants for every dataset-level (H-A3) outcome; participant repetition indices are never mixed post hoc.
- Within a participant/budget/repetition the SAME random mask is applied to Logistic Regression, Random Forest, XGBoost and the majority baseline.
- Repetitions estimate Monte-Carlo variability on ONE fixed dataset. They are never n = 500 participants or datasets.

## 8. Full reference

For every participant and model the full-reference value is computed on all binary-eligible windows of that participant using the frozen Phase-3 predictions (this reproduces `results/phase3/per_participant_metrics.csv`). Other protocol states are never treated as negatives. Frozen full-reference all-15 participant-mean BA per learned model is denoted **F_m**.

## 9. Observed label set

After a mask is fixed, labels are revealed conceptually for the selected windows only: homogeneous code 1 -> binary reference 0 (baseline); homogeneous code 2 -> 1 (protocol stress); anything else -> binary target unavailable (budget still consumed). Recorded per participant/budget/policy(/repetition): `selected_count`, `eligible_selected_count`, `ineligible_selected_count`, `label_yield_fraction = eligible_selected / selected`, `baseline_selected_count`, `stress_selected_count`, class balance among eligible selected labels, `baseline_coverage = baseline_selected / full_reference_baseline`, `stress_coverage = stress_selected / full_reference_stress`.

## 10. Metric availability table (SA-1)

| Metric | Requires | Otherwise |
|---|---|---|
| Balanced accuracy | >= 1 observed baseline AND >= 1 observed stress | NA |
| AUROC | both classes | NA |
| Baseline recall | >= 1 observed baseline | NA |
| Stress recall | >= 1 observed stress | NA |
| Two-class macro-F1 | both classes (for THIS study) | NA |
| Average precision | both classes (for THIS study) | NA |
| Brier score | >= 1 eligible binary label | NA |
| Log loss | >= 1 eligible binary label; fixed class space {0, 1} | NA |
| Zero eligible labels | - | all binary metrics NA |

NA is never replaced by 0.5, majority labels, borrowed labels or pooled participants. Library warning behaviour must never produce pseudo-values; availability is checked explicitly before any library call. `ba_estimable` is recorded per participant/budget/policy(/repetition) as an OUTCOME of the policy; because the mask is common across models, estimability is common across the learned families.

## 11. H-A1 - performance distortion (conditional estimand; SA-1)

For participant p, budget b, learned model m, and observation realisation o (targeted, or random repetition k), where BA is estimable:
`delta_BA = BA_observed(p, m, b, o) - BA_full(p, m)` (positive = appears better), `abs_delta_BA = |delta_BA|`, and the primary distortion `D(p, b, o) = mean(abs_delta_BA over {logistic, random_forest, xgboost})` (majority excluded).

Let `E_T(p,b) = 1` if targeted BA is estimable else 0; `E_R(p,b,k) = 1` if random repetition k has estimable BA else 0; `K(p,b) = {k : E_R(p,b,k) = 1}`; `D_T(p,b)` and `D_R(p,b,k)` the targeted and random distortions.

**Frozen H-A1 participant contrast:**

`C_BA(p,b) = D_T(p,b) - mean_{k in K(p,b)} D_R(p,b,k)`, **defined only if `E_T(p,b) = 1` AND `|K(p,b)| > 0`.**

Interpretation: among participants whose targeted observations permit BA estimation, compare targeted BA distortion against random BA distortion conditional on random BA also being estimable. This is explicitly a CONDITIONAL estimand. If targeted BA is unavailable, H-A1 is unavailable for that participant: no penalty, no 0 / 0.5 substitution, no redrawn targeted mask, no borrowed labels, no promoted endpoint. Per budget report: n contributing participants, their ids, targeted estimability, random estimability fraction, valid random repetition count |K(p,b)|.

## 12. H-A2 - estimability / class coverage (primary contrast; SA-1)

**Frozen primary quantity:** `C_E(p,b) = mean_{k=0..499} E_R(p,b,k) - E_T(p,b)` (mean over ALL 500 repetitions). Positive: targeted is less likely than random to yield a two-class subset on which BA can be estimated; negative: more likely; zero: same availability. Exists for ALL 15 participants. **Primary H-A2 summary:** equal-participant mean C_E.

Supporting H-A2 quantities (never combined into the primary endpoint): eligible-label yield; baseline coverage; stress coverage; selected baseline/stress counts; selected class balance.

## 13. H-A3 - model selection and regret (DATASET-LEVEL; SA-1)

H-A3 is dataset-level. No sign test, Wilcoxon or participant-level NHST is used for H-A3.

**Model selection under an observation realisation o:** observed participant-level BA per learned model where estimable; `S_o` = participants with estimable BA. Dataset-level selection requires `|S_o| >= 10` of 15 (two-thirds cohort sufficiency); otherwise `model_selection = NOT ESTIMABLE` (no forced ranking; regret NA, never zero). If `|S_o| >= 10`: equal-participant mean observed BA over the SAME evaluable participants S_o for all three learned models; `m_hat(o)` = highest; ties within 1e-12 resolve in the fixed canonical order Logistic Regression > Random Forest > XGBoost.

**Primary: EMPIRICAL FULL-COHORT SELECTION REGRET.** `R_cohort(o) = max_m F_m - F_{m_hat(o)}`, with F_m the frozen full-reference all-15 participant-mean BA (global best under current Phase-3 results: XGBoost, so `max_m F_m = F_xgboost`). `top_model_changed(o) = (m_hat(o) != argmax_m F_m)`. Interpretation: how much full-cohort Phase-3 balanced accuracy would be sacrificed because incomplete observed labels led to selecting another learned model. NOT validated deployment regret for future populations. Unavailable selection -> regret NA.

**Required diagnostic: SAME-EVALUABLE-COHORT REGRET.** `F_m(S_o)` = equal-participant mean full-reference BA of model m over S_o (full reference labels of those same participants); `subset_full_best_model(o)` = argmax_m F_m(S_o) with the same 1e-12 tie rule and canonical order; `R_subset(o) = max_m F_m(S_o) - F_{m_hat(o)}(S_o)`; `subset_top_model_changed(o) = (m_hat(o) != subset_full_best_model(o))`. Purpose: separate incomplete-label ranking error within the evaluable subset from changes caused merely by a different participant subset becoming evaluable. Secondary; never replaces R_cohort.

**Targeted, per budget, report:** number and identities of participants with estimable BA; selection estimable yes/no; selected model if estimable; `top_model_changed`; `R_cohort`; `R_subset`; `subset_full_best_model`; `subset_top_model_changed`.

**Random, per budget:** for each dataset-level replicate k = 0..499 assemble all participant masks with repetition k and compute the same outcomes. Across the 500 realisations report: distribution of |S_o|; fraction NOT ESTIMABLE; among valid repetitions (denominator always stated): frequency XGBoost / Logistic / RF selected; probability the global top model changes; R_cohort distribution; R_subset distribution. An LR <-> RF swap (F_logistic - F_random_forest ~ 2e-5) is not exaggerated; the meaningful event is displacement of XGBoost.

## 14. H-A4 - probabilistic-performance distortion (SA-1)

**Primary: Brier distortion.** For binary label y and frozen stress probability p, `BS = mean((p - y)^2)`. For participant p, budget b, realisation o, learned model m: `delta_BS = BS_observed - BS_full`, `abs_delta_BS = |delta_BS|`, `D_BS(p,b,o) = mean abs_delta_BS over {LR, RF, XGB}`. Let `K_BS(p,b)` = random repetitions with >= 1 eligible binary selected label. **Frozen contrast:** `C_BS(p,b) = D_BS_targeted(p,b) - mean_{k in K_BS(p,b)} D_BS_random(p,b,k)`, **defined only if targeted selected >= 1 eligible binary label AND |K_BS(p,b)| > 0** (conditional estimand; denominators reported).

**Secondary within H-A4: log loss** with analogous `D_LL`, `C_LL`. Brier and log loss are never averaged into a composite. **Numerical convention (frozen):** binary fixed label space {0, 1}; natural-log loss; float64 probabilities; `labels=[0, 1]` passed explicitly to `sklearn.metrics.log_loss`; probabilities clipped to `[eps, 1 - eps]` with `eps = numpy.finfo(numpy.float64).eps` (the pinned scikit-learn 1.9.1 convention) applied IDENTICALLY to observed and full-reference calculations; Brier is computed without clipping. Recorded in `configs/base.yaml: observation_policies.probabilistic_metrics`.

Brier and log loss are PROBABILISTIC PERFORMANCE measures; neither alone is called calibration distortion (KI-17). Calibration slope/intercept/ECE are not confirmatory endpoints; reliability plots, if used later, are descriptive.

## 15. Secondary classification metrics

Macro-F1, baseline recall, stress recall, AUROC, average precision on selected eligible labels under the availability table (Section 10); never redefined to avoid missingness.

## 16. Confirmatory hypotheses (hypotheses, not expected outcomes)

- **H-A1:** at matched budgets, targeted observation produces larger absolute BA distortion than uniform random observation - primary quantity C_BA (Section 11).
- **H-A2:** targeted observation more often prevents valid two-class BA estimation and/or produces poorer class coverage than matched random observation - primary quantity C_E (Section 12).
- **H-A3:** targeted observation more often changes the learned-model selection away from the full-reference top model and produces greater model-selection regret than matched random observation - dataset-level (Section 13).
- **H-A4:** targeted observation produces larger distortion in Brier (primary) / log-loss (secondary) estimates than matched random observation (Section 14).

Null or reversed results are scientifically valid and will be reported as such.

## 17. Confirmatory hierarchy (SA-1)

1. **PRIMARY confirmatory endpoint:** H-A1 `C_BA` at the 50 % budget.
2. **Required companion:** H-A2 `C_E` at 50 %, reported regardless of whether H-A1 is estimable.
3. **Supporting at 50 %:** H-A3 dataset-level model-selection consequences; H-A4 Brier distortion `C_BS`.
4. **Supporting budget-response:** 25 % and 75 % versions of H-A1/H-A2/H-A3/H-A4.
5. **Negative control:** 100 %.
6. **Secondary / descriptive:** log loss; per-model distortions; signed distortions; class recalls; macro-F1; AUROC; AP; label-yield distributions; participant plots; selected-state composition.

No supporting/secondary quantity is promoted to replace an unavailable or weak primary result after outcomes are seen.

## 18. Negative control (pipeline sanity)

At the 100 % budget both policies select the complete candidate frame, so: the complete candidate frame is recovered; all 434 binary-eligible labels are recovered; participant metrics equal the frozen full-reference values exactly; BA, Brier and log-loss distortions are 0; regrets are 0; the full-reference model ranking is recovered. Selected window ids are canonicalised (sorted) before scoring/comparison so ordering alone cannot cause a control failure. **Frozen numerical tolerance:** exact equality for counts, ids and selected model; `abs <= 1e-12` for metric and distortion values. Any deviation means STOP and debug before any other Study-A output is examined.

## 19. Mask pairing

For every policy/budget/repetition the mask is independent of the evaluated model: all families receive the same selected window ids. Enforced by data structure (one mask table keyed by participant/policy/budget/repetition, joined to all families) and by tests. Never one random mask per model; never a model's correctness in a mask.

## 20. Inference and reporting plan (T-14; frozen by SA-1)

For H-A1, H-A2 and H-A4: one participant-level contrast per contributing participant, equal participant weight, participant as inferential unit. Report: mean contrast; median; IQR; range; counts of positive / approximately-zero / negative participants (numerical zero: `abs(contrast) <= 1e-12`); `n_evaluable / 15`; included participant ids.

Monte-Carlo characterisation: for random conditional means, the empirical random-repetition SD and `MCSE = SD / sqrt(number of valid random repetitions)`; for random probabilities (e.g. estimability) the binomial Monte-Carlo standard error `sqrt(p_hat (1 - p_hat) / 500)`, or with the actual denominator where conditioning applies.

No formal p-value threshold is the primary confirmatory decision rule; no sign test / Wilcoxon is chosen after seeing results; windows and random masks are never treated as independent participants; no participant bootstrap CI is required as primary evidence. Summaries are interpreted as finite-cohort paired effects for this WESAD experiment, not complete uncertainty for new populations. For H-A3: dataset-level descriptive / Monte-Carlo summaries only (Section 13).

## 21. Mandatory pre-output implementation invariants (SA-1)

The later implementation must pass ALL of these tests before any 25/50/75 % result is viewed:

1. The approved frozen prediction hash verifies (fail closed otherwise).
2. Exactly B_p unique selected windows per participant/policy/budget(/repetition).
3. The same selected window ids are applied to every model family.
4. Removing or arbitrarily perturbing reference labels, binary eligibility and protocol-state columns (with frozen predictions fixed) leaves targeted and random masks unchanged.
5. The candidate frame is reconstructed from ALL complete windows, never from a previously filtered binary-only table.
6. Input row reordering does not change masks.
7. Tie handling: identical model scores; consensus ties; stable hash ordering.
8. Synthetic edge cases: N_p = 1; zero selected binary labels; one selected class; exactly one label per class.
9. Explicit dataset-level random replicate k assembly (same k across participants).
10. The >= 10/15 model-selection sufficiency rule.
11. Full-cohort regret computation.
12. Same-evaluable-cohort regret computation.
13. Metric availability table enforcement (no library pseudo-values).
14. 100 % control: complete frame recovered; all 434 eligible labels recovered; frozen participant metrics reproduced; full ranking reproduced; BA/Brier/log-loss distortions zero; regrets zero (tolerance per Section 18).

## 22. Claim scope (recorded before results)

If Study A supports its hypotheses, defensible wording is approximately: *"In this controlled WESAD observation-policy experiment, acquiring labels according to a fixed ensemble's within-recording stress ranking altered apparent performance and/or model-selection conclusions relative to full-reference evaluation and same-budget random acquisition."*

Not inferred automatically: the same distortion magnitude in other datasets; real-world annotator behaviour or cost; clinical validity; psychological-stress specificity; pure calibration failure; prospective online triggering performance; regret generalisation to new participants; robustness to independent acquisition mechanisms. The consensus selector bounds the claim to the specified model-triggered observation mechanism.

## 23. Implementation questions still open (not decided here)

- Reporting format for non-estimable cells across budgets (tables must show NA counts, never drop them silently).
- Whether the exploratory 10 % stress test is run at all.
- Study B/C designs remain separate (T-08, T-09, KI-23); the Study-C `mixed`/audit policies are not part of Study A.

---

## Amendments

- **SA-1, 2026-09-14 (D-034), pre-execution, after independent review; no mask existed.** Added: 50 % primary budget hierarchy (Section 4); selector terminology and scope statement, explicit tie-key encoding (Sections 5-6); canonical fraction strings, exact random seed serialisation and dataset-level replicate definition (Section 7); metric availability table (Section 10); conditional H-A1 estimand C_BA (Section 11); primary H-A2 contrast C_E (Section 12); H-A3 restated as dataset-level with full-cohort and same-evaluable-cohort regret (Section 13); Brier primary / log-loss secondary with frozen numerical convention (Section 14); confirmatory hierarchy (Section 17); tolerance in the negative control (Section 18); inference plan T-14 (Section 20); mandatory implementation invariants (Section 21); claim scope (Section 22). The earlier cross-reference to "Section 12" for the negative control was corrected (it is Section 18); the earlier open item wrongly numbered T-11 is now T-14.
