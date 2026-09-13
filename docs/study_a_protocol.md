# Study A Protocol - FROZEN before any selective-label result

**Status:** FROZEN 2026-09-13 (D-033), prior to any mask construction, selector execution or distortion computation. Changes to confirmatory elements require a dated amendment at the bottom and a new decision entry. Nothing in this document was informed by a Study-A outcome; no mask has been built.

**Inputs (immutable):** the independently approved Phase-3 prediction artifact `results/phase3/oof_predictions.parquet`, SHA-256 `34010de95b3779b652fd5c9f7d372e69b3d7d61aac020912fd871d1298022212` (D-032), and the Phase-2 reference annotation it carries (`binary_eligible`, `analysis_label`, raw-code provenance; schema 1.1.0). Any Study-A runner must load predictions only through `wsr.experiments.phase3_artifact.load_approved_predictions`, which verifies `frozen: true` and the hash, and must fail closed on mismatch. No model is retrained, retuned or replaced.

---

## 1. Scientific question

When the same frozen wearable-stress predictions are evaluated under different label-observation policies, how much can apparent performance, probabilistic performance and model-selection conclusions differ from the full-reference result? Predictions and model parameters stay frozen; only the observation mask changes.

## 2. Candidate / observation unit (resolves T-05)

- **Primary candidate unit:** one complete, non-overlapping 60-s Phase-2 time-grid window.
- **Candidate frame:** ALL complete windows of each participant (currently 1442 in total; the number is recomputed from the artifact, never hard-coded into mask logic).
- **Boundaries:** exactly the Phase-2 time-only boundaries. Never restarted at reference-label boundaries; never prefiltered to `binary_eligible`; never constructed from stress/baseline blocks; protocol labels play no part in candidate generation (invariant #19, D-023).
- **Selection unit != inferential unit.** The participant remains the inferential unit.

## 3. Annotation cost and out-of-target selection

- One selected candidate window costs **one annotation unit**, regardless of the label subsequently revealed.
- A selected window that turns out to be transient (code 0), amusement (3), meditation (4), a reading block (5/6/7) or mixed-label still consumes its unit. It does NOT become baseline, does NOT become stress, and does NOT contribute to binary performance metrics; it contributes only to label-yield statistics. It is annotation budget spent outside the target population, not "wasted data" in a general sense.
- **The selector never receives oracle knowledge of binary eligibility.**

## 4. Budget grid (resolves T-06)

For each participant p independently, with N_p = number of complete candidate windows:

- Primary annotation fractions f in {0.25, 0.50, 0.75}; negative-control/sanity fraction 1.00.
- Budget B_p(f) = floor(f x N_p).
- Targeted and random policies use EXACTLY the same B_p for a participant and fraction.
- Fractions are not altered after inspecting mask outcomes. 10 % is NOT a confirmatory budget; if ever run it is labelled an exploratory stress test and does not replace the primary grid.
- At f = 1.00 both policies select the complete candidate frame (Section 12).

## 5. Primary targeted selector (resolves T-07)

One COMMON targeted observation policy, shared by every evaluated model family, built from the frozen Phase-3 predictions of the THREE learned families (Logistic Regression, Random Forest, XGBoost; the majority baseline is excluded from the selector).

For each participant p:

1. take every complete candidate window w of p;
2. for each learned family m, compute the within-participant rank of `prob_positive_m(w)` with `scipy.stats.rankdata(method="average")` (1 = lowest); convert to a percentile rank `r_m(w) = (rank - 1) / (N_p - 1)` (0 = lowest, 1 = highest; if N_p = 1, r = 0);
3. `consensus_score(w) = mean(r_logistic(w), r_random_forest(w), r_xgboost(w))`;
4. select the B_p windows with the highest consensus score.

Rank aggregation (not raw-probability averaging) prevents one family's probability scale from dominating the policy. The selector is prediction-driven, label-independent, identical across evaluated models, and deterministic once ties are resolved (Section 6). It answers the "self-consistent detector-triggered labelling" scenario: labels are sought where the deployed models collectively look most confident of stress. It does NOT model an independent external detector; that is a documented limitation, not an open choice.

## 6. Targeted tie-breaking

Ties in consensus score are broken without labels and without time order:

- `tie_key(w) = SHA-256( f"{tie_seed}:{participant_id}:{window_id}" )` interpreted as an unsigned integer, where `tie_seed = child_seed(global_seed, "study_a_selector_tie")` (global seed 42, `wsr.utils.seeds.child_seed`);
- among tied consensus scores, lower `tie_key` is selected first.

Same repository state + same prediction artifact => identical targeted mask. The exact key string, seed stream name and ordering are part of the frozen protocol.

## 7. Random comparator

- Uniform sampling WITHOUT replacement of exactly B_p windows from ALL complete candidate windows of the participant (no restriction to eligible windows, no label stratification, no guaranteed class coverage, no conditioning on protocol state).
- **500 repetitions** per participant and budget. Repetition k uses `rng = numpy.random.default_rng(child_seed(global_seed, f"study_a_random:{participant_id}:{fraction}:{k}"))` and `rng.choice(window_indices_sorted_by_window_index, size=B_p, replace=False)`.
- Within a participant/budget/repetition the SAME random mask is applied to Logistic Regression, Random Forest, XGBoost and the majority baseline.
- Repetitions estimate Monte-Carlo variability of the random policy. They are never treated as independent participants.

## 8. Full reference

For every participant and model the full-reference value is computed on all binary-eligible windows of that participant using the frozen Phase-3 predictions (this reproduces `results/phase3/per_participant_metrics.csv`). Other protocol states are never treated as negatives.

## 9. Observed label set

After a mask is fixed, labels are revealed conceptually for the selected windows only: homogeneous code 1 -> binary reference 0 (baseline); homogeneous code 2 -> 1 (protocol stress); anything else -> binary target unavailable (budget still consumed). Recorded per participant/budget/policy(/repetition): `selected_count`, `eligible_selected_count`, `ineligible_selected_count`, `label_yield_fraction = eligible_selected / selected`, `baseline_selected_count`, `stress_selected_count`, and the class balance among eligible selected labels (`stress_selected / eligible_selected`).

## 10. Metric availability rule

Participant-level balanced accuracy requires at least one selected baseline window AND at least one selected stress window. Otherwise BA = NA (not estimable): never 0.5, never majority-filled, never borrowed from another participant, never pooled to force a value. `ba_estimable` (true/false) is recorded explicitly as an OUTCOME of the observation policy. Because the mask and reference subset are common across models, `ba_estimable` is common across the learned families.

## 11. Primary performance-distortion outcome

For participant p, budget b, policy, learned model m, where BA is estimable:

- `BA_full(p, m)` = frozen Phase-3 full-reference participant BA;
- `BA_observed(p, m, b, policy)` = BA on the selected binary-eligible labels only (threshold 0.5, stored hard predictions);
- signed `delta_BA = BA_observed - BA_full` (positive = model appears better than under full reference);
- `abs_delta_BA = |delta_BA|`;
- **primary participant-level distortion** `D(p, b, policy) = mean(abs_delta_BA over {logistic, random_forest, xgboost})`. The majority baseline is excluded from D and remains a negative-control/reference model.

## 12. Primary targeted-vs-random comparison

For each participant and budget: targeted gives one deterministic D (if estimable); random gives a distribution of D over the repetitions in which BA is estimable. Recorded separately: targeted `ba_estimable`; random probability of BA being estimable (fraction of 500 repetitions); targeted label yield; random label-yield distribution. Non-estimability is never hidden. Among estimable cases: targeted D vs mean and median random D. Participant-level contrast: `C(p, b) = D_targeted(p, b) - mean_k D_random(p, b, k)` over estimable repetitions, computed only after the implementation has been checked against this protocol. The participant is the inferential unit; 500 repetitions are not n = 500.

## 13. Model-selection / ranking outcome

- Full-reference primary criterion (frozen, D-032): equal-participant-weight mean BA; learned-model order XGBoost > Logistic Regression > Random Forest (LR and RF essentially tied). Majority is a reference baseline, never a candidate "winning learned model".
- For each mask: observed participant-level BA per learned model where estimable.
- **Cohort sufficiency rule:** dataset-level model selection requires >= 10 of the 15 participants with estimable BA under that mask; otherwise `model_selection = NOT ESTIMABLE` (no forced ranking).
- If >= 10: equal-participant mean observed BA over the SAME evaluable participants for all three learned models; choose the highest. Ties within 1e-12 resolve in the fixed canonical order Logistic Regression > Random Forest > XGBoost (simpler-first; not outcome-driven).

## 14. Model-selection regret

`full_best_model = XGBoost` (frozen full-reference criterion). If a policy selects model m: `selection_regret = full_reference_mean_BA(XGBoost) - full_reference_mean_BA(m)`, using the frozen Phase-3 full-reference participant-mean values (all 15 participants). Regret is never evaluated with the incomplete labels used to choose m. Also recorded: `top_model_changed = (m != XGBoost)`. An LR <-> RF swap is not to be exaggerated (they differ by ~2e-5); the meaningful event is displacement of XGBoost.

## 15. Probabilistic-performance distortion (H-A4)

Brier score and log loss on selected binary-eligible labels vs each participant/model's full-reference value; signed and absolute differences recorded. These are PROBABILISTIC PERFORMANCE measures and are never called "calibration distortion" on their own (KI-17). They remain computable for a single-class selected subset if at least one eligible label exists; with zero eligible selected labels they are NA. Calibration slope/intercept/ECE are not confirmatory Study-A endpoints; reliability plots, if used later, are descriptive.

## 16. Secondary classification metrics

Macro-F1, baseline recall, stress recall, AUROC, average precision on selected eligible labels; metrics requiring both classes follow the availability rule (NA when invalid) and are not redefined to avoid missingness.

## 17. Confirmatory hypotheses (hypotheses, not expected outcomes)

- **H-A1:** at matched annotation budgets, targeted observation produces larger absolute BA distortion than uniform random observation (primary quantity D(p, b, policy)).
- **H-A2:** targeted observation more often prevents valid two-class BA estimation and/or produces poorer class coverage than matched random observation.
- **H-A3:** targeted observation more often changes the learned-model selection away from the full-reference top model and produces greater model-selection regret than matched random observation.
- **H-A4:** targeted observation produces larger distortion in Brier/log-loss estimates than matched random observation (probabilistic-performance evidence, not pure calibration).

Null or reversed results are scientifically valid and will be reported as such.

## 18. Negative control (pipeline sanity)

At the 100 % budget both policies select the complete candidate frame, so: all 434 binary-eligible labels are recovered; participant BA equals the frozen full-reference BA exactly; BA distortion is 0; Brier/log-loss distortion is 0; the full-reference model ranking is recovered. Any deviation means STOP and debug before any other Study-A output is examined.

## 19. Mask pairing

For every policy/budget/repetition the mask is independent of the evaluated model: all families receive the same selected window ids. This is enforced by data structure (one mask table keyed by participant/policy/budget/repetition, joined to all families) and by tests. Never one random mask per model; never a model's correctness in a mask.

## 20. Implementation questions still open (not decided here)

- **T-11 (inference for H-A1..H-A4):** the participant-level inferential procedure (e.g. exact paired sign / Wilcoxon signed-rank on the 15 participant contrasts, reported with n_evaluable and effect sizes; no window-level tests). Proposed, awaiting approval; not frozen by this document.
- Reporting format for non-estimable cells across budgets (tables must show counts of NA, never drop them silently).
- Whether the exploratory 10 % stress test is run at all.
- Study B/C designs remain separate (T-08, T-09, KI-23).

---

## Amendments

_None._
