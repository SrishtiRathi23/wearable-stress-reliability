# Label-Selection-Aware Evaluation of Calibration and Abstention in Wearable Stress Prediction

*Controlled label-observation experiments and a nurse stress case study.*

Undergraduate minor project (Electrical & Electronics Engineering). Python package name: `wsr` (wearable stress reliability).

## What this project studies

Most wearable-stress papers train a model, predict stress / non-stress, and score the predictions against whatever labels exist. This project asks a question about the *scoring* step:

> If only some moments ever receive a reference label - and those moments were picked because they looked interesting - how much of a model's apparent accuracy and confidence is real, and how much is an artifact of which moments got labelled?

Concretely, we keep a trained model and its predictions fixed, and change only *which labels the evaluator is allowed to see*. We then measure how far the apparent accuracy, calibration, model ranking and abstention ("I am not sure, I will not answer") behaviour drift from what the full reference labels would show.

## Why it differs from ordinary stress classification

- The classifier is not the contribution; it is a fixed instrument. We use simple, reproducible models (logistic regression, random forest, XGBoost).
- The experimental variable is the **label-observation policy**, not the model architecture.
- We care about **calibration** (do probabilities mean what they say?) and **abstention** (error *and* coverage), not just accuracy.
- We evaluate strictly **across participants** (a person is never in both training and test), and we treat the participant - not the 60-second window - as the unit of evidence.
- We explicitly test whether biased development labels can lead to choosing the **wrong operating policy**, and whether a small random labelling audit helps.

## Dataset roles

| Dataset | Role |
|---|---|
| **WESAD** (Schmidt et al. 2018) | Primary controlled benchmark. Protocol-labelled conditions let us *simulate* hiding labels while keeping a full reference for evaluation. Used first. |
| **Nurse Stress** (Hosseini et al. 2022) | Real hospital-shift wearable data. Labels exist only for validated events; the rest of the shift is **unknown, not non-stress**. Used as a bounded case study. |
| **Stress-Predict** (Iqbal et al. 2022) | Later replication under a different controlled protocol (not a daily-life dataset). Optional. |

No new human data is collected; all datasets are public and anonymised. Raw files live under `data/raw/` and are never committed or modified.

## Planned studies

- **Study A - Measurement distortion.** Fixed predictions; vary the observation mask (full / random / detector-triggered / duration-preferred / mixed). How much do apparent metrics and rankings move?
- **Study B - Policy-selection distortion.** Choose calibration/abstention policies using selectively observed *development* labels, lock them, evaluate on the full reference. Do biased labels pick worse policies?
- **Study C - Mitigation.** At fixed labelling budget, does reserving part of it for random audit reduce the distortion?
- **Study D - Nurse case study.** What can honestly be said from the real data, given that most shift time is unlabelled?

The full protocol, hypotheses, and open decisions are in [`docs/research_protocol.md`](docs/research_protocol.md).

## Current status

**Repository bootstrap only.** No dataset has been inspected, no preprocessing written, no model trained, no result produced. Next step (pending approval): WESAD audit (`src/wsr/data/audit_wesad.py`) to verify participants, signals, sampling rates and label codes against the official documentation.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (or any Python 3.11-3.13 with pip). Python 3.12 is what the project is developed and locked against.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt -e .
.venv/Scripts/python -m pytest        # Windows; use .venv/bin/python on Linux/macOS
```

Exact package versions used for reported results are in `requirements-lock.txt`.

Datasets are downloaded manually into `data/raw/<dataset>/` (see `data/raw/README.md`), then snapshotted:

```bash
.venv/Scripts/python -m wsr.utils.integrity snapshot data/raw/wesad
```

## Repository layout

```
configs/        base.yaml + per-dataset overrides (TODOs mark unresolved decisions)
data/           raw/ (never committed, never modified) interim/ processed/ manifests/ (committed)
docs/           PROJECT_CONTEXT, research_protocol, decisions, experiment_log, known_issues, dataset_notes
notebooks/      exploration only; no result may depend on notebook-only logic
src/wsr/        data, preprocessing, features, models, experiments, evaluation, utils
tests/          invariant tests (participant leakage, mask independence, raw immutability, seeding)
results/        tables, figures, predictions, models, logs (large artifacts not committed)
```

## Important research constraints (short form)

1. Unknown Nurse periods are never treated as non-stress.
2. The held-out participant never influences preprocessing, model, calibration or thresholds.
3. Observation masks never use test labels.
4. Windows are not independent participants; inference is at participant level.
5. Raw files are never modified; exclusions are always logged with reasons.
6. Random-row splits are diagnostics, never headline results.
7. No clinical, diagnostic or deployment-safety claims.
8. Novelty is not assumed; null results are valid.

The full list of 18 invariants is in [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md), Section 8.
