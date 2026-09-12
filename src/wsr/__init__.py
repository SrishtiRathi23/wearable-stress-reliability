"""wsr - wearable stress reliability.

Code for "Label-Selection-Aware Evaluation of Calibration and Abstention in
Wearable Stress Prediction". See docs/PROJECT_CONTEXT.md for the scientific
context and docs/research_protocol.md for the pre-registration-style protocol.

Package layout (modules are added when the corresponding stage is implemented;
empty stub modules are deliberately not created):

    wsr.data           dataset audits, loaders, label handling
    wsr.preprocessing  windowing, signal quality, alignment
    wsr.features       per-modality window features + feature table builder
    wsr.models         classifiers, calibration, abstention policies
    wsr.experiments    Study A/B/C runners and the Nurse case study
    wsr.evaluation     metrics, calibration metrics, coverage, statistics, splits
    wsr.utils          paths, seeds, logging, config, integrity
"""

__version__ = "0.0.1"
