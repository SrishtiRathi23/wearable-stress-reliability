"""Dataset access layer.

Planned modules (created when implemented, after the corresponding audit):
    audit_wesad.py   - inspect the raw WESAD release; verify participants, signals,
                       sampling rates, label codes; write data/manifests/*.
    audit_nurse.py   - same for the Nurse Stress dataset, plus timestamp/label alignment.
    load_wesad.py    - read raw WESAD into a provenance-preserving interim format.
    load_nurse.py    - read raw Nurse data likewise; unknown labels stay unknown.
    labels.py        - label mapping helpers; every mapping is explicit and logged.

Invariant: loaders never modify files under data/raw (see wsr.utils.integrity).
"""
