"""Window-level feature extraction per modality.

Planned modules: eda.py, cardiac.py, temperature.py, activity.py, build_features.py.

Modality-appropriate processing; signals are NOT blindly resampled to a common
rate. HRV features only when beat coverage supports them.
"""
