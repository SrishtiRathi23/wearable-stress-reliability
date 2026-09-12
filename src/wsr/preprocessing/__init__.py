"""Windowing, signal-quality flags and cross-modality alignment.

Planned modules: windowing.py, quality.py, alignment.py.

Every window row must carry provenance columns (dataset, participant, session,
window_start, window_end, episode/block id if applicable, source label,
label_provenance, quality flags). Identifiers are never dropped here.
"""
