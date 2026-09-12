"""Column schema registry for the Phase-2 feature table.

Every column written to the canonical table is declared here with its role.
`model_feature` is the single source of truth for what a future model may
consume; provenance, reference-label and quality columns are always False.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = "1.1.0"  # 1.1.0: q_acc_clipped -> q_acc_near_rail; per-axis constant check; no feature value changes


@dataclass(frozen=True)
class ColumnDef:
    name: str
    role: str  # "provenance" | "reference" | "quality" | "feature" | "feature_provisional"
    modality: str  # "meta" | "label" | "acc" | "bvp" | "eda" | "temp" | "window"
    definition: str
    unit: str = ""
    nullable: bool = False
    missing_reason: str = ""
    dtype: str = "float64"

    @property
    def model_feature(self) -> bool:
        return self.role == "feature"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_feature"] = self.model_feature
        return d


def feature(name: str, modality: str, definition: str, unit: str = "", nullable: bool = False, missing_reason: str = "", provisional: bool = False) -> ColumnDef:
    """A numeric window feature. `provisional=True` keeps the column in the table but
    sets model_feature=False until its quality has been reviewed (docs/known_issues.md)."""
    return ColumnDef(name, "feature_provisional" if provisional else "feature", modality, definition, unit, nullable, missing_reason, "float64")


def meta(name: str, modality: str, definition: str, dtype: str, unit: str = "", nullable: bool = False, missing_reason: str = "", role: str = "provenance") -> ColumnDef:
    return ColumnDef(name, role, modality, definition, unit, nullable, missing_reason, dtype)


PROVENANCE_DEFS: list[ColumnDef] = [
    meta("dataset", "meta", "Dataset name", "string"),
    meta("participant_id", "meta", "WESAD subject folder id (S2..S17)", "string"),
    meta("window_id", "meta", "Stable id '<dataset>:<participant>:w<index>'", "string"),
    meta("window_index", "meta", "0-based index of the window on the t=0-anchored grid", "int32"),
    meta("start_seconds", "window", "Window start on the synchronised pickle timeline, inclusive", "float64", "s"),
    meta("end_seconds", "window", "Window end, exclusive", "float64", "s"),
    meta("start_label_sample", "label", "First 700-Hz label sample index (inclusive)", "int64"),
    meta("end_label_sample", "label", "Label sample index end (exclusive)", "int64"),
    meta("acc_start_sample", "acc", "First wrist ACC sample (32 Hz), inclusive", "int64"),
    meta("acc_end_sample", "acc", "Wrist ACC sample end, exclusive", "int64"),
    meta("bvp_start_sample", "bvp", "First wrist BVP sample (64 Hz), inclusive", "int64"),
    meta("bvp_end_sample", "bvp", "Wrist BVP sample end, exclusive", "int64"),
    meta("eda_start_sample", "eda", "First wrist EDA sample (4 Hz), inclusive", "int64"),
    meta("eda_end_sample", "eda", "Wrist EDA sample end, exclusive", "int64"),
    meta("temp_start_sample", "temp", "First wrist TEMP sample (4 Hz), inclusive", "int64"),
    meta("temp_end_sample", "temp", "Wrist TEMP sample end, exclusive", "int64"),
    meta("source_file", "meta", "Release-relative pickle path the window was derived from", "string"),
    meta("source_sha256", "meta", "SHA-256 of that pickle as recorded in the raw checksum baseline", "string"),
]

REFERENCE_DEFS: list[ColumnDef] = [
    meta("raw_label_codes_present", "label", "Sorted raw label codes occurring in the window, comma-joined", "string", role="reference"),
    meta("n_label_samples", "label", "Number of 700-Hz label samples in the window (42000 for a complete 60-s window)", "int64", role="reference"),
    meta("is_label_homogeneous", "label", "True if exactly one raw code covers the whole window", "bool", role="reference"),
    meta("homogeneous_raw_label", "label", "The single raw code if homogeneous", "Int16", nullable=True, missing_reason="window is mixed", role="reference"),
    meta("condition_name", "label", "Readme name of the homogeneous code, or 'mixed'", "string", role="reference"),
    meta("binary_eligible", "label", "True only if homogeneous raw code 1 (baseline reference) or 2 (protocol-stress reference)", "bool", role="reference"),
    meta("analysis_label", "label", "0 = baseline reference, 1 = protocol-stress reference; null otherwise. Never majority-voted.", "Int8", nullable=True, missing_reason="window not binary-eligible", role="reference"),
    meta("ineligibility_reason", "label", "Why binary_eligible is False ('' if eligible)", "string", role="reference"),
]


def quality_defs() -> list[ColumnDef]:
    defs: list[ColumnDef] = []
    for mod, expected in (("acc", 1920), ("bvp", 3840), ("eda", 240), ("temp", 240)):
        defs += [
            meta(f"q_{mod}_n_samples", mod, f"Samples of {mod.upper()} actually in the window", "int64", role="quality"),
            meta(f"q_{mod}_count_ok", mod, f"n_samples == expected native count ({expected} for 60 s)", "bool", role="quality"),
            meta(f"q_{mod}_finite_fraction", mod, "Fraction of finite samples", "float64", role="quality"),
            meta(f"q_{mod}_constant", mod, "True if the signal range within the window is below constant_eps", "bool", role="quality"),
        ]
    defs += [
        meta("q_acc_near_rail", "acc", "At least one sample with |raw count| >= acc_rail_counts (provisional 127; E4 range +-2 g = +-128 counts). Near-rail, not proven clipping; informational, never an exclusion criterion. Was q_acc_clipped in schema 1.0.0.", "bool", role="quality"),
        meta("q_eda_below_plausible", "eda", "Any EDA sample below eda_min_plausible_us (provisional threshold)", "bool", role="quality"),
        meta("q_temp_out_of_range", "temp", "Any TEMP sample outside temp_plausible_range_c (provisional)", "bool", role="quality"),
        meta("q_eda_decomposition_ok", "eda", "Recording-level tonic/phasic decomposition succeeded for this participant", "bool", role="quality"),
        meta("q_bvp_pulse_detection_ok", "bvp", "Recording-level pulse-peak detection ran for this participant", "bool", role="quality"),
        meta("q_bvp_hr_available", "bvp", "The in-house detector met MECHANICAL count/coverage criteria (>= min_beats_for_hr valid IBIs and coverage >= min_beat_coverage). Says nothing about physiological validity of the beats (KI-21).", "bool", role="quality"),
        meta("q_feature_error", "window", "Exception text if any feature family failed for this window ('' otherwise)", "string", role="quality"),
        meta("q_structural_ok", "window", "All modality counts ok, all finite, no constant signal, no feature error. NOT a physiological-credibility claim.", "bool", role="quality"),
    ]
    return defs


class FeatureContractError(RuntimeError):
    """The table does not match the committed feature schema."""


def model_feature_columns_from_schema(schema: dict) -> list[str]:
    """Exact ordered allowlist of model inputs from a schema document (role == 'feature')."""
    cols = [c["name"] for c in schema["columns"] if c.get("role") == "feature"]
    declared = schema.get("model_feature_columns")
    if declared is not None and declared != cols:
        raise FeatureContractError("schema model_feature_columns disagrees with column roles")
    if len(set(cols)) != len(cols):
        raise FeatureContractError("duplicate model feature names in schema")
    return cols


def select_model_features(table, schema: dict):
    """Return table[model features] in schema order; fail loudly on any disagreement.

    Rejects: missing required feature, duplicate column names in the table,
    a provisional/quality/reference/provenance column in the allowlist, or a
    table whose declared column set disagrees with the schema.
    """
    allow = model_feature_columns_from_schema(schema)
    table_cols = list(table.columns)
    if len(set(table_cols)) != len(table_cols):
        dup = sorted({c for c in table_cols if table_cols.count(c) > 1})
        raise FeatureContractError(f"duplicate columns in table: {dup}")
    missing = [c for c in allow if c not in table_cols]
    if missing:
        raise FeatureContractError(f"required model features missing from table: {missing}")
    schema_cols = [c["name"] for c in schema["columns"]]
    if table_cols != schema_cols:
        raise FeatureContractError("table columns differ from the committed schema (order or membership)")
    roles = {c["name"]: c["role"] for c in schema["columns"]}
    bad = [c for c in allow if roles.get(c) != "feature"]
    if bad:
        raise FeatureContractError(f"non-feature columns in allowlist: {bad}")
    return table[allow]


def all_column_defs(feature_defs: list[ColumnDef]) -> list[ColumnDef]:
    return PROVENANCE_DEFS + REFERENCE_DEFS + quality_defs() + feature_defs


def schema_document(feature_defs: list[ColumnDef], parameters: dict) -> dict:
    cols = all_column_defs(feature_defs)
    names = [c.name for c in cols]
    assert len(names) == len(set(names)), "duplicate column names in schema"
    return {
        "schema_version": SCHEMA_VERSION,
        "n_columns": len(cols),
        "n_model_features": sum(c.model_feature for c in cols),
        "model_feature_columns": [c.name for c in cols if c.model_feature],
        "provisional_feature_columns": [c.name for c in cols if c.role == "feature_provisional"],
        "parameters": parameters,
        "columns": [c.to_dict() for c in cols],
    }
