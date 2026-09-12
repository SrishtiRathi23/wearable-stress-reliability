# Dataset Notes

Everything here must be **verified from the actual files and official documentation** before it is treated as FACT. Provenance tags: `[readme]` = the release file `WESAD/wesad_readme.pdf` (section number); `[audit]` = empirical result from `src/wsr/data/audit_wesad.py`, machine-readable in `data/manifests/wesad_audit.json`; `[source]` = `data/manifests/wesad_source.json`; `[paper]` = Schmidt et al. 2018 (not yet consulted directly - only the readme was read). Anything not tagged is INFERENCE and says so.

---

## WESAD (Schmidt et al. 2018) - audited 2026-09-12

### Release provenance `[source]`

- Official chain: UCI ML Repository entry (DOI 10.24432/C57K5T, "linked on 2018-09-13") -> University of Siegen dataset page (`https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html`) -> public sciebo share `HGdUkoNlW1Ub0Gx` (no authentication).
- Downloaded file: `WESAD.zip`, **2,249,444,501 bytes**, 92 zip members, `zipfile.testzip()` clean. Locally computed SHA-256 `5e15d260 6adf16d8 19ba535c 1786d92e 94bae5ff 5392f7ae 01fb63f9 938fd71c` (not a publisher checksum - **none is published**; checked UCI and Siegen pages on the access date).
- Terms of use (Siegen page): scientific, non-commercial use with credit to the owners. Citation requested: Schmidt, Reiss, Duerichen, Marberger, Van Laerhoven, ICMI 2018, doi:10.1145/3242969.3242985.
- No dataset version string is published anywhere in the chain; the archive hash above is the only version identifier we have.
- Local layout: `data/raw/wesad/WESAD.zip` (kept) and `data/raw/wesad/WESAD/` (extracted with `zipfile.extractall`, nothing renamed or re-encoded; inner `S<id>_E4_Data.zip` left unextracted). Raw checksum baseline: `data/manifests/raw_checksums_wesad.json`, 77 files.

### Participants

- FACT `[readme I.2]`: 17 subjects participated; S1 and S12 were discarded for sensor malfunction and are absent from the release.
- FACT `[audit]`: folders present = S2-S11, S13-S17 (**15**), exactly the readme's expectation; no unexpected folders. Every folder has all five documented files `[readme I.1]`, every `.pkl` is readable, the `subject` field matches the folder name.
- FACT `[audit]`: no participant has missing signals, empty arrays, NaN or Inf in any stream, or a chest/label length mismatch. All 15 are classified **apparently_usable** structurally. **No exclusion is proposed.**
- Per-subject readme notes `[S<id>_readme.txt]` that matter for interpretation (not structural defects): S2 and S17 - RespiBAN (chest) temperature sensor "not fully attached" throughout; S3 - baseline in a sunny workplace, self-reported valence 7 after stress; S5 - may have fallen asleep in meditation 1; S6 - reported not really stressed by the TSST (recent interview practice); S8 - stressful day before the study, felt cold during stress; S15 - did not believe the TSST cover story; S16 - felt cold during stress. Demographics: 12 male / 3 female (S8, S11, S17), ages 24-35, one left-handed (S8).

### Devices, signals, sampling rates

- FACT `[readme II.1]`: chest device RespiBAN Professional, all channels 700 Hz: ECG, EDA, EMG, TEMP, 3-axis ACC, RESPIRATION. Raw counts in `S<id>_respiban.txt` (columns nSeq, DI, CH1-CH8); SI conversion formulas given in the readme.
- FACT `[readme II.2]`: wrist device Empatica E4 on the non-dominant wrist: ACC 32 Hz (1/64 g), BVP 64 Hz, EDA 4 Hz (uS), TEMP 4 Hz (degC). `HR.csv`, `IBI.csv`, `tags.csv` in the E4 zip are derived and to be ignored.
- FACT `[audit]`: `S<id>.pkl` is a dict with keys `signal`, `label`, `subject` `[readme II.3]`. `signal.chest` keys are `ACC` (n,3), `ECG`, `EMG`, `EDA`, `Temp`, `Resp` (n,1); `signal.wrist` keys are `ACC` (m,3), `BVP`, `EDA`, `TEMP` (m,1). dtypes float64 except chest `Temp` float32. Every E4 csv header declares exactly the documented rate; every RespiBAN header declares 700 Hz.
- FACT `[audit]`: for all 15, every chest array has exactly the label-vector length, and every wrist array's length / nominal rate equals the label duration to within 0.25 s (float rate x duration). Value ranges are physically plausible (chest ECG +-1.5 mV, wrist ACC -128..127 counts, wrist TEMP ~32-36 degC).
- **Discrepancy (minor, naming only):** readme II.3 lists chest modalities as "RESP" and "TEMP"; the pickle keys are `Resp` and `Temp`.

### Label vector and codes (resolves the factual part of T-02)

- FACT `[readme II.3]`: `label` is sampled at 700 Hz; 0 = not defined/transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = "should be ignored".
- FACT `[audit]`: codes actually present: 0, 1, 2, 3, 4, 6, 7 in all 15; code 5 in 14/15 (absent in S2, whose `quest.csv` lists no `bRead` block). **No undocumented code occurs.** dtype int32.
- INFERENCE (consistent in all 15): codes 5/6/7 are the `bRead` / `sRead` / `fRead` reading blocks of `quest.csv` (their intervals match), i.e. the readme's "ignore" instruction is about reading tasks.
- FACT `[audit]`: per-code totals over 15 participants - baseline 293.5 min, stress 166.1 min, amusement 92.9 min, meditation 196.8 min, transient (0) 658.4 min, reading (5+6+7) 40.1 min. Total recording 24.1 h. **Protocol-labelled time (codes 1-4) is ~52 % of the recording; code 0 is ~45 %.**

### Condition durations and contiguous blocks (evidence for KI-05 / T-05)

FACT `[audit]` - every participant has **exactly one contiguous baseline run and exactly one contiguous stress run**; two meditation runs; one amusement run; 8-9 transient runs; the only runs shorter than 30 s are S13's `fRead` block (20 s) and one 26-s transient gap in S14.

| id | recording s | baseline s | stress s | amusement s | meditation s | transient s | chest crop s | E4 crop s | clock residual s | readme note |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 6079 | 1144 | 615 | 362 | 768 | 3061 | 131.453 | 1545.219 | +2.2 | chest temp sensor loose |
| S3 | 6493 | 1140 | 640 | 375 | 780 | 3351 | 61.146 | 1090.594 | +3.6 | sunny baseline; valence 7 after stress |
| S4 | 6423 | 1158 | 635 | 372 | 805 | 3306 | 76.706 | 1475.500 | +2.2 | |
| S5 | 6258 | 1198 | 645 | 374 | 794 | 3061 | 67.629 | 1254.312 | +1.3 | may have slept in meditation 1 |
| S6 | 7071 | 1180 | 650 | 372 | 787 | 3905 | 44.304 | 1206.500 | +0.8 | reported not stressed by TSST |
| S7 | 5238 | 1186 | 640 | 372 | 790 | 2103 | 43.369 | 1179.000 | +1.4 | |
| S8 | 5466 | 1169 | 670 | 370 | 796 | 2309 | 87.469 | 1128.562 | -0.1 | stressful day before; cold |
| S9 | 5223 | 1180 | 645 | 372 | 793 | 2051 | 42.754 | 961.531 | +1.2 | |
| S10 | 5496 | 1180 | 725 | 372 | 796 | 2270 | 87.143 | 1280.844 | -0.7 | |
| S11 | 5233 | 1180 | 680 | 368 | 791 | 2062 | 67.376 | 1188.094 | +1.3 | |
| S13 | 5537 | 1180 | 664 | 382 | 795 | 2399 | 70.093 | 1262.656 | +1.4 | |
| S14 | 5548 | 1180 | 675 | 372 | 794 | 2355 | 103.804 | 1393.375 | +0.4 | |
| S15 | 5252 | 1175 | 686 | 372 | 794 | 2082 | 164.770 | 1307.344 | +2.4 | did not believe TSST cover story |
| S16 | 5631 | 1180 | 673 | 368 | 792 | 2453 | 91.630 | 1386.062 | +1.6 | cold during stress |
| S17 | 5920 | 1181 | 723 | 372 | 731 | 2739 | 99.766 | 1277.969 | +0.8 | chest temp sensor loose |

- Baseline runs: 1140-1198 s (median 1180) -> **19** non-overlapping 60-s windows per participant (285 total).
- Stress runs: 615-725 s (median 664) -> **10-12** windows per participant (160 total).
- Baseline:stress window ratio ~ 1.8:1.
- **Window counts depend on the construction rule; neither number below is "the" sample size until the Phase-2 grid is built:**
  - **445** = complete 60-s windows obtained by restarting windowing separately at each true baseline/stress run boundary (285 baseline + 160 stress). This construction uses reference-label timing and therefore will **not** be the primary label-independent candidate construction.
  - **434** = complete, label-homogeneous baseline/stress windows on a fixed 60-s grid anchored at synchronised pickle time t=0 (282 baseline + 152 stress), independently recomputed from `wesad_audit.json` on 2026-09-12; 60 further grid windows straddle a baseline/stress boundary and are ineligible for the primary binary task (they are kept in provenance, not relabelled). This time-only grid is the approved Phase-2 primary frame (D-021).
- Condition order `[quest.csv line 2]`: 8 participants Base > Fun > Medi 1 > TSST > Medi 2 (S4, S5, S7, S8, S10, S13, S15, S17); 6 participants Base > TSST > Medi 1 > Fun > Medi 2 (S2, S3, S6, S9, S11, S16); S14's file names its meditation blocks "Medi 2" then "Medi 1" (both are code 4; naming quirk only). Baseline is always first.

### Alignment (resolves "which timeline do labels follow")

- FACT `[readme II, III.1]`: labels are synchronised with the RespiBAN raw data (same start); `quest.csv` START/END are in `minutes.seconds` **counted from the raw RespiBAN recording start**; the E4 was synchronised to the RespiBAN by the authors using a double-tap gesture, and `S<id>.pkl` holds the synchronised result.
- FACT `[audit]`: for all 15, the pkl **chest ECG** matches the raw `respiban.txt` ECG channel (after the readme's mV conversion) over its full length at a unique offset, with max absolute error 1.1e-16 mV (rtol = 0, atol = 1e-9), starting `chest crop s` into the raw file (42.8-164.8 s), with 55-215 s of raw data after the pkl end. INFERENCE: the pkl chest streams are a crop of the raw RespiBAN recording at that offset and **pkl time = raw RespiBAN time - crop**, so the label vector is on the RespiBAN timeline. **Only the ECG channel was compared**; chest EDA/EMG/Temp/Resp were not individually checked against the raw file.
- EMPIRICAL FINDING `[audit]` (not a documented release rule): after applying that crop, the relationship between `quest.csv` START/END and the label runs of codes 1-4 is consistent with **approximately 10 seconds trimmed at each scheduled-condition boundary**, agreeing within one 700-Hz label sample (~1.43 ms) across all 75 checked condition boundaries (15 participants x 5 conditions; residuals 0 or -1 sample at both start and end). Exact sample-level values are retained in `wesad_audit.json` (`schedule_vs_labels`); seconds are rounded for display only. The readme does not state this trim.
- FACT `[audit]`: for all 15, the pkl **wrist ACC** (integer-valued on both sides, verified before comparison) matches the raw E4 `ACC.csv` over its full length at a unique offset (crop 961-1545 s into the E4 recording, which started earlier than the RespiBAN). INFERENCE: the pkl wrist streams are a crop of the raw E4 recording at that offset. **Only ACC was compared**; wrist BVP/EDA/TEMP were not individually checked against their csv files (their lengths match the ACC-implied duration at the documented rates).
- INFERENCE `[audit]`: wall-clock of pkl t=0 from the RespiBAN header (device-local time, 1-s resolution) vs from the E4 unix timestamp (UTC) differ by 2 h + (-0.7 ... +3.6 s) for all 15; 2 h is CEST-UTC in Germany in May-Aug 2017. The authors' double-tap synchronisation is therefore consistent with the two devices' independent clocks to within a few seconds; we cannot verify it more finely without re-deriving the double-tap alignment, and we do not attempt to.
- `quest.csv` time format trap: `7.08` means 7 min 08 s; `50.3` means 50 min 30 s. Implemented in `audit_wesad.parse_min_sec` with tests.

### Data-quality findings

- No NaN/Inf, no empty arrays, no length mismatches, no undocumented codes, no missing streams in any participant `[audit]`.
- Chest `Temp` for S2 and S17 is documented as unreliable by the authors (sensor loose) - flag for any later chest-temperature feature.
- Self-reports and readme notes indicate the stress manipulation may have been weak for S6 and S15 (FACT that the notes exist; whether it affects physiology is not assessed here and must not be used to drop them without a pre-declared rule).
- Code 0 ("transient/undefined") is ~40 min per participant. It is **not** verified non-stress; it must be treated as excluded/unknown, never as a negative class (same logic as the Nurse invariant).

### Questionnaires `[readme III.2]`

- After each of the five conditions: PANAS (24 items incl. 4 added), STAI-6, SAM valence/arousal; after stress additionally SSSQ-6. Stored in `S<id>_quest.csv`. Not parsed by the audit (not needed for the structural study); available for later supporting analysis.

### Doc-vs-file discrepancies found

1. Chest pickle keys `Temp`/`Resp` vs readme "TEMP"/"RESP" (naming only).
2. Readme does not document the 10-s trim between `quest.csv` intervals and label runs (empirical, consistent).
3. Readme does not state that the pkl is a crop of the raw files (empirical; verified on chest ECG and wrist ACC only, inferred for the other modalities).
4. S14 quest names meditation blocks in swapped order (no effect on codes).
5. S2 has no `bRead` block and hence no code 5 (consistent between quest and labels; not an error).

### Open questions this audit does NOT settle

- ~~T-01 device choice~~ - resolved after review as a DESIGN choice (D-021: wrist E4 primary, chest optional sensitivity). The audit itself gave no structural reason to prefer one.
- T-05 selection unit: see the block evidence above and `known_issues.md` KI-05 - protocol blocks give exactly 2 relevant blocks per participant, which makes block-level selection an extremely coarse and poorly representative annotation model (not an impossibility).
- Whether code 3 (amusement) or 4 (meditation) should ever enter the study: excluded under the proposed initial mapping; any inclusion is a pre-declared amendment.

---

## Nurse Stress dataset (Hosseini et al. 2022)

**Source:** TODO (URL, access date, checksum manifest)

To verify (Gate 1):
- [ ] File layout: per-nurse / per-session signal files; which E4 streams exist (EDA, BVP, HR, IBI, TEMP, ACC) and their sampling rates.
- [ ] Timestamp conventions: unit (s / ms), epoch, timezone, and how E4 file headers encode start time and rate.
- [ ] Survey / event records: columns, label values and their exact meaning, whether labels apply to intervals or instants, how intervals were obtained (detector + validation).
- [ ] Number of nurses, sessions per nurse, total recording duration, labelled duration, validated-event counts per nurse and per label value.
- [ ] Fraction of shift time that is labelled vs unlabelled (this determines what Study D can say).
- [ ] Any context / stressor category fields and their counts.
- [ ] Any known issues (missing streams, clock drift, duplicated files).

Findings: _none yet_

---

## Stress-Predict (Iqbal et al. 2022)

**Source:** TODO

To verify (only when the replication stage is reached):
- [ ] Participants, protocol tasks and their order/durations, labelling scheme.
- [ ] Signal streams and sampling rates.
- [ ] Timestamp/label alignment.

Findings: _none yet_
