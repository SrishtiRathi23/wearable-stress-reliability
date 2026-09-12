# Dataset Notes

Everything here must be **verified from the actual files and official documentation** by the audit scripts (`src/wsr/data/audit_*.py`) before it is treated as FACT. Until then each section is a checklist of what must be established. Do not fill values from memory.

Sources to cite when filling in: official dataset readme / data descriptor paper / repository page, with URL and access date.

---

## WESAD (Schmidt et al. 2018)

**Source:** TODO (URL, access date, archive checksum -> `data/manifests/raw_checksums_wesad.json`)

To verify:
- [ ] Participant ids present in the release; number of usable participants; any participants the authors excluded and why.
- [ ] Devices and signal streams per device (chest RespiBAN, wrist Empatica E4) and their sampling rates.
- [ ] Label vector: integer codes and their meaning (transient/baseline/stress/amusement/meditation and any others); sampling rate of the label vector.
- [ ] Duration of each protocol condition per participant (needed for the episode-definition decision T-05 and the class balance).
- [ ] Alignment between wrist and chest streams and the label vector (which clock the labels follow).
- [ ] Self-report questionnaires available and whether they are relevant (probably not for the core study).
- [ ] Any known data-quality issues reported in the paper or by other users.

Findings: _none yet_

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
