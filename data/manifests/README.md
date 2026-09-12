# Manifests

Small, versioned files required to reproduce a result:

- raw_checksums_<dataset>.json   SHA-256 of every raw file (wsr.utils.integrity)
- splits_<dataset>.json          participant-level outer/inner splits per seed
- exclusions_<dataset>.csv       excluded participants/sessions with reasons

These ARE committed (see .gitignore).
