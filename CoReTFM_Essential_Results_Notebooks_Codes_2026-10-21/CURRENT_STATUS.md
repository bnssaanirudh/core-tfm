# Current status of the CoRe-TFM evidence package

The October 21, 2026 source ZIP contained the **617-file essential recovery
collection**. This folder also contains this status note. The upload is not a claim
that every historical artifact from the larger 646-file conversation backup has
been transferred.

## Current evidence

- The S2 primary matrix contains 160 configurations: 10 datasets × 2 TFMs × 4
  held-out split settings × 2 training caps.
- The collection includes 160 NPZ caches, 160 JSON sidecars, and 40 frozen split
  specifications.
- Six targeted clean reruns cover Car, Nursery, and Diamonds with both TFMs at
  seed 23 and training cap 256. These reruns do not repeat the full matrix.
- The S2 data release is published at
  [Zenodo DOI 10.5281/zenodo.23507196](https://doi.org/10.5281/zenodo.23507196).
- The matrix's independent audit reports 160 valid configurations and zero audit
  errors. It also records that exact checkpoint-weight byte hashes were not
  verified.

Run `python experiments/validate_s2_evidence.py` from the repository root to verify
all primary and clean-rerun caches and JSON sidecars against their SHA-256
manifests.

## Reading the preserved files

README and audit files nested in this recovery collection are preserved byte-for-byte
from their source exports. Some describe earlier evidence states, including the
pre-S2 five-fold study or the initial 160-cache audit. Those files are historical
records, not current repository status. Use the top-level `README.md` and
`REPRODUCE.md` for current guidance. The separate `00_MANIFESTS/` hashes describe
the original recovery collection and are not rewritten by this status note.
