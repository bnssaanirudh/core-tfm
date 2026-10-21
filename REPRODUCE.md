# Reproduce CoRe-TFM S2 evidence

## Current evidence release

The completed S2 protocol and its source artifacts are stored in
[`CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21/`](CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21/).
The public S2 dataset is deposited at
[Zenodo DOI 10.5281/zenodo.23507196](https://doi.org/10.5281/zenodo.23507196).

The primary matrix contains 10 datasets × 2 TFMs × 4 held-out split settings × 2
training caps (128 and 256), totaling **160 configurations**. The four split
settings are repeated tests within each dataset, not 40 independent datasets. Each
configuration has an NPZ prediction cache and a JSON metadata sidecar. The release
also contains 40 frozen split specifications. The primary analysis setting uses a
training cap of 256, a validation selection budget of 52, and 128 held-out test
examples. Model package identifiers recorded in the evidence are TabICLv2 2.2.0
and TabPFN 9.2.0 (V3).

Six targeted clean environment reruns are included for Car, Nursery, and Diamonds
with both TFMs (seed 23, training cap 256). They verify cache generation in a clean
environment; they are not six replications of the whole 160-configuration matrix.
The detailed run events, environment record, split files, and checksums are under
`03_RESULTS/REPLICATION_AND_AUDITS/UNPACKED_SIX_REPLICATIONS/`.

## Validate the released caches

From the repository root, run:

```bash
python experiments/validate_s2_evidence.py
```

The check validates the 160 primary NPZ caches and their 160 JSON sidecars against
`COMBINED_SHA256_MANIFEST.json`, then validates the six clean-run caches and their
sidecars against `NEW_REPLICATION_SHA256_MANIFEST.json`. It fails on a missing
cache, missing sidecar, unlisted cache file, incorrect file size, or SHA-256
mismatch. GitHub Actions runs this check on pushes and pull requests.

To validate a copy of the evidence package at another path:

```bash
python experiments/validate_s2_evidence.py --package-root /path/to/CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21
```

## Run software checks

Requires Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[test]'
pytest -q
python experiments/smoke_synthetic.py
python experiments/validate_s2_evidence.py
```

The cache validation does not download model checkpoints or rerun inference. To
reproduce model inference separately, follow the pinned environment, checkpoint,
and license information in the S2 evidence package; never place access tokens in
notebooks, logs, or the repository.

## Historical five-fold benchmark

The earlier bounded-context five-fold study is retained under
`results/q1_fast_complete_256_v1/` and `notebooks/`. Its audit command is:

```bash
PYTHONPATH=src python experiments/analyze_q1_benchmark.py
```

That study is historical context and is not the S2 protocol. The old README files
inside the dated recovery collection are preserved as source artifacts so their
recorded checksums remain valid. See the package-root `CURRENT_STATUS.md` for the
current status and how to interpret those older notes.

## Claim guardrails

- Report the four split settings as repeated tests within datasets; use datasets as
  the independent units for dataset-level inference.
- Keep the six targeted clean reruns distinct from independent full-matrix
  replications.
- Do not infer checkpoint-weight byte identity from a checkpoint identifier alone;
  the evidence package records the outstanding checkpoint-hash limitation.
- Do not treat reduced factorization inconsistency as evidence of improved
  predictive accuracy.
