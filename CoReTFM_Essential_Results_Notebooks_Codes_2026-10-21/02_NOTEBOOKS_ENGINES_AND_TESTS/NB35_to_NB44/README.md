# CoRe-TFM v2 — robust independent experimental notebooks (19 October 2026)

## What is in this bundle

Ten **independently bootstrapped** notebooks (`NB35.ipynb`–`NB44.ipynb`), the transparent engine source (`coretfm_v2_engine.py`), and local unit tests. Every notebook embeds a SHA-256-verified compressed copy of that engine so it can be launched directly in Colab even after a runtime disconnect without re-uploading Python modules. It tries to mount **Google Drive** and stores intermediate evidence under `MyDrive/CoRe_TFM_Resumable_v2/`. If Drive is unavailable it prints a warning and uses a local ephemeral folder. An explicit `COREFM_OUT` environment variable overrides the output root.

1. **NB35** — environment, mathematical invariants, and pinned OpenML paired-target splits (CPU).
2. **NB36** — one authentic TabICLv2 four-view GPU pilot (defaults Car/seed23/fold0/train64).
3. **NB37** — corresponding TabPFN **version 3** GPU pilot (explicit `ModelVersion.V3`).
4. **NB38** — optional TabFM third-model terms/API preflight, opt-in and no weights downloaded by default.
5. **NB39** — resumable one-job-per-execution GPU queue; repeat across Colab sessions.
6. **NB40** — fixed-test two-factor training-cap × validation-budget study, model-specific NLL/Brier, dataset-blocked uncertainty.
7. **NB41** — honest partial/complete-pilot evidence integrity gate and ZIP generator.
8. **NB42** — end-to-end model wall-time/cost measurements *from actual recorded GPU inference* (not synthetic timings).
9. **NB43** — separate known-truth **synthetic-only** reliability/validation-shift study, eight-repeat CPU shards.
10. **NB44** — evidence quality/manuscript claim audit.

## Correct order for Colab Free

- Upload these notebooks to Colab and open NB35. Set **Runtime → Change runtime type → T4 GPU** for NB36, NB37 and NB39. NB35, NB38 (disabled), NB40–NB44 can run on CPU.
- First run **NB35** to freeze Car/seed23/fold0. In Colab it downloads pinned OpenML data over the network. The first download needs working internet.
- Run **NB36** on a real GPU. Confirm output `REAL_MODEL_CACHE_WRITTEN` or `CACHED_VERIFIED`. The first run may need to download TabICLv2 weights.
- Run **NB37** on the same GPU class. It explicitly selects TabPFN-3, not any newer default. Accept the Prior Labs license when required. Store a `TABPFN_TOKEN` only in Colab Secrets (key `TABPFN_TOKEN`) if the official client needs it.
- Only if both pilots actually produce `.npz` **and** matching `.json` sidecars, advance to NB39. Each NB39 invocation computes **at most one new valid cache**, so you avoid a 90-minute notebook cell and can resume after a disconnect. Increase the matrix deliberately after pilot confidence. The queue currently enumerates 3 datasets × 2 models × 2 seeds × 1 fold × 3 training caps = 36 independent tasks.
- NB40, NB42, NB44 are CPU data-processing stages. NB41 creates a separate status and ZIP, but after successful GPU experiments run it again to include the latest evidence. NB43 can run any time on CPU; reruns append an 8-repeat deterministic shard rather than discarding earlier results.

## Scientific reproducibility constraints

The v2 experiment is **not** an exact rerun of the historical CoRe-TFM 17-run manuscript benchmark. The old main-branch benchmark used older package versions and different split and sample allocation. This package uses `tabicl==2.2.0` and `tabpfn==9.2.0`, which were the versions shown successfully installed in the latest error log. Model checkpoint identifiers are recorded, but **weight-byte SHA-256 hashes are not automatically certified**. Do not claim that the code proves checkpoint provenance until the model weights are independently authenticated.

Each cache records dataset, source identifiers, seed, fold, fixed train/validation/test sample IDs, model/package version, checkpoint identifier, raw probability arrays, wall-clock timings, and a SHA-256 checksum. An analysis should only use `.npz` files with matching JSON metadata, and never mix smoke caches with true TFM evidence. The local engine tests use **scikit-learn** only as a code-contract smoke fixture; those outputs are placed in `smoke_caches`, and NB40 refuses to treat them as true TFM experimental results.

Methods compared in the new CPU analysis: arithmetic pooling, both raw chain-rule orders, geometric pooling, hard marginal projection, and three soft reconciliation strengths. The new **eight-method** family is *not* the historical 48-configuration family. A predeclared regularization gate uses `beta = 0.5`; this is a provisional research policy, not a statistically guaranteed safety bound. No test labels are used in candidate selection. The fixed-test factorial independently varies training-cap and validation-budget; dataset-blocked confidence intervals are exploratory for small dataset counts. Results for one Car fold **must not** be presented as statistically significant evidence across datasets.

TabFM weights have a distinct `tabfm-non-commercial-v1.0` license. NB38 does not bypass it or claim that a third model was successfully evaluated. You must review the terms and opt in yourself. Model checkpoints may require acceptance of upstream terms and network authentication.

**No-GPU limitations:** In a CPU runtime, NB36, NB37 and NB39 report a transparent `SKIPPED_NO_GPU` status instead of raising the exact errors seen earlier. NB40 reports `BLOCKED_NO_REAL_CACHE` until authentic model data exists. NB41 produces `CoReTFM_V2_PARTIAL_EVIDENCE.zip` unless the modest pilot gate passes. Neither means the final manuscript is publication-ready.

## Local verification (conducted without real TFMs)

```bash
pip install numpy pandas scipy scikit-learn pytest nbformat nbclient
python -m pytest -q tests/test_engine.py
```

Four local tests passed: frozen split non-overlap, conditional-shape/cache integrity using a smoke estimator, probability/marginal contracts, and synthetic-shift output creation. All 10 notebooks were also executed in CPU mode with zero notebook-cell errors. GPU checkpoint loading and full real-model inference have **not** been executed in the artifact-generation environment.

## Source references and compatibility

- Own baseline implementation: https://github.com/bnssaanirudh/core-tfm
- TabICLv2 model documentation: https://github.com/soda-inria/tabicl
- TabPFN explicit V3 model API: https://github.com/PriorLabs/TabPFN
- TabFM model and license: https://github.com/google-research/tabfm

The bundled engine is new and fully inspectable; it ports the original paired-target OpenML dataset specification and four-view extraction logic while improving persistence, selection separation and failure reporting. It has **not** been merged to the user's GitHub repository.
