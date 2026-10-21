# CoRe-TFM — Full 160-cache combined evidence (2026-10-20)

This archive combines two independent Colab-produced evidence exports: v5 (Car/Credit/Wine/Phishing/Nursery) and the standalone next-only extension (Anneal/MIC/Customer/Marketing/Diamonds). The 160 real prediction caches preserve their original NPZ byte content and original model sidecars, and no new models were trained in combining them.

STRUCTURE
- real_caches/: 160 `.npz` plus 160 associated JSON metadata sidecars.
- splits/: 40 frozen split specifications; two training-context sizes share split data.
- datasets/: five cached processed dataset pickles, from new standalone extension only. IMPORTANT: loading any pickle requires trusting its origin; these are not raw OpenML downloads.
- source_reports/: original analyses and experiment coverage reports, separated by source stage.
- source_manifests/: unmodified source evidence checksum manifests.
- independent_audit/: fresh independent numerical audit of all 160 caches and dataset-blocked summaries.
- COMBINED_SHA256_MANIFEST.json: archive-wide checksums.

PRIMARY PROTOCOL
10 datasets x 2 models (TabICLv2 2.2.0 / TabPFN 9.2.0, V3) x 4 randomized held-out split settings x 2 training caps (128/256). Each split has 205 validation examples, 128 test examples; main validation selection budget 52. Randomized split settings are repeated tests within dataset, not 40 independent datasets.

LIMITATIONS
- This is completed experimental *data collection* and exploratory analysis, not a finished journal-submission replication.
- The exact checkpoint weight byte hashes have not been verified. Checkpoint identifiers alone do not prove bytes loaded.
- Processed dataset pickles are hashed, but original OpenML source revisions/bytes need separate confirmation.
- Separate Colab runs used the same named cache protocol and package identifiers, but protocol/code differences (including the Diamonds target pairing fix) must be disclosed and independently reproduced on clean pinned environments.
- Average metrics, exploratory dataset-blocked p-values, and observed consistency-vs-accuracy effects must not be interpreted as broad statistical proof or automatic improvements from reconciliation.
- Never tune method choices using test labels.

Use independent_audit/CoReTFM_160_Independent_Configuration_Results.csv as the full comparative starting point.
