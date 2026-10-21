# CoRe-TFM: Clean replication checklist

1. Use `CoReTFM_FULL160_Combined_Verified_Evidence.zip` as the required INPUT for v8; the other v7 audit-output archive is not an input.
2. Open `CoReTFM_v8_Clean_Replication_and_Provenance_No_Drive.ipynb` on Colab with a new Google account; no Drive mount.
3. Run CPU archive verification. Set `RUN_ONLINE_OPENML_PROVENANCE=True` to authenticate source versions/raw bytes online. OpenML may limit requests.
4. On T4 GPU, accept TabPFN-3 weight license, configure `TABPFN_TOKEN` as Colab Secret and set `TABPFN_LICENSE_ACCEPTED=True`. Then set `RUN_GPU_REPLICATION=True`.
5. Run six genuinely new configurations: Car, Nursery, Diamonds at context 256, seed23/fold0 for TabICLv2 and TabPFN-3. Label/split alignment is a hard gate.
6. Download resulting `CoReTFM_NEW_CLEAN_REPLICATION_REPORTS.zip`. Inspect new checkpoint paths/hashes and paired score differences. Do not replace old evidence files with replicated predictions.
7. Keep unverified historical checkpoint hashes as NOT_VERIFIED forever unless genuine execution-time provenance emerges.
8. Do not publish S2 until authors approve and dataset/model license review is complete. Historical S1 is not included.
