# CoRe-TFM AMD/ROCm robustness runner

This folder is the recommended replacement for the fragile Colab robustness notebooks when you have access to a Linux AMD GPU with a working ROCm PyTorch stack.

The design deliberately keeps the proven Q1 inference implementation. For each robustness variant it generates a **local AMD-compatible copy** of `notebooks/CoRe_TFM_Q1_FAST_COMPLETE_256_Colab.ipynb`, replaces only the Colab setup/protocol constants, truncates after Cell 12E, and executes it with a real Jupyter kernel using `nbclient`. The original Q1 frozen result directory is never overwritten.

## Important compatibility note

PyTorch officially exposes ROCm GPUs through the `torch.cuda` Python API, so the existing CoRe-TFM factories that select `device="cuda"` when `torch.cuda.is_available()` is true can use ROCm. TabPFN itself recommends installing the accelerator-specific PyTorch build before installing TabPFN, but ROCm compatibility can still depend on the exact AMD GPU, ROCm, PyTorch, and TabPFN kernels. Therefore **both TabICLv2 and TabPFN-3 must pass `preflight.py` before the benchmark starts**.

## 1. Requirements

Use Linux with an AMD GPU supported by ROCm. Python 3.10+ is recommended. Install the ROCm version supported by your GPU/driver, then install the matching PyTorch ROCm wheel using the official PyTorch selector.

Do not install a CUDA/NVIDIA PyTorch wheel on this machine.

## 2. Clone/update the repository

```bash
git clone https://github.com/bnssaanirudh/core-tfm.git
cd core-tfm
git checkout main
git pull
```

## 3. Install ROCm PyTorch

The safest route is to obtain the exact command from:

https://pytorch.org/get-started/locally/

Choose **Linux → Pip → Python → ROCm** for the ROCm version supported by your machine.

You can either install PyTorch yourself first, or let the setup script install from an index URL you explicitly provide.

Example pattern only — change the ROCm URL to the version supported by your system:

```bash
export INSTALL_ROCM_TORCH=1
export ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm6.4
bash amd_gpu_runner/setup_amd.sh
```

If PyTorch ROCm is already installed:

```bash
bash amd_gpu_runner/setup_amd.sh
```

The script creates `.venv-amd`, installs the project and frozen research dependencies, and refuses to continue if `torch.version.hip` is missing or the GPU is not visible.

## 4. TabPFN authentication

Create/obtain your Prior Labs TabPFN token and export it:

```bash
export TABPFN_TOKEN='YOUR_TOKEN_HERE'
```

The token is read from the environment and is **not written into the repository**.

## 5. Mandatory AMD model preflight

```bash
source .venv-amd/bin/activate
python -m amd_gpu_runner.preflight
```

Success ends with:

```text
AMD PREFLIGHT: PASS
```

The report is stored at:

```text
results/amd_preflight/amd_gpu_preflight.json
```

Do not start the full experiment if either `tabiclv2` or `tabpfn3` reports `ok: false`. That means the specific AMD/ROCm stack is not compatible yet; the error is preserved in the JSON report.

## 6. Run one resumable robustness variant

Recommended first run:

```bash
python -m amd_gpu_runner.run_queue --max-variants 1
```

The first pending variant is run. Completed folds are checkpointed inside:

```text
results/amd_jmlr_robustness_v1/
```

If the process stops or the machine reboots, run the same command again. The Q1 shard code reads the existing `fold_results.csv` and skips completed dataset/model/fold tasks.

## 7. Run all variants

After the first variant works correctly:

```bash
python -m amd_gpu_runner.run_queue --all
```

Or only one experiment family:

```bash
python -m amd_gpu_runner.run_queue --group multi_seed --all
python -m amd_gpu_runner.run_queue --group context_size --all
```

## 8. One-command launcher

After setup and after exporting `TABPFN_TOKEN`:

```bash
bash amd_gpu_runner/run.sh --max-variants 1
```

or:

```bash
bash amd_gpu_runner/run.sh --all
```

The launcher runs the AMD preflight first on every invocation and starts the queue only if it passes.

## 9. What each variant contains

Every real-TFM robustness variant uses:

- 10 benchmark datasets
- TabICLv2 + TabPFN-3 as the primary TFMs
- CatBoost as the boundary baseline
- 5 outer folds
- the same 8 evaluated methods as the Q1 benchmark
- validation-only Selective CoRe policy choice
- per-fold checkpointing
- a source-commit freeze in `frozen_source_revisions.json`
- generated and executed `.ipynb` files for auditability

The multi-seed queue uses the seeds from `configs/reliability_aware_experiments.yaml`. The context-size queue uses the configured train sizes and context seeds from the same frozen experiment specification.

## 10. Output structure

Example:

```text
results/amd_jmlr_robustness_v1/
├── multi_seed/
│   ├── seed_11/
│   │   ├── fold_results.csv
│   │   ├── fold_manifests.json
│   │   ├── fold_failures.json
│   │   ├── frozen_source_revisions.json
│   │   ├── generated_amd_variant.ipynb
│   │   ├── executed_amd_variant.ipynb
│   │   └── COMPLETE.json          # only after 150 fold cells / 1200 method rows are complete
│   └── ...
├── context_size/
│   └── ...
├── QUEUE_STATUS.json
└── ROBUSTNESS_STATUS.json
```

## 11. If a variant fails

Check, in this order:

```text
results/amd_jmlr_robustness_v1/<group>/<variant>/EXECUTION_ERROR.json
results/amd_jmlr_robustness_v1/<group>/<variant>/fold_failures.json
results/amd_jmlr_robustness_v1/<group>/<variant>/executed_amd_variant.ipynb
```

`EXECUTION_ERROR.json` is an orchestration/Jupyter-cell failure. `fold_failures.json` contains model/data failures caught by the original resumable Q1 shard runner.

Do not delete successful `fold_results.csv` rows just because one later fold fails; the point of this runner is to resume them.

## 12. Quick environment checks

```bash
source .venv-amd/bin/activate
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('GPU visible:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

For a real AMD ROCm environment, `torch.version.hip` must not be `None` and `torch.cuda.is_available()` must be `True`.
