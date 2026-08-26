from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    import torch

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "amd_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda_available_api": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {},
    }

    if torch.version.hip is None:
        raise RuntimeError("PyTorch is not a ROCm/HIP build.")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is not visible through torch.cuda.is_available().")

    # ROCm uses the torch.cuda Python API. Avoid optional NVIDIA-only paths.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(root / ".model-cache" / "tabpfn"))

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TABPFN_TOKEN in the environment before running preflight.py")

    from core_tfm.models.tfm_adapters import tabiclv2_adapter, tabpfn3_adapter

    X = pd.DataFrame(
        {
            "x1": np.linspace(-1.0, 1.0, 30),
            "x2": np.sin(np.linspace(0.0, 3.0, 30)),
            "x3": np.arange(30) % 5,
        }
    )
    y = pd.Series(np.arange(30) % 3)
    X_train, X_test = X.iloc[:24], X.iloc[24:]
    y_train = y.iloc[:24]

    checks = {
        "tabiclv2": lambda: tabiclv2_adapter(
            device="cuda",
            n_estimators=1,
            kv_cache=False,
            random_state=11,
            n_jobs=1,
            verbose=False,
        ),
        "tabpfn3": lambda: tabpfn3_adapter(
            device="cuda",
            random_state=11,
        ),
    }

    failures: list[str] = []
    for name, factory in checks.items():
        started = time.time()
        try:
            model = factory().fit(X_train, y_train)
            p = model.predict_proba(X_test)
            if p.shape[0] != len(X_test):
                raise RuntimeError(f"unexpected probability shape {p.shape}")
            if not np.isfinite(p).all():
                raise RuntimeError("non-finite probabilities")
            report["models"][name] = {
                "ok": True,
                "seconds": time.time() - started,
                "shape": list(p.shape),
                "row_sum_min": float(p.sum(axis=1).min()),
                "row_sum_max": float(p.sum(axis=1).max()),
            }
        except Exception as exc:
            report["models"][name] = {
                "ok": False,
                "seconds": time.time() - started,
                "error": repr(exc),
            }
            failures.append(name)

    report_path = out_dir / "amd_gpu_preflight.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", report_path)

    if failures:
        raise SystemExit(
            "AMD preflight failed for: " + ", ".join(failures) + ". "
            "Do not start the full benchmark until both primary TFMs pass."
        )

    print("AMD PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
