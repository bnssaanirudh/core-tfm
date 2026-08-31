from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .common import json_dump, resolve_paths


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Bundle final paper evidence after the suite finishes")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    paths = resolve_paths(args.repo, args.config)

    out_zip = Path(args.output).resolve() if args.output else (paths.output_root / "CoRe_TFM_FINAL_PAPER_EVIDENCE.zip")
    include_names = {
        "PREFLIGHT.json",
        "MATRIX_STATUS.json",
        "FINAL_COMPLETION_GATE.json",
        "COMPLETE.json",
        "RUN_STATUS.json",
        "environment_metadata.json",
        "protocol.json",
        "protocol_amendment.json",
        "selection_ablations.csv",
        "validation_fraction_sensitivity.csv",
        "fold_results.csv",
        "fold_failures.json",
    }

    files = []
    for p in paths.output_root.rglob("*"):
        if not p.is_file() or p == out_zip:
            continue
        rel = p.relative_to(paths.output_root)
        if "analysis" in rel.parts or "cpu_experiments" in rel.parts or p.name in include_names:
            files.append(p)

    manifest = []
    for p in sorted(files):
        manifest.append({
            "path": str(p.relative_to(paths.output_root)).replace("\\", "/"),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    manifest_path = paths.output_root / "PAPER_BUNDLE_MANIFEST.json"
    json_dump(manifest_path, {"files": manifest})
    files.append(manifest_path)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(set(files)):
            zf.write(p, arcname=str(p.relative_to(paths.output_root)).replace("\\", "/"))
    print(out_zip)


if __name__ == "__main__":
    main()
