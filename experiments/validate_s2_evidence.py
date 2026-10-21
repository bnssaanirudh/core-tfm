"""Validate S2 prediction-cache files against their packaged SHA-256 manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


PACKAGE_NAME = "CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_cache_paths(manifest: list[dict[str, Any]], root: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for row in manifest:
        relative = row.get("path")
        if not isinstance(relative, str):
            errors.append("manifest entry has no string path")
            continue
        normalized = PurePosixPath(relative)
        if normalized.is_absolute() or ".." in normalized.parts:
            errors.append(f"manifest path escapes evidence root: {relative}")
            continue
        if not relative.startswith("real_caches/"):
            continue
        if normalized.suffix not in {".npz", ".json"}:
            continue
        key = normalized.as_posix()
        if key in entries:
            errors.append(f"duplicate manifest path: {key}")
            continue
        entries[key] = row
    return entries


def audit_cache_bundle(
    label: str,
    evidence_root: Path,
    manifest_path: Path,
    expected_npz_count: int,
) -> dict[str, Any]:
    """Check cache counts, JSON sidecars, and all cache hashes against a manifest."""
    errors: list[str] = []
    evidence_root = evidence_root.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        manifest = []
        errors.append(f"cannot read manifest {manifest_path}: {exc}")
    if not isinstance(manifest, list):
        manifest = []
        errors.append(f"manifest must be a list of file records: {manifest_path}")

    entries = _manifest_cache_paths(manifest, evidence_root, errors)
    cache_root = evidence_root / "real_caches"
    actual_paths = {
        path.relative_to(evidence_root).as_posix(): path
        for path in cache_root.rglob("*")
        if path.is_file() and path.suffix in {".npz", ".json"}
    } if cache_root.exists() else {}
    npz_paths = {name for name in actual_paths if name.endswith(".npz")}
    json_paths = {name for name in actual_paths if name.endswith(".json")}
    manifested_npz = {name for name in entries if name.endswith(".npz")}
    manifested_json = {name for name in entries if name.endswith(".json")}

    if not cache_root.is_dir():
        errors.append(f"cache directory is missing: {cache_root}")
    for name in sorted(actual_paths.keys() - entries.keys()):
        errors.append(f"unlisted cache file: {name}")
    if len(npz_paths) != expected_npz_count:
        errors.append(f"expected {expected_npz_count} NPZ caches, found {len(npz_paths)}")
    for name in sorted(npz_paths):
        if name not in entries:
            errors.append(f"unlisted cache: {name}")
        sidecar = str(PurePosixPath(name).with_suffix(".json"))
        if sidecar not in actual_paths:
            errors.append(f"missing JSON sidecar for {name}")
        elif sidecar not in entries:
            errors.append(f"unlisted cache sidecar: {sidecar}")
    for name in sorted(manifested_npz | manifested_json):
        if name not in actual_paths:
            errors.append(f"manifested cache file is missing: {name}")

    checked = 0
    for name in sorted((manifested_npz | manifested_json) & actual_paths.keys()):
        row = entries[name]
        path = actual_paths[name]
        expected_hash = row.get("sha256")
        expected_bytes = row.get("bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"invalid SHA-256 in manifest for {name}")
            continue
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        checked += 1
        if actual_hash.lower() != expected_hash.lower():
            errors.append(f"SHA-256 mismatch for {name}")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            errors.append(f"byte-count mismatch for {name}: expected {expected_bytes}, found {actual_bytes}")

    return {
        "label": label,
        "evidence_root": str(evidence_root),
        "manifest": str(manifest_path),
        "expected_npz_count": expected_npz_count,
        "cache_count": len(npz_paths),
        "checked_cache_files": checked,
        "errors": errors,
        "ok": not errors,
    }


def validate_s2_evidence(package_root: Path) -> dict[str, Any]:
    results_root = package_root / "03_RESULTS"
    primary_root = results_root / "FINAL_160_CONFIGURATIONS" / "UNPACKED"
    replication_root = results_root / "REPLICATION_AND_AUDITS" / "UNPACKED_SIX_REPLICATIONS"
    bundles = [
        audit_cache_bundle(
            "S2 primary evidence (160 configurations)",
            primary_root,
            primary_root / "COMBINED_SHA256_MANIFEST.json",
            expected_npz_count=160,
        ),
        audit_cache_bundle(
            "six clean replication runs",
            replication_root,
            replication_root / "NEW_REPLICATION_SHA256_MANIFEST.json",
            expected_npz_count=6,
        ),
    ]
    return {"ok": all(bundle["ok"] for bundle in bundles), "bundles": bundles}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / PACKAGE_NAME,
        help="path to the extracted dated evidence package",
    )
    args = parser.parse_args()
    report = validate_s2_evidence(args.package_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
