import hashlib
import json
import sys
from pathlib import Path

# The validator is an executable under experiments, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.validate_s2_evidence import audit_cache_bundle


def write_manifest(root: Path, entries: list[tuple[str, bytes]]) -> Path:
    rows = []
    for relative_path, content in entries:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return manifest


def test_audits_every_cache_and_sidecar_against_manifest(tmp_path: Path) -> None:
    evidence = tmp_path / "bundle"
    manifest = write_manifest(
        evidence,
        [
            ("real_caches/example.npz", b"cache bytes"),
            ("real_caches/example.json", b"metadata bytes"),
            ("reports/readme.txt", b"outside cache scope"),
        ],
    )

    report = audit_cache_bundle("fixture", evidence, manifest, expected_npz_count=1)

    assert report["ok"] is True
    assert report["cache_count"] == 1
    assert report["checked_cache_files"] == 2
    assert report["errors"] == []


def test_fails_when_cache_bytes_do_not_match_uploaded_manifest(tmp_path: Path) -> None:
    evidence = tmp_path / "bundle"
    manifest = write_manifest(
        evidence,
        [("real_caches/example.npz", b"original"), ("real_caches/example.json", b"metadata")],
    )
    (evidence / "real_caches/example.npz").write_bytes(b"changed")

    report = audit_cache_bundle("fixture", evidence, manifest, expected_npz_count=1)

    assert report["ok"] is False
    assert any("SHA-256 mismatch" in error for error in report["errors"])


def test_fails_for_missing_manifest_entry_or_unpaired_cache(tmp_path: Path) -> None:
    evidence = tmp_path / "bundle"
    manifest = write_manifest(evidence, [("real_caches/example.npz", b"cache")])
    (evidence / "real_caches/unlisted.npz").write_bytes(b"extra cache")

    report = audit_cache_bundle("fixture", evidence, manifest, expected_npz_count=1)

    assert report["ok"] is False
    assert any("unlisted cache" in error for error in report["errors"])
    assert any("missing JSON sidecar" in error for error in report["errors"])


def test_rejects_manifest_paths_that_escape_bundle_root(tmp_path: Path) -> None:
    evidence = tmp_path / "bundle"
    evidence.mkdir()
    manifest = evidence / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": "../outside.npz",
                    "sha256": "0" * 64,
                    "bytes": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = audit_cache_bundle("fixture", evidence, manifest, expected_npz_count=0)

    assert report["ok"] is False
    assert any("escapes evidence root" in error for error in report["errors"])
