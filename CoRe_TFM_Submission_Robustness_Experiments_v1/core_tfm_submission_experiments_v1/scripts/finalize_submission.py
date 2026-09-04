from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

CONFIGS = {
    "minimal": ("minimal_submission.yaml", "results/core_tfm_submission_minimal_v1"),
    "full": ("full_protocol.yaml", "results/core_tfm_submission_full_v1"),
}

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--preset", choices=CONFIGS, default="minimal")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    bundle = Path(__file__).resolve().parents[1]
    cfg_name, rel_out = CONFIGS[args.preset]
    out = repo / rel_out
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final_submission_audit"
    final.mkdir(parents=True, exist_ok=True)

    # 1. Re-audit frozen evidence without altering it.
    audit_out = final / "frozen_evidence_audit.json"
    cmd = [
        sys.executable,
        str(repo / "experiments" / "audit_evidence_package.py"),
        "--results", str(repo / "results" / "q1_fast_complete_256_v1"),
        "--output", str(audit_out),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo, check=True)

    # 2. Read the repository-native completion gate. Presence alone is insufficient.
    gate_path = out / "FINAL_COMPLETION_GATE.json"
    gate_payload = None
    gate_pass = False
    if gate_path.exists():
        try:
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_pass = bool(gate_payload.get("complete") is True)
        except Exception:
            gate_payload = None
            gate_pass = False

    # 3. Hash all paper-facing analysis artifacts. Exclude executed notebooks/model caches
    #    to keep the manifest compact.
    include_suffixes = {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}
    hashes = {}
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        if "model_cache" in p.parts:
            continue
        if p.suffix.lower() not in include_suffixes:
            continue
        if p == final / "sha256_manifest.json":
            continue
        hashes[str(p.relative_to(out))] = sha256(p)

    manifest = {
        "preset": args.preset,
        "status": "NEW_COMPLETE" if gate_pass else "PARTIAL_DO_NOT_CLAIM",
        "output_root": str(out),
        "frozen_evidence_audit": str(audit_out),
        "completion_gate": str(gate_path) if gate_path.exists() else None,
        "completion_gate_pass": gate_pass,
        "completion_gate_missing": None if gate_payload is None else gate_payload.get("missing", []),
        "operationally_unsupported_but_recorded": None if gate_payload is None else gate_payload.get("operationally_unsupported_but_recorded", []),
        "hashed_artifacts": len(hashes),
        "claim_rule": "New numerical results are manuscript-admissible only when completion_gate_pass is true.",
    }
    (final / "FINAL_STATUS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (final / "sha256_manifest.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    if not gate_pass:
        print("\nThe repository-native completion gate did not report complete=true.")
        print("Status remains PARTIAL_DO_NOT_CLAIM.")
        raise SystemExit(2)
    print("\nCompletion gate reports complete=true. New package status is NEW_COMPLETE.")

if __name__ == "__main__":
    main()
