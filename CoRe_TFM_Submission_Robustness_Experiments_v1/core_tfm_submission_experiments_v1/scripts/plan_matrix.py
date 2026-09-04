from __future__ import annotations
import argparse, csv, json
from pathlib import Path

PRESETS = {
    "minimal": {
        "multi_seed": {"seeds": [11, 23, 42], "train": 256},
        "context": {"seeds": [23, 42], "trains": [256, 512, 1024]},
        "config": "minimal_submission.yaml",
    },
    "full": {
        "multi_seed": {"seeds": [11, 23, 42, 71, 101], "train": 256},
        "context": {"seeds": [23, 42, 71], "trains": [64, 128, 256, 512, 1024]},
        "config": "full_protocol.yaml",
    },
}

def planned(preset):
    p = PRESETS[preset]
    rows = []
    seen = set()
    for seed in p["multi_seed"]["seeds"]:
        key = (seed, p["multi_seed"]["train"])
        rows.append(("multi_seed", *key))
        seen.add(key)
    for seed in p["context"]["seeds"]:
        for train in p["context"]["trains"]:
            key = (seed, train)
            if key not in seen:
                rows.append(("context", *key))
                seen.add(key)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="minimal")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rows = planned(args.preset)
    payload = {
        "preset": args.preset,
        "gpu_notebook_variants": len(rows),
        "variants": [
            {"group": g, "seed": s, "train_limit": t, "test_limit": 128}
            for g, s, t in rows
        ],
        "note": "A variant executes the repository's bounded notebook matrix. This file is a plan, not evidence."
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["group","seed","train_limit","test_limit"])
                w.writeheader()
                w.writerows(payload["variants"])
        else:
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
