#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src pytest -q
PYTHONPATH=src python experiments/smoke_synthetic.py
