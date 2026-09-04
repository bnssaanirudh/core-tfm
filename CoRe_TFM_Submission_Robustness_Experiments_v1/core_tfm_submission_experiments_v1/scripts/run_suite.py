from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

CONFIGS = {
    "minimal": "minimal_submission.yaml",
    "full": "full_protocol.yaml",
}

def run(cmd, cwd):
    print("\nRUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)

def module_cmd(py, module, repo, config):
    return [py, "-m", module, "--repo", str(repo), "--config", str(config)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--preset", choices=CONFIGS, default="minimal")
    ap.add_argument("--stage", choices=["cpu","gpu","postprocess","all"], default="all")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    bundle = Path(__file__).resolve().parents[1]
    config = bundle / "config" / CONFIGS[args.preset]
    suite = repo / "core_tfm_rtx3050_final_suite"
    if not suite.exists():
        raise SystemExit(f"Missing existing runner: {suite}")

    py = sys.executable

    # The repository completion gate requires output_root/PREFLIGHT.json.
    # Run the native preflight before any substantive stage.
    if args.stage in {"cpu","gpu","all"}:
        run(module_cmd(py, "core_tfm_rtx3050_suite.preflight", repo, config), suite)

    if args.stage in {"cpu","all"}:
        run(module_cmd(py, "core_tfm_rtx3050_suite.run_cpu_experiments", repo, config), suite)

    if args.stage in {"gpu","all"}:
        cmd = module_cmd(py, "core_tfm_rtx3050_suite.run_matrix", repo, config)
        cmd += ["--profile", "paper"]
        if args.stop_on_error:
            cmd += ["--stop-on-error"]
        run(cmd, suite)

    if args.stage in {"postprocess","all"}:
        for mod in [
            "core_tfm_rtx3050_suite.aggregate",
            "core_tfm_rtx3050_suite.rare_class",
            "core_tfm_rtx3050_suite.completion_gate",
            "core_tfm_rtx3050_suite.make_paper_bundle",
        ]:
            run(module_cmd(py, mod, repo, config), suite)

    print("\nRequested stage finished. Run finalize_submission.py before using any new result in the manuscript.")

if __name__ == "__main__":
    main()
