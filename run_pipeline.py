"""
run_pipeline.py

Executes the full Car Insurance Claim Prediction pipeline sequentially,
in the order defined in the project README.

Each stage is run as a subprocess so that a crash in one stage does not
take down the Python interpreter running this script, and so that each
stage's own logging/print output streams to the console in real time.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-preprocessing        # resume from stage 2
    python run_pipeline.py --only 03a_ModelTRaining_Bootstrap.py
    python run_pipeline.py --dry-run                   # print plan, don't execute
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Pipeline stages in the exact order defined in the README.
# (name, script_filename, description)
PIPELINE_STAGES = [
    ("Preprocessing",              "01_Preprocessing.py",             "Data cleaning, encoding, artifact generation"),
    ("Feature Importance",         "02a_LGBM_FeatureImportance.py",   "Baseline LightGBM feature importance + pruning"),
    ("Baseline Model",             "02b_ModelBaseline.py",            "Out-of-the-box LightGBM baseline (5-fold CV)"),
    ("Hyperparameter Tuning",      "02c_HyperParameterTuning.py",     "Optuna/grid-search hyperparameter sweep"),
    ("Training + Calibration",     "03a_ModelTRaining_Bootstrap.py",  "Final model, isotonic calibration, bootstrap CIs"),
    ("SHAP Interpretability",      "03b_Shap_Interpretability.py",    "TreeSHAP global/local explainability"),
    ("Business Impact Simulation", "03c_Business_Impact_Simulation.py", "Threshold/cost-benefit simulation"),
]

LOG_FILE = Path("pipeline_run.log")


def log(message: str) -> None:
    """Print to console and append to a persistent log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_stage(name: str, script: str, description: str, dry_run: bool = False) -> float:
    """Run a single pipeline stage as a subprocess. Returns elapsed seconds."""
    script_path = Path(script)

    if not script_path.exists():
        log(f"ERROR: '{script}' not found in current directory. Aborting pipeline.")
        sys.exit(1)

    log(f"▶️  STAGE START: {name} ({script}) — {description}")

    if dry_run:
        log(f"   [dry-run] would execute: {sys.executable} {script}")
        return 0.0

    start = time.time()
    result = subprocess.run([sys.executable, str(script_path)])
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"STAGE FAILED: {name} (exit code {result.returncode}) after {elapsed:.1f}s")
        log("Pipeline halted. Fix the error above and re-run, optionally with --skip-* flags to resume.")
        sys.exit(result.returncode)

    log(f"STAGE COMPLETE: {name} in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the car insurance claim prediction pipeline end-to-end.")
    parser.add_argument("--only", type=str, default=None,
                         help="Run a single stage by its script filename, e.g. --only 02c_HyperParameterTuning.py")
    parser.add_argument("--skip-preprocessing", action="store_true",
                         help="Skip stage 1 (01_Preprocessing.py), useful when preprocessed artifacts already exist.")
    parser.add_argument("--from-stage", type=str, default=None,
                         help="Resume pipeline starting at the given script filename (inclusive).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the execution plan without running anything.")
    args = parser.parse_args()

    stages = PIPELINE_STAGES

    if args.only:
        stages = [s for s in PIPELINE_STAGES if s[1] == args.only]
        if not stages:
            print(f"No stage found matching '{args.only}'. Valid options:")
            for _, script, _ in PIPELINE_STAGES:
                print(f"  - {script}")
            sys.exit(1)

    elif args.from_stage:
        names = [s[1] for s in PIPELINE_STAGES]
        if args.from_stage not in names:
            print(f"'{args.from_stage}' is not a recognized stage script.")
            sys.exit(1)
        idx = names.index(args.from_stage)
        stages = PIPELINE_STAGES[idx:]

    elif args.skip_preprocessing:
        stages = [s for s in PIPELINE_STAGES if s[1] != "01_Preprocessing.py"]

    log("=" * 70)
    log(f"PIPELINE RUN START — {len(stages)} stage(s) scheduled")
    log("=" * 70)

    total_start = time.time()
    for name, script, description in stages:
        run_stage(name, script, description, dry_run=args.dry_run)

    total_elapsed = time.time() - total_start
    log("=" * 70)
    log(f"PIPELINE RUN COMPLETE — total elapsed: {total_elapsed / 60:.1f} minutes")
    log("=" * 70)


if __name__ == "__main__":
    main()