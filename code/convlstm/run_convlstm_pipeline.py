import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS = ["M1", "M2", "M3", "M4", "M5"]


def run(command, dry_run):
    print(" ".join(map(str, command)))
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run selected stages of the ConvLSTM workflow.")
    parser.add_argument("--steps", default="1,2,3", help="Any of 1,2,3,4,5,6.")
    parser.add_argument("--model", choices=MODELS, default="M1")
    parser.add_argument("--run", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    steps = {int(value.strip()) for value in args.steps.split(",") if value.strip()}
    if not steps <= {1, 2, 3, 4, 5, 6}:
        parser.error("--steps only accepts 1 through 6.")
    models = MODELS if args.all_models else [args.model]
    runs = range(1, 11) if args.all_runs else [args.run]
    python = sys.executable

    if 1 in steps:
        run([python, SCRIPT_DIR / "01_build_dataset.py"], args.dry_run)
    if 2 in steps:
        for model in models:
            for run_number in runs:
                run([python, SCRIPT_DIR / "02_train_convlstm.py", "--model", model, "--run", str(run_number)], args.dry_run)
    if 3 in steps:
        for model in models:
            for run_number in runs:
                run([python, SCRIPT_DIR / "03_evaluate_convlstm.py", "--model", model, "--run", str(run_number)], args.dry_run)
    if 4 in steps:
        run([python, SCRIPT_DIR / "04_summarize_convlstm_results.py"], args.dry_run)
    if 5 in steps:
        run([python, SCRIPT_DIR / "05_evaluate_robustness.py"], args.dry_run)
    if 6 in steps:
        run([python, SCRIPT_DIR / "06_export_prediction_frequency.py"], args.dry_run)


if __name__ == "__main__":
    main()

