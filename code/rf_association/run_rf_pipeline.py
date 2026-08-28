"""Run the complete first-stage random-forest association pipeline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RF_CODE_DIR = PROJECT_ROOT / "code" / "rf_association"
DEFAULT_WSM_INPUT = PROJECT_ROOT / "data" / "raw" / "WSM_Database_2025.csv"


@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    command: tuple[str, ...]


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    python = sys.executable
    wsm_input = Path(args.wsm_input).expanduser().resolve()

    train_command = [
        python,
        str(RF_CODE_DIR / "04_run_random_forest.py"),
        "--n-repeats",
        str(args.n_repeats),
    ]
    if args.overwrite:
        train_command.append("--overwrite")

    return [
        PipelineStep(
            1,
            "Filter WSM2025 and extract the 10 study regions",
            (
                python,
                str(RF_CODE_DIR / "01_prepare_wsm_data.py"),
                "--input",
                str(wsm_input),
                "--regions",
                "all",
            ),
        ),
        PipelineStep(
            2,
            "Compute the nine seismotectonic features",
            (
                python,
                str(RF_CODE_DIR / "02_compute_seismotectonic_features.py"),
                "--regions",
                "all",
            ),
        ),
        PipelineStep(
            3,
            "Convert magnitudes and create threshold labels",
            (
                python,
                str(RF_CODE_DIR / "03_prepare_magnitude_labels.py"),
                "--regions",
                "all",
            ),
        ),
        PipelineStep(
            4,
            "Run the 23 formal random-forest tasks",
            tuple(train_command),
        ),
        PipelineStep(
            5,
            "Create manuscript-ready RF summary tables",
            (python, str(RF_CODE_DIR / "05_summarize_rf_results.py")),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wsm-input",
        type=Path,
        default=DEFAULT_WSM_INPUT,
        help="Path to WSM_Database_2025.csv.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=range(1, 6),
        default=1,
        metavar="{1,2,3,4,5}",
        help="First pipeline step to execute (default: 1).",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        choices=range(1, 6),
        default=5,
        metavar="{1,2,3,4,5}",
        help="Last pipeline step to execute (default: 5).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1000,
        help="Number of repeated RF experiments per formal task (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow step 4 to replace existing formal-task run directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
    )
    args = parser.parse_args()
    if args.from_step > args.to_step:
        parser.error("--from-step cannot be greater than --to-step.")
    if args.n_repeats < 1:
        parser.error("--n-repeats must be positive.")
    return args


def main() -> None:
    args = parse_args()
    steps = [
        step
        for step in build_steps(args)
        if args.from_step <= step.number <= args.to_step
    ]

    wsm_input = Path(args.wsm_input).expanduser().resolve()
    if not args.dry_run and args.from_step == 1 and not wsm_input.exists():
        raise FileNotFoundError(
            f"WSM2025 input not found: {wsm_input}\n"
            "Download the official CSV and see data/README.md for placement."
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Selected steps: {args.from_step}-{args.to_step}")
    print(f"RF repeats per task: {args.n_repeats}")

    for step in steps:
        print(f"\n[{step.number}/5] {step.name}")
        print(shlex.join(step.command))
        if args.dry_run:
            continue
        try:
            subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Pipeline stopped at step {step.number}: {step.name} "
                f"(exit code {exc.returncode})."
            ) from exc

    if args.dry_run:
        print("\nDry run complete; no analysis commands were executed.")
    else:
        print("\nRF pipeline complete.")


if __name__ == "__main__":
    main()

