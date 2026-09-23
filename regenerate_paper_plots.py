#!/usr/bin/env python3
"""Regenerate the data-backed PNGs included by writing/iclr2027/main.tex.

Run ``./regenerate_paper_plots.py`` from any directory. All plotting commands
use this repository's ``.env/bin/python``. Pass ``--add-theory-plots`` to
render the two theory confirmation plots from an existing validation report.
This script never runs the theory validation sweep. Use ``--dry-run`` to print
the commands without running them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".env" / "bin" / "python"
MANUSCRIPT = ROOT / "writing" / "iclr2027" / "main.tex"
MAIN_SYNTHETIC_REPORT = ROOT / "results_paper" / "synthetic_gamma_sweep.json"
THEORY_REPORT = ROOT / "results_paper" / "synthetic_theory_checks.json"
THEORY_OUTPUT = ROOT / "plots" / "synthetic_theory"
PAPER_OUTPUT = ROOT / "plots" / "synthetic_fin"
THEORY_PLOT_NAMES = ("theorem2_confirmation", "proposition2_confirmation")


def included_plots() -> set[Path]:
    source = MANUSCRIPT.read_text(encoding="utf-8")
    names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+\.png)\}", source)
    return {(MANUSCRIPT.parent / name).resolve() for name in names}


def theory_report_is_complete() -> bool:
    if not THEORY_REPORT.is_file():
        return False
    try:
        with THEORY_REPORT.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        return False
    return all(
        report.get(key, {}).get("results")
        for key in ("theorem2_validation", "proposition2_validation")
    )


def validate_comparison_inputs() -> None:
    """Keep the manuscript's limited AFR grid and NT selection protocol."""
    comparison_dir = ROOT / "results_paper" / "comparisons"
    for dataset in ("WB_resnet50", "CelebA_resnet50", "multiNLI_BERT"):
        afr = comparison_dir / f"{dataset}_AFR_aggregate.json"
        neurotune = comparison_dir / f"{dataset}_NEUROTUNE_aggregate_cb.json"
        if not afr.is_file() or not neurotune.is_file():
            raise FileNotFoundError(
                f"Missing comparison aggregate(s) for {dataset}: {afr} or {neurotune}"
            )
        with afr.open(encoding="utf-8") as handle:
            experiments = json.load(handle)["experiments"]
        if not experiments or any(
            experiment.get("candidate_afr_gammas") != [1.0, 2.0, 4.0, 8.0]
            for experiment in experiments
        ):
            raise ValueError(
                f"{afr} must select AFR from gamma values 1, 2, 4, and 8."
            )


def run_step(name: str, arguments: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    command = [str(PYTHON), *arguments]
    print(f"\n{name}\n  {' '.join(arguments)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show every step without changing files."
    )
    parser.add_argument(
        "--add-theory-plots",
        action="store_true",
        help="Render theory plots from the existing theory-validation report.",
    )
    args = parser.parse_args()
    if not PYTHON.is_file():
        parser.error(f"Missing virtual environment Python: {PYTHON}")
    if not MAIN_SYNTHETIC_REPORT.is_file():
        parser.error(f"Missing synthetic results: {MAIN_SYNTHETIC_REPORT}")
    validate_comparison_inputs()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "whitening-mpl"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "Synthetic main and appendix performance figures",
            ["viz_synthetic.py", str(MAIN_SYNTHETIC_REPORT),
             "--output-dir", "plots/synthetic_fin", "--overwrite"],
        ),
        (
            "ERM group accuracy: Figures 3 and 12",
            ["viz.py", "--figure", "both", "--group-accuracy-panels",
             "--output", "plots/group_accuracy", "--overwrite"],
        ),
        (
            "Spectrum and normalized Rayleigh quotient: Figure 4",
            ["viz_experiments.py", "--preset", "main", "--plot-type", "spectrum",
             "--overwrite"],
        ),
        (
            "Method comparisons: Figure 5",
            ["viz_comparison.py", "--use-aggregates", "--neurotune-selection",
             "class-balanced", "--output", "plots/comparison_worst_group_accuracy",
             "--overwrite"],
        ),
        (
            "Comparison Rayleigh quotients: Figure 6",
            ["viz_experiments.py", "--plot-type", "comparison_plot", "--overwrite"],
        ),
        (
            "Whitening-estimator appendix figure",
            ["viz.py", "--figure", "main", "--group-accuracy-panels",
             "--compare-whitening-estimators", "--output",
             "plots/whitening_estimators_group_accuracy", "--overwrite"],
        ),
        (
            "Equal-group method comparisons: Figure 13",
            ["viz_comparison.py", "--use-aggregates", "--neurotune-selection",
             "class-balanced", "--metric", "group_balanced_accuracy",
             "--output", "plots/comparison_equal_group_accuracy", "--overwrite"],
        ),
    ]
    for name, arguments in steps:
        run_step(name, arguments, dry_run=args.dry_run, env=env)

    if args.add_theory_plots:
        if not theory_report_is_complete():
            raise FileNotFoundError(
                "Theory plots require a completed validation report at "
                f"{THEORY_REPORT}. Run the theory validation separately, then "
                "rerun this script with --add-theory-plots."
            )
        run_step(
            "Render the two appendix theory plots",
            ["viz_synthetic.py", str(THEORY_REPORT), "--output-dir",
             str(THEORY_OUTPUT), "--overwrite"],
            dry_run=args.dry_run,
            env=env,
        )
        for name in THEORY_PLOT_NAMES:
            source = THEORY_OUTPUT / f"{name}.png"
            destination = PAPER_OUTPUT / f"current_{name}.png"
            print(f"  copy {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}",
                  flush=True)
            if not args.dry_run:
                shutil.copy2(source, destination)

    expected = included_plots()
    theory_plots = {PAPER_OUTPUT / f"current_{name}.png" for name in THEORY_PLOT_NAMES}
    expected = expected if args.add_theory_plots else expected - theory_plots
    expected_count = 12 if args.add_theory_plots else 10
    if len(expected) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} manuscript PNGs, found {len(expected)} in {MANUSCRIPT}. "
            "Update this script when manuscript figures change."
        )
    if not args.dry_run:
        missing = sorted(path for path in expected if not path.is_file())
        if missing:
            raise RuntimeError("Missing manuscript figure(s): " + ", ".join(map(str, missing)))
    print(f"\n{'Would regenerate' if args.dry_run else 'Regenerated'} "
          f"all {len(expected)} PNGs included by {MANUSCRIPT.relative_to(ROOT)}.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
