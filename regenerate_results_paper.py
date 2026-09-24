#!/usr/bin/env python3
"""Regenerate every JSON input used by ``regenerate_paper_plots.py``.

Run this from any directory.  The script records the paper experiment grid in
one place instead of depending on the editable defaults in the individual
sweep scripts.  Existing paper results are replaced only with ``--overwrite``.

The default run writes the 25 JSON files required by the standard paper-plot
workflow.  ``--include-theory-checks`` additionally creates the synthetic
theory-validation report consumed by ``regenerate_paper_plots.py
--add-theory-plots``.  ``--dry-run`` lists the files without fitting models.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from aggregate import aggregate_runs, load_sweep_runs, save_aggregate
from comparison import COMPARISON_CONFIG, run_comparisons
from finetune_results import FINETUNE_RESULTS_CONFIG, run_finetune_sweeps
from synthetic_sweep import SYNTHETIC_SWEEP_CONFIG, run_sweep, save_results


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_paper"
COMPARISON_DIR = RESULTS_DIR / "comparisons"
RIDGE_LAMBDAS = [
    1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1,
    1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1_000.0, 5_000.0,
    10_000.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0,
    5_000_000.0,
]
MAIN_EXPERIMENTS = (
    ("WB", "resnet50", range(1, 11), True),
    ("CelebA", "resnet50", range(1, 11), True),
    ("multiNLI", "BERT", range(1, 6), False),
    ("WB", "dino_vitb16", range(1, 6), True),
    ("CelebA", "dino_vitb16", range(1, 6), True),
    ("multiNLI", "debertav3", range(1, 4), False),
)
COMPARISON_EXPERIMENTS = MAIN_EXPERIMENTS[:3]


def _paper_experiments(
    experiments: Iterable[tuple[str, str, Iterable[int], bool]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "embedding": embedding,
            "seeds": list(seeds),
            "class_balanced": class_balanced,
        }
        for dataset, embedding, seeds, class_balanced in experiments
    ]


def expected_paths(*, include_theory_checks: bool) -> list[Path]:
    """Return the complete, deterministic set of paper-plot JSON inputs."""
    paths = [
        RESULTS_DIR / f"{dataset}_{embedding}_finetune.json"
        for dataset, embedding, _, _ in MAIN_EXPERIMENTS
    ]
    for dataset, embedding, _, _ in COMPARISON_EXPERIMENTS:
        stem = f"{dataset}_{embedding}"
        for method in ("DFR", "AFR", "NEUROTUNE"):
            paths.append(COMPARISON_DIR / f"{stem}_{method}.json")
            suffix = "_aggregate_cb" if method == "NEUROTUNE" else "_aggregate"
            paths.append(COMPARISON_DIR / f"{stem}_{method}{suffix}.json")
    paths.append(RESULTS_DIR / "synthetic_gamma_sweep.json")
    if include_theory_checks:
        paths.append(RESULTS_DIR / "synthetic_theory_checks.json")
    return paths


def _finetune_config(*, overwrite: bool, workers: int) -> dict[str, Any]:
    config = deepcopy(FINETUNE_RESULTS_CONFIG)
    config.update(
        results_dir=str(RESULTS_DIR), overwrite=overwrite, workers=workers,
        blas_threads=1, experiments=_paper_experiments(MAIN_EXPERIMENTS),
    )
    config["sweep_defaults"].update(
        ridge_lambdas=RIDGE_LAMBDAS,
        whitening_variants=[
            {"whitening_estimator": "empirical", "class_weighted": False},
            {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
        ],
    )
    # The appendix's multiNLI BERT sweep also includes linear Ledoit--Wolf.
    config["experiments"][2]["whitening_variants"] = [
        {"whitening_estimator": "empirical", "class_weighted": False},
        {"whitening_estimator": "ledoit-wolf", "class_weighted": False},
        {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
    ]
    return config


def _comparison_config(method: str, *, overwrite: bool, workers: int) -> dict[str, Any]:
    config = deepcopy(COMPARISON_CONFIG)
    config.update(
        results_dir=str(COMPARISON_DIR), overwrite=overwrite, workers=workers,
        blas_threads=1, methods=[method],
        experiments=_paper_experiments(COMPARISON_EXPERIMENTS),
    )
    defaults = config["sweep_defaults"]
    defaults.update(ridge_lambdas=RIDGE_LAMBDAS)
    if method == "DFR":
        defaults.update(
            whitening_variants=[{"whitening_estimator": "ledoit-wolf-nonlinear"}],
        )
    elif method == "AFR":
        defaults.update(
            afr_gammas=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            whitening_variants=[
                {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
                {"whitening_estimator": "empirical", "class_weighted": False},
            ],
        )
        # The saved MultiNLI sweep intentionally omitted the two smallest
        # gamma values; retain that paper-specific exception.
        for experiment in config["experiments"]:
            if experiment["dataset"] == "multiNLI":
                experiment["afr_gammas"] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    else:
        defaults.update(
            neurotune_threshold=[0.0, 0.1, 0.25, 0.5],
            whitening_variants=[
                {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
                {"whitening_estimator": "empirical", "class_weighted": False},
            ],
        )
    return config


def _aggregate(
    sweep: Path, output: Path, *, method: str, overwrite: bool,
) -> Path:
    runs, source_files = load_sweep_runs([sweep])
    options: Mapping[str, Any] = {
        "DFR": {},
        "AFR": {
            "selection_afr_gammas": [1.0, 2.0, 4.0, 8.0],
            "exclude_single_iteration": True,
        },
        "NEUROTUNE": {
            "selection_rules": {"neurotune": "class_balanced_accuracy"},
            "selection_neurotune_thresholds": [0.0, 0.1, 0.25, 0.5],
            "exclude_single_iteration": True,
            "refit_train_val": False,
        },
    }[method]
    report = aggregate_runs(
        runs,
        default_selection_rule="class_balanced_accuracy",
        confidence=0.90,
        refit_workers=4,
        refit_blas_threads=1,
        **options,
    )
    report["source_files"] = source_files
    return save_aggregate(report, output=output, overwrite=overwrite)


def _synthetic_config(*, output: Path, overwrite: bool, workers: int, theory: bool) -> dict[str, Any]:
    config = deepcopy(SYNTHETIC_SWEEP_CONFIG)
    config.update(
        output=str(output), overwrite=overwrite, workers=workers, blas_threads=1,
        run_main_sweep=not theory, run_theory_checks=theory,
    )
    if theory:
        config.pop("gamma_values", None)
    return config


def regenerate(*, overwrite: bool, workers: int, include_theory_checks: bool) -> list[Path]:
    """Run the fixed paper protocol and return every saved JSON path."""
    outputs = run_finetune_sweeps(_finetune_config(overwrite=overwrite, workers=workers))
    for method in ("DFR", "AFR", "NEUROTUNE"):
        outputs.extend(run_comparisons(_comparison_config(method, overwrite=overwrite, workers=workers)))
    for dataset, embedding, _, _ in COMPARISON_EXPERIMENTS:
        stem = f"{dataset}_{embedding}"
        for method in ("DFR", "AFR", "NEUROTUNE"):
            suffix = "_aggregate_cb" if method == "NEUROTUNE" else "_aggregate"
            outputs.append(_aggregate(
                COMPARISON_DIR / f"{stem}_{method}.json",
                COMPARISON_DIR / f"{stem}_{method}{suffix}.json",
                method=method, overwrite=overwrite,
            ))
    main_output = RESULTS_DIR / "synthetic_gamma_sweep.json"
    outputs.append(save_results(run_sweep(_synthetic_config(
        output=main_output, overwrite=overwrite, workers=workers, theory=False,
    )), main_output, overwrite=overwrite))
    if include_theory_checks:
        theory_output = RESULTS_DIR / "synthetic_theory_checks.json"
        outputs.append(save_results(run_sweep(_synthetic_config(
            output=theory_output, overwrite=overwrite, workers=workers, theory=True,
        )), theory_output, overwrite=overwrite))
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing paper JSON files.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers per sweep (default: 1).")
    parser.add_argument("--include-theory-checks", action="store_true", help="Also generate synthetic_theory_checks.json.")
    parser.add_argument("--dry-run", action="store_true", help="List outputs without fitting models.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive.")
    paths = expected_paths(include_theory_checks=args.include_theory_checks)
    if args.dry_run:
        print("Would generate:")
        print("\n".join(str(path.relative_to(ROOT)) for path in paths))
        return
    if not args.overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            parser.error("Existing results require --overwrite: " + ", ".join(map(str, existing)))
    saved = regenerate(overwrite=args.overwrite, workers=args.workers,
                       include_theory_checks=args.include_theory_checks)
    if set(saved) != set(paths):
        raise RuntimeError(
            "Paper-result plan drifted: expected "
            f"{len(paths)} files but saved {len(set(saved))}."
        )
    print(f"Regenerated {len(saved)} paper-result JSON files.")


if __name__ == "__main__":
    main()
