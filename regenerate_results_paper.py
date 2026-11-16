#!/usr/bin/env python3
"""Regenerate the results for writing/iclr2027/main_v2.tex.

Uses existing NumPy embedding bundles. Run with --dry-run to list outputs
without any fitting, or --section controls / --section synthetic to rerun part
of the paper. Existing results are replaced only with --overwrite.
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
from paper_protocol import RIDGE_LAMBDAS


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_paper"
MAIN_EXPERIMENTS = (
    ("WB", "resnet50", range(1, 11), True),
    ("CelebA", "resnet50", range(1, 11), True),
    ("multiNLI", "BERT", range(1, 6), False),
    ("WB", "dino_vitb16", range(1, 6), True),
    ("CelebA", "dino_vitb16", range(1, 6), True),
    ("multiNLI", "debertav3", range(1, 4), False),
)
COMPARISON_EXPERIMENTS = MAIN_EXPERIMENTS[:3]
CONTROL_EXPERIMENTS = (
    ("WB_no_spurious", "resnet50", range(1, 6), True),
    ("CIFAR10", "resnet50", range(3), False),
)
CONTROL_SELECTION = {
    "WB_no_spurious": "class_balanced_accuracy",
    "CIFAR10": "balanced_log_loss",
}
SYNTHETIC_REPORT = "synthetic_gamma_sweep.json"
SYNTHETIC_APPENDIX_REPORT = "synthetic_gamma_appendix.json"
THEORY_REPORT = "synthetic_theory_checks.json"
APPENDIX_GAMMAS = (0.5, 1.0, 3.0)
SECTIONS = ("all", "controls", "synthetic")


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


def expected_paths(
    *, results_dir: Path = RESULTS_DIR, include_theory_checks: bool, section: str = "all",
) -> list[Path]:
    """Return the complete, deterministic set of paper-plot JSON inputs."""
    comparison_dir = results_dir / "comparisons"
    paths = []
    if section == "all":
        paths.extend(
            results_dir / f"{dataset}_{embedding}_finetune.json"
            for dataset, embedding, _, _ in MAIN_EXPERIMENTS
        )
        for dataset, embedding, _, _ in COMPARISON_EXPERIMENTS:
            stem = f"{dataset}_{embedding}"
            for method in ("DFR", "AFR", "NEUROTUNE"):
                paths.append(comparison_dir / f"{stem}_{method}.json")
                suffix = "_aggregate_cb" if method == "NEUROTUNE" else "_aggregate"
                paths.append(comparison_dir / f"{stem}_{method}{suffix}.json")
    if section in ("all", "controls"):
        for dataset, embedding, _, _ in CONTROL_EXPERIMENTS:
            stem = f"{dataset}_{embedding}_finetune"
            paths.extend((results_dir / f"{stem}.json", results_dir / f"{stem}_aggregate.json"))
    if section in ("all", "synthetic"):
        paths.extend(results_dir / name for name in (SYNTHETIC_REPORT, SYNTHETIC_APPENDIX_REPORT))
        if include_theory_checks:
            paths.append(results_dir / THEORY_REPORT)
    return paths


def _finetune_config(
    *, results_dir: Path = RESULTS_DIR, overwrite: bool, workers: int, controls: bool = False,
) -> dict[str, Any]:
    config = deepcopy(FINETUNE_RESULTS_CONFIG)
    config.update(
        results_dir=str(results_dir), overwrite=overwrite, workers=workers,
        blas_threads=1, append=False, warm_start=False,
        experiments=_paper_experiments(CONTROL_EXPERIMENTS if controls else MAIN_EXPERIMENTS),
    )
    config["sweep_defaults"].update(
        ridge_lambdas=RIDGE_LAMBDAS,
        whitening_variants=[
            {"whitening_estimator": "empirical", "class_weighted": False},
            {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
        ],
    )
    for experiment in config["experiments"]:
        if experiment["dataset"] == "CIFAR10":
            experiment["whitening_variants"] = [
                {"whitening_estimator": "ledoit-wolf-nonlinear", "class_weighted": False},
            ]
        elif experiment["embedding"] == "BERT":
            # The MultiNLI BERT sweep also includes linear Ledoit–Wolf.
            experiment["whitening_variants"] = [
                {"whitening_estimator": estimator, "class_weighted": False}
                for estimator in ("empirical", "ledoit-wolf", "ledoit-wolf-nonlinear")
            ]
    return config


def _comparison_config(
    method: str, *, results_dir: Path = RESULTS_DIR, overwrite: bool, workers: int,
) -> dict[str, Any]:
    config = deepcopy(COMPARISON_CONFIG)
    config.update(
        results_dir=str(results_dir / "comparisons"), overwrite=overwrite, workers=workers,
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


def _synthetic_config(
    *, output: Path, overwrite: bool, workers: int, theory: bool, appendix: bool = False,
) -> dict[str, Any]:
    config = deepcopy(SYNTHETIC_SWEEP_CONFIG)
    config.update(
        output=str(output), overwrite=overwrite, workers=workers, blas_threads=1,
        run_main_sweep=not theory, run_theory_checks=theory,
        n_train=[1000], n_test=1000, n_sim=100, base_seed=1,
        d_n_ratio=[step / 10 for step in range(1, 21)],
        exact_train_label_balance=True, exact_test_label_balance=True,
        kappa_train=0.9, kappa_test=0.5,
        sigma_y=0.1, sigma_a=0.1, sigma_epsilon=20.0,
        baselines=["identity", "standardize"],
        whitening_estimators=["empirical", "ledoit-wolf-nonlinear"],
        whitening_relative_tolerance=None,
        ridge_lambda=1e-9, sklearn_C=None, class_balanced=False,
        plot_sample_size=1000, plot_sample_simulation=0, boundary_d_n_ratio=2.0,
        gamma=max(APPENDIX_GAMMAS) if appendix else 5.0,
    )
    config.pop("gamma_values", None)
    if appendix:
        config["gamma_values"] = list(APPENDIX_GAMMAS)
    return config


def _regenerate_controls(*, results_dir: Path, overwrite: bool, workers: int) -> list[Path]:
    outputs = run_finetune_sweeps(_finetune_config(
        results_dir=results_dir, overwrite=overwrite, workers=workers, controls=True,
    ))
    for dataset, embedding, _, _ in CONTROL_EXPERIMENTS:
        stem = f"{dataset}_{embedding}_finetune"
        runs, source_files = load_sweep_runs([results_dir / f"{stem}.json"])
        report = aggregate_runs(
            runs, default_selection_rule=CONTROL_SELECTION[dataset],
            exclude_single_iteration=True, refit_train_val=False,
            confidence=0.95, both=dataset == "WB_no_spurious",
        )
        report["source_files"] = source_files
        outputs.append(save_aggregate(
            report, output=results_dir / f"{stem}_aggregate.json", overwrite=overwrite,
        ))
    return outputs


def regenerate(
    *, results_dir: Path = RESULTS_DIR, overwrite: bool, workers: int,
    include_theory_checks: bool, section: str = "all",
) -> list[Path]:
    """Run the selected paper protocol and return every saved JSON path."""
    outputs = []
    if section == "all":
        comparison_dir = results_dir / "comparisons"
        outputs.extend(run_finetune_sweeps(_finetune_config(
            results_dir=results_dir, overwrite=overwrite, workers=workers,
        )))
        for method in ("DFR", "AFR", "NEUROTUNE"):
            outputs.extend(run_comparisons(_comparison_config(
                method, results_dir=results_dir, overwrite=overwrite, workers=workers,
            )))
        for dataset, embedding, _, _ in COMPARISON_EXPERIMENTS:
            stem = f"{dataset}_{embedding}"
            for method in ("DFR", "AFR", "NEUROTUNE"):
                suffix = "_aggregate_cb" if method == "NEUROTUNE" else "_aggregate"
                outputs.append(_aggregate(
                    comparison_dir / f"{stem}_{method}.json",
                    comparison_dir / f"{stem}_{method}{suffix}.json",
                    method=method, overwrite=overwrite,
                ))
    if section in ("all", "controls"):
        outputs.extend(_regenerate_controls(
            results_dir=results_dir, overwrite=overwrite, workers=workers,
        ))
    if section in ("all", "synthetic"):
        names = [SYNTHETIC_REPORT, SYNTHETIC_APPENDIX_REPORT]
        if include_theory_checks:
            names.append(THEORY_REPORT)
        for name in names:
            output = results_dir / name
            config = _synthetic_config(
                output=output, overwrite=overwrite, workers=workers,
                theory=name == THEORY_REPORT, appendix=name == SYNTHETIC_APPENDIX_REPORT,
            )
            outputs.append(save_results(run_sweep(config), output, overwrite=overwrite))
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing paper JSON files.")
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Directory for regenerated JSON files (default: results_paper).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers per sweep (default: 1).")
    parser.add_argument("--include-theory-checks", action="store_true", help="Also generate synthetic_theory_checks.json.")
    parser.add_argument("--dry-run", action="store_true", help="List outputs without fitting models.")
    parser.add_argument("--section", choices=SECTIONS, default="all",
                        help="Run the whole paper, only the controls, or only the synthetic sweeps.")
    args = parser.parse_args(argv)
    if args.section == "controls" and args.include_theory_checks:
        parser.error("--include-theory-checks requires the all or synthetic section.")
    if args.workers < 1:
        parser.error("--workers must be positive.")
    results_dir = Path(args.results_dir).expanduser()
    results_dir = (ROOT / results_dir).resolve()
    paths = expected_paths(
        results_dir=results_dir, include_theory_checks=args.include_theory_checks, section=args.section,
    )
    if args.dry_run:
        print("Would generate:")
        print("\n".join(str(path) for path in paths))
        return
    if not args.overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            parser.error("Existing results require --overwrite: " + ", ".join(map(str, existing)))
    saved = regenerate(results_dir=results_dir, overwrite=args.overwrite, workers=args.workers,
                       include_theory_checks=args.include_theory_checks, section=args.section)
    if set(saved) != set(paths):
        raise RuntimeError(
            "Paper-result plan drifted: expected "
            f"{len(paths)} files but saved {len(set(saved))}."
        )
    print(f"Regenerated {len(saved)} paper-result JSON files.")


if __name__ == "__main__":
    main()
