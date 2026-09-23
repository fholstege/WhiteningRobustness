#!/usr/bin/env python3
"""Run frozen-feature comparison methods over the final-result experiment grid.

Comparison files contain only the additional methods. Load them together with
the matching ``finetune_results.py`` JSON in ``aggregate.py`` to avoid repeating
the baseline and whitening fits.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from data import EmbeddingBundle, SavedLinearHead
from evaluate import find_bundle
from finetune_results import (
    FINETUNE_RESULTS_CONFIG,
    _experiment_sweep_config,
    _filename_component,
    _resolved_root,
    _validated_parallelism,
    _validated_progress_every,
    _validated_whitening_variants,
)
from hyper_param_sweep import _validated_ridge_lambdas, save_sweep
from hyper_param_sweep import _resolved_representation
from methods import (
    sweep_afr_bundle,
    sweep_dfr_bundle,
    sweep_ho_erm_bundle,
    sweep_neurotune_bundle,
    valid_neurotune_thresholds,
)


COMPARISON_METHOD_REGISTRY: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "DFR": sweep_dfr_bundle,
    "AFR": sweep_afr_bundle,
    "HO_ERM": sweep_ho_erm_bundle,
    "NEUROTUNE": sweep_neurotune_bundle,
}


COMPARISON_CONFIG: dict[str, Any] = {
    # Ordinary binary and multiclass fits use the solvers fixed in model.py.
    # DFR balances by group sampling; AFR uses adaptation weights.
    "artifact_root": FINETUNE_RESULTS_CONFIG["artifact_root"],
    "saved_head_root": "artifacts/last_layers",
    "results_dir": "results_paper/comparisons",
    "verify_hashes": FINETUNE_RESULTS_CONFIG["verify_hashes"],
    "overwrite": False,
    "warm_start": False,
    "workers": 1,
    "blas_threads": 1,
    # Print every fitted candidate so NeuroTune mask sizes are visible for the
    # complete threshold-by-transform-by-ridge grid.
    "progress_every": 1,
    "unseeded_split_seed": 0,
    "methods": ["HO_ERM"],
    "sweep_defaults": {
        **copy.deepcopy(FINETUNE_RESULTS_CONFIG["sweep_defaults"]),
        "dfr_subsamples": 10,
        # Comparison methods are evaluated without preprocessing, with
        # standardization, and with nonlinear Ledoit--Wolf whitening.
        # The ridge grid is inherited unchanged from FINETUNE_RESULTS_CONFIG.
        # Edit this list to
        # change the representation transforms used by every method.
        "transforms": ["identity", "standardize", "whiten"],
        "whitening_variants": [
            {   
                "whitening_estimator": "ledoit-wolf-nonlinear",
                "class_weighted": False,
            },
              {   
                "whitening_estimator": "empirical",
                "class_weighted": False,
            },
        ],
        # AFR tunes gamma jointly with ridge_lambda. Gamma controls how
        # strongly high-confidence examples are downweighted; 
        "afr_gammas": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "afr_random_state": 0,
        # NeuroTune identifies suspicious raw coordinates on one validation
        # half, fits on the other half, and selects threshold/ridge by exact
        # class-balanced accuracy on the identification half. Thresholds that
        # remove every coordinate are rejected explicitly for each seed.
        "neurotune_threshold": [
            0.0, 0.1, 0.25, 0.5
        ],
    },
    # Edit experiments here independently of finetune_results.py. `embedding`
    # selects the stored encoder; `seeds` selects its representation bundles.
    # Entries may override sweep_defaults (e.g. ridge_lambdas). NeuroTune uses
    # each entry's class_balanced value for its ordinary retrained head; the
    # other comparison methods define their own balancing protocol.
    "experiments": [
        {
            "dataset": "WB",
            "embedding": "resnet50",
            "seeds": list(range(1, 11)),
            "class_balanced": True,
        },
        {
             "dataset": "WB",
             "embedding": "dino_vitb16",
             "seeds": [1, 2, 3, 4, 5],
             "class_balanced": True,
         },
        {
            "dataset": "CelebA",
            "embedding": "resnet50",
            "seeds": list(range(1, 11)),
            "class_balanced": True,
        },
        {
             "dataset": "CelebA",
             "embedding": "dino_vitb16",
             "seeds": [1, 2, 3, 4, 5],
             "class_balanced": True,
         },
        {
            "dataset": "multiNLI",
            "embedding": "BERT",
            "seeds": [1, 2, 3, 4, 5],
            "class_balanced": False,
        },
        {
             "dataset": "multiNLI",
             "embedding": "debertav3",
             "seeds": [1, 2, 3],
             "class_balanced": False,
         }
    ],
}


def comparison_output_path(
    *,
    results_dir: str | Path,
    dataset: str,
    embedding: str,
    methods: Sequence[str] = ("DFR",),
) -> Path:
    method_values = tuple(str(method).upper() for method in methods)
    if len(method_values) != 1:
        raise ValueError("Each comparison output file must contain exactly one method.")
    method_suffix = method_values[0]
    return _resolved_root(results_dir) / (
        f"{_filename_component(dataset)}_"
        f"{_filename_component(embedding)}_{method_suffix}.json"
    )


def _validated_methods(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = config.get("methods")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("methods must be a non-empty sequence.")
    methods = tuple(str(value).strip().upper() for value in values)
    unknown = sorted(set(methods).difference(COMPARISON_METHOD_REGISTRY))
    if unknown:
        raise ValueError("Unknown comparison methods: " + ", ".join(unknown))
    if len(set(methods)) != len(methods):
        raise ValueError("comparison methods contains duplicates.")
    return methods


def _validated_experiments(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list.")
    outputs: set[Path] = set()
    resolved = []
    methods = _validated_methods(config)
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise TypeError("Every comparison experiment must be a mapping.")
        dataset = str(experiment["dataset"])
        embedding = str(experiment["embedding"])
        sweep_config = _experiment_sweep_config(config, experiment)
        sweep_config["saved_head_root"] = config.get(
            "saved_head_root", COMPARISON_CONFIG["saved_head_root"]
        )
        for method in methods:
            method_sweep_config = copy.deepcopy(sweep_config)
            # DFR, AFR, and HO_ERM define their own balancing protocol.
            # NeuroTune retrains the ordinary head and therefore preserves the
            # experiment's configured class-balancing choice.
            if method != "NEUROTUNE":
                method_sweep_config["class_balanced"] = True
            output = comparison_output_path(
                results_dir=config["results_dir"],
                dataset=dataset,
                embedding=embedding,
                methods=(method,),
            )
            if output in outputs:
                raise ValueError(f"Duplicate comparison output: {output}")
            outputs.add(output)
            resolved.append(
                {
                    "dataset": dataset,
                    "embedding": embedding,
                    "method": method,
                    "output": output,
                    "sweep_config": method_sweep_config,
                }
            )
    return resolved


def _method_candidate_multiplier(method: str, job: Mapping[str, Any]) -> int:
    if method == "AFR":
        return len(job["afr_gammas"])
    if method == "NEUROTUNE":
        return len(job["neurotune_threshold"])
    return 1


def _run_method_for_transforms(
    method: str,
    bundle: EmbeddingBundle,
    transforms: Sequence[str],
    *,
    whitening_estimator: str,
    whitening_class_weighted: bool,
    whitening_relative_tolerance: float | None,
    job: Mapping[str, Any],
    progress_callback: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    common = {
        "transforms": transforms,
        "ridge_lambdas": job["ridge_lambdas"],
        "whitening_estimator": whitening_estimator,
        "whitening_class_weighted": whitening_class_weighted,
        "whitening_relative_tolerance": whitening_relative_tolerance,
        "progress_callback": progress_callback,
    }
    if method == "DFR":
        return sweep_dfr_bundle(
            bundle,
            split_seed=job["split_seed"],
            subsample_count=job.get("dfr_subsamples", 10),
            erm_head=job.get("saved_head"),
            warm_start=bool(job["warm_start"]),
            **common,
        )
    if method == "HO_ERM":
        return sweep_ho_erm_bundle(
            bundle,
            split_seed=job["split_seed"],
            erm_head=job.get("saved_head"),
            warm_start=bool(job["warm_start"]),
            **common,
        )
    if method == "AFR":
        return sweep_afr_bundle(
            bundle, score_head=job["saved_head"],
            afr_gammas=job["afr_gammas"],
            random_state=job["afr_random_state"],
            warm_start=bool(job["warm_start"]),
            **common,
        )
    if method == "NEUROTUNE":
        return sweep_neurotune_bundle(
            bundle,
            selector_head=job["saved_head"],
            threshold=job["neurotune_threshold"],
            class_balanced=job["class_balanced"],
            split_seed=job["split_seed"],
            warm_start=bool(job["warm_start"]),
            **common,
        )
    raise RuntimeError(f"Unhandled comparison method: {method}")


def _run_comparison_seed_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = threadpool_limits(limits=int(job["blas_threads"]))
    with context:
        bundle_path = find_bundle(
            Path(job["artifact_root"]),
            dataset=str(job["dataset"]),
            representation=str(job["representation"]),
            seed=job["seed"],
        )
        bundle = EmbeddingBundle.load(
            bundle_path,
            verify_hashes=bool(job["verify_hashes"]),
        )
        if job["warm_start"] or any(
            method in {"AFR", "NEUROTUNE"} for method in job["methods"]
        ):
            job = dict(job)
            saved_head_path = (
                Path(job["saved_head_root"])
                / str(bundle.manifest.get("dataset", job["dataset"]))
                / str(bundle.manifest.get("representation", job["representation"]))
                / (
                    "default"
                    if job["seed"] is None
                    else f"seed_{int(job['seed']):03d}"
                )
            )
            job["saved_head"] = SavedLinearHead.load(
                saved_head_path,
                classes=np.unique(bundle.split("train").y),
                input_features=bundle.split("train").X.shape[1],
                verify_hashes=bool(job["verify_hashes"]),
            )
        transform_count = len(job["baselines"]) + (
            len(job["variants"]) if job["include_whitening"] else 0
        )
        if "NEUROTUNE" in job["methods"]:
            job = dict(job)
            valid_thresholds = valid_neurotune_thresholds(
                bundle,
                selector_head=job["saved_head"],
                thresholds=job["neurotune_threshold"],
                split_seed=job["split_seed"],
            )
            if not valid_thresholds:
                raise ValueError(
                    "Every configured NeuroTune threshold removes every "
                    "coordinate on this seed's identification half."
                )
            if len(valid_thresholds) != len(job["neurotune_threshold"]):
                invalid_thresholds = [
                    value
                    for value in job["neurotune_threshold"]
                    if value not in valid_thresholds
                ]
                raise ValueError(
                    f"NeuroTune seed={job['seed']} cannot evaluate threshold(s) "
                    + ", ".join(f"{value:g}" for value in invalid_thresholds)
                    + " because they remove every coordinate on the identification "
                    "half. Increase or replace these thresholds; the configured "
                    "hyperparameter grid is not pruned silently."
                )
        total = len(job["ridge_lambdas"]) * transform_count * sum(
            _method_candidate_multiplier(method, job)
            for method in job["methods"]
        )
        completed = 0

        def report_progress(run: dict[str, Any]) -> None:
            nonlocal completed
            completed += 1
            interval = int(job["progress_every"])
            if completed % interval and completed != total:
                return
            neurotune = run.get("neurotune", {})
            retained = (
                f"{neurotune['retained_coordinates']}/"
                f"{neurotune['original_coordinates']}"
                if neurotune
                else "n/a"
            )
            print(
                f"  seed={job['seed']} progress: "
                f"{completed}/{total}; method={run['method']} "
                f"threshold={neurotune.get('threshold', 'n/a')} "
                f"retained={retained} "
                f"ridge={float(run['ridge_lambda']):g}",
                flush=True,
            )

        runs: list[dict[str, Any]] = []
        for method in job["methods"]:
            if job["baselines"]:
                runs.extend(
                    _run_method_for_transforms(
                        method,
                        bundle,
                        job["baselines"],
                        whitening_estimator="empirical",
                        whitening_class_weighted=False,
                        whitening_relative_tolerance=job[
                            "whitening_relative_tolerance"
                        ],
                        job=job,
                        progress_callback=report_progress,
                    )
                )
            if job["include_whitening"]:
                for variant in job["variants"]:
                    runs.extend(
                        _run_method_for_transforms(
                            method,
                            bundle,
                            ("whiten",),
                            whitening_estimator=variant[
                                "whitening_estimator"
                            ],
                            whitening_class_weighted=variant["class_weighted"],
                            whitening_relative_tolerance=variant.get(
                                "whitening_relative_tolerance",
                                job["whitening_relative_tolerance"],
                            ),
                            job=job,
                            progress_callback=report_progress,
                        )
                    )
        return runs


def _validated_float_grid(
    values: Any,
    *,
    name: str,
    strictly_positive: bool,
) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence.")
    grid = tuple(float(value) for value in values)
    if len(set(grid)) != len(grid):
        raise ValueError(f"{name} contains duplicates.")
    lower_bound_ok = (
        (lambda value: value > 0.0)
        if strictly_positive
        else (lambda value: value >= 0.0)
    )
    if any(not lower_bound_ok(value) or value == float("inf") for value in grid):
        qualifier = "strictly positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must contain finite {qualifier} values.")
    return grid


def _validated_finite_float_grid(values: Any, *, name: str) -> tuple[float, ...]:
    """Validate a finite grid whose values may be negative."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence.")
    grid = tuple(float(value) for value in values)
    if len(set(grid)) != len(grid) or not np.isfinite(grid).all():
        raise ValueError(f"{name} must contain unique finite values.")
    return grid


def _run_comparison_sweep(config: Mapping[str, Any]) -> dict[str, Any]:
    representation, seeds = _resolved_representation(dict(config))
    if not seeds:
        raise ValueError("At least one comparison seed is required.")
    artifact_root = _resolved_root(config["artifact_root"])
    saved_head_root = _resolved_root(
        config.get("saved_head_root", COMPARISON_CONFIG["saved_head_root"])
    )
    ridge_lambdas = _validated_ridge_lambdas(config["ridge_lambdas"])
    methods = _validated_methods(config)
    afr_gammas = _validated_float_grid(
        config.get("afr_gammas", (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)),
        name="afr_gammas",
        strictly_positive=False,
    )
    afr_random_state = int(config.get("afr_random_state", 0))
    if afr_random_state < 0:
        raise ValueError("afr_random_state must be nonnegative.")
    neurotune_threshold = _validated_finite_float_grid(
        config.get("neurotune_threshold", (0.0,)),
        name="neurotune_threshold",
    )
    variants = _validated_whitening_variants(config["whitening_variants"])
    workers, blas_threads = _validated_parallelism(config)
    progress_every = _validated_progress_every(config)
    unseeded_split_seed = int(config["unseeded_split_seed"])
    if unseeded_split_seed < 0:
        raise ValueError("unseeded_split_seed must be nonnegative.")

    requested_transforms = [str(value) for value in config["transforms"]]
    baselines = tuple(
        value
        for value in requested_transforms
        if value != "whiten"
    )
    include_whitening = "whiten" in requested_transforms
    jobs = [
        {
            "artifact_root": str(artifact_root),
            "dataset": str(config["dataset"]),
            "representation": representation,
            "seed": seed,
            "dfr_subsamples": config.get("dfr_subsamples", 10),
            "split_seed": (
                int(seed) if seed is not None else unseeded_split_seed
            ),
            "verify_hashes": bool(config["verify_hashes"]),
            "warm_start": bool(config.get("warm_start", False)),
            "saved_head_root": str(saved_head_root),
            "ridge_lambdas": ridge_lambdas,
            "methods": methods,
            "afr_gammas": afr_gammas,
            "afr_random_state": afr_random_state,
            "neurotune_threshold": neurotune_threshold,
            "class_balanced": bool(config.get("class_balanced", False)),
            "baselines": baselines,
            "include_whitening": include_whitening,
            "variants": variants,
            "whitening_relative_tolerance": config[
                "whitening_relative_tolerance"
            ],
            "blas_threads": blas_threads,
            "progress_every": progress_every,
        }
        for seed in seeds
    ]

    seed_runs: list[list[dict[str, Any]] | None] = [None] * len(jobs)
    effective_workers = min(workers, len(jobs))
    if effective_workers == 1:
        for index, job in enumerate(jobs):
            print(
                f"[{index + 1}/{len(jobs)}] comparisons "
                f"dataset={config['dataset']} "
                f"representation={representation} seed={job['seed']} "
                f"split_seed={job['split_seed']}",
                flush=True,
            )
            seed_runs[index] = _run_comparison_seed_job(job)
    else:
        print(
            f"Running {len(jobs)} comparison jobs with {effective_workers} workers "
            f"and {blas_threads} BLAS thread(s) per worker.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_indices = {
                executor.submit(_run_comparison_seed_job, job): index
                for index, job in enumerate(jobs)
            }
            completed = 0
            for future in as_completed(future_indices):
                index = future_indices[future]
                seed_runs[index] = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] completed comparisons "
                    f"dataset={config['dataset']} seed={jobs[index]['seed']}",
                    flush=True,
                )

    runs = [
        run
        for per_seed in seed_runs
        if per_seed is not None
        for run in per_seed
    ]
    return {
        "schema_version": 1,
        "kind": "ridge_lambda_sweep",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "comparison_methods": list(methods),
            "dataset": config["dataset"],
            "representation": representation,
            "seeds": list(seeds),
            "dfr_subsamples": config.get("dfr_subsamples", 10),
            "dfr_split_seeds": [job["split_seed"] for job in jobs],
            "transforms": requested_transforms,
            "whitening_variants": list(variants),
            "ridge_lambdas": list(ridge_lambdas),
            "afr_gammas": list(afr_gammas) if "AFR" in methods else [],
            "afr_random_state": afr_random_state,
            "saved_head_root": str(saved_head_root),
            "warm_start": bool(config.get("warm_start", False)),
            "neurotune_threshold": neurotune_threshold,
            "class_balanced": bool(config.get("class_balanced", False)),
            "class_balance_method": (
                "subgroup_sampling"
                if methods == ("DFR",)
                else "method_specific"
            ),
            "verify_hashes": bool(config["verify_hashes"]),
            "artifact_root": str(artifact_root),
            "workers": workers,
            "effective_workers": effective_workers,
            "blas_threads": blas_threads,
            "model_choices": [
                "ridge_lambda",
                *(["afr_gamma"] if "AFR" in methods else []),
                *(
                    ["neurotune_threshold"]
                    if "NEUROTUNE" in methods
                    else []
                ),
            ],
            "oracle_group_labels": any(
                method in {"DFR", "AFR"} for method in methods
            ),
        },
        "run_count": len(runs),
        "failed_run_count": sum(run.get("status") != "complete" for run in runs),
        "runs": runs,
    }


def run_comparisons(config: Mapping[str, Any]) -> list[Path]:
    experiments = _validated_experiments(config)
    overwrite = bool(config.get("overwrite", False))
    existing = [item["output"] for item in experiments if item["output"].exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite comparison results: "
            + ", ".join(str(path) for path in existing)
        )
    outputs = []
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[{index}/{len(experiments)}] comparisons: "
            f"{experiment['method']} "
            f"dataset={experiment['dataset']} "
            f"embedding={experiment['embedding']}",
            flush=True,
        )
        sweep_config = experiment["sweep_config"]
        sweep_config["methods"] = [experiment["method"]]
        sweep_config["unseeded_split_seed"] = config[
            "unseeded_split_seed"
        ]
        report = _run_comparison_sweep(sweep_config)
        output = save_sweep(
            report,
            output=experiment["output"],
            results_dir=config["results_dir"],
            overwrite=overwrite,
        )
        outputs.append(output)
        print(f"Saved: {output}", flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run frozen-feature comparison methods over final experiments."
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dfr-subsamples", type=int, help="Balanced heads averaged per DFR candidate (default 10).")
    parser.add_argument(
        "--afr-gamma", "--afr-gammas", dest="afr_gammas", nargs="+", type=float,
        help="Nonnegative AFR gamma grid, crossed with the ridge grid.",
    )
    parser.add_argument(
        "--ridge-lambda", "--ridge-lambdas", dest="ridge_lambdas",
        nargs="+", type=float,
        help="Positive ridge grid used by each requested comparison method.",
    )
    parser.add_argument("--afr-random-state", type=int)
    parser.add_argument(
        "--neurotune-threshold", "--neurotune-thresholds",
        dest="neurotune_threshold", nargs="+", type=float,
        help=(
            "NeuroTune threshold grid, crossed with ridge and selected on the "
            "identification half using exact class-balanced accuracy."
        ),
    )
    parser.add_argument(
        "--method", "--methods", dest="methods", nargs="+", type=str.upper,
        choices=tuple(COMPARISON_METHOD_REGISTRY),
        help=(
            "Comparison methods to run (configured default: NEUROTUNE). "
            "Saves separate files per method selection."
        ),
    )
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--unseeded-split-seed", type=int)
    parser.add_argument(
        "--results-dir",
        help="Write all selected comparison sweeps under this directory.",
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=None,
        help="Initialize fitted heads from the saved ERM head.",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        **COMPARISON_CONFIG,
        "methods": list(COMPARISON_CONFIG["methods"]),
        "sweep_defaults": copy.deepcopy(COMPARISON_CONFIG["sweep_defaults"]),
        "experiments": copy.deepcopy(COMPARISON_CONFIG["experiments"]),
    }
    for key in (
        "methods",
        "workers",
        "blas_threads",
        "progress_every",
        "unseeded_split_seed",
        "results_dir",
        "overwrite",
        "warm_start",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.dfr_subsamples is not None:
        if args.dfr_subsamples < 1:
            raise ValueError("--dfr-subsamples must be positive.")
        config["sweep_defaults"]["dfr_subsamples"] = args.dfr_subsamples
    if args.afr_gammas is not None:
        config["sweep_defaults"]["afr_gammas"] = list(args.afr_gammas)
    if args.ridge_lambdas is not None:
        config["sweep_defaults"]["ridge_lambdas"] = list(args.ridge_lambdas)
    if args.afr_random_state is not None:
        if args.afr_random_state < 0:
            raise ValueError("--afr-random-state must be nonnegative.")
        config["sweep_defaults"]["afr_random_state"] = args.afr_random_state
    if args.neurotune_threshold is not None:
        if not np.isfinite(args.neurotune_threshold).all():
            raise ValueError("--neurotune-threshold must be finite.")
        config["sweep_defaults"]["neurotune_threshold"] = (
            list(args.neurotune_threshold)
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    run_comparisons(resolved_config(parse_args(argv)))


if __name__ == "__main__":
    main()
