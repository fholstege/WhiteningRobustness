#!/usr/bin/env python3
"""Run configured ridge sweeps for fine-tuned representations.

This file is a thin multi-experiment wrapper around ``hyper_param_sweep.py``.
Edit ``FINETUNE_RESULTS_CONFIG`` below and run::

    python finetune_results.py --workers 3 --blas-threads 1

Each experiment produces an aggregate-compatible sweep JSON containing every
configured ridge value. Files are named
``results/<dataset>_<embedding>_finetune.json`` and are protected unless
``--overwrite`` is supplied.
Parallel workers operate on different representation seeds; ridge values for
one fitted transform remain together in the same worker. Fine-tuned
representation seeds must use the fixed dataset train/validation split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from data import EmbeddingBundle, SavedLinearHead
from evaluate import ROOT, find_bundle
from hyper_param_sweep import (
    SWEEP_CONFIG,
    _resolved_representation,
    _validated_ridge_lambdas,
    save_sweep,
    sweep_bundle,
)
from whitening import validate_whitening_estimator


# =============================================================================
# Editable fine-tuned-results configuration
# =============================================================================

FINETUNE_RESULTS_CONFIG: dict[str, Any] = {
    "artifact_root": "artifacts/embeddings",
    "saved_head_root": "artifacts/last_layers",
    "results_dir": "results_ws",
    "verify_hashes": True,
    "overwrite": False,
    "append": False,
    "warm_start": False,
    # Representation seeds run in separate processes. Keep BLAS at one thread
    # per process unless deliberately benchmarking another combination.
    "workers": 1,
    "blas_threads": 1,
    # Print one progress line after every N completed configurations per seed.
    "progress_every": 10,
    # These settings use hyper_param_sweep.py's validated ridge path.
    # Individual experiments may override any of them.
    "sweep_defaults": {
        "transforms": ["identity", "standardize", "whiten"],
        # Baselines run only once; whitening variants select the estimator.
        "whitening_variants": [
            {
                "whitening_estimator": "empirical",
                "class_weighted": False,
            },
            {
                "whitening_estimator": "ledoit-wolf-nonlinear",
                "class_weighted": False,
            },
        ],
        "ridge_lambdas":[1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0, 500000.0, 1000000.0, 5000000.0],
        # A deliberately small numerical guard: discard only eigenvalues below
        # spectral truncation. This remains close to numerical-rank whitening.
        "whitening_relative_tolerance": None
    },
    "experiments": [
        {
            "dataset": "WB",
            "embedding": "resnet50",
            "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "class_balanced": True,
        },
        # {
        #     "dataset": "WB",
        #     "embedding": "dino_vitb16",
        #     "seeds": [1, 2, 3, 4, 5],
        #     "class_balanced": True,
        # },
        {
            "dataset": "CelebA",
            "embedding": "resnet50",
            "seeds":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "class_balanced": True,
        },
        # {
        #     "dataset": "CelebA",
        #     "embedding": "dino_vitb16",
        #     "seeds": [1, 2, 3, 4, 5],
        #     "class_balanced": True,
        # },
        {
            "dataset": "multiNLI",
            "embedding": "BERT",
            "seeds": [1, 2, 3, 4, 5],
            "class_balanced": False,
        },
        # {
        #     "dataset": "multiNLI",
        #     "embedding": "debertav3",
        #     "seeds": [1, 2, 3],
        #     "class_balanced": False,
        # }
    ],
}


# =============================================================================
# Configuration resolution
# =============================================================================

def _resolved_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _filename_component(value: Any) -> str:
    component = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    if not component:
        raise ValueError(f"Invalid empty filename component from {value!r}.")
    return component


def finetune_output_path(
    *,
    results_dir: str | Path,
    dataset: str,
    embedding: str,
) -> Path:
    """Return the canonical fine-tuned sweep output path."""
    return _resolved_root(results_dir) / (
        f"{_filename_component(dataset)}_"
        f"{_filename_component(embedding)}_finetune.json"
    )


def _experiment_sweep_config(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate one editable entry into hyper_param_sweep.py configuration."""
    dataset = str(experiment["dataset"])
    embedding = str(experiment["embedding"])
    sweep_config = {
        **SWEEP_CONFIG,
        **config["sweep_defaults"],
        **{
            key: value
            for key, value in experiment.items()
            if key not in {"embedding"}
        },
        "dataset": dataset,
        "artifact_root": config["artifact_root"],
        "verify_hashes": bool(config["verify_hashes"]),
        "results_dir": config["results_dir"],
        "saved_head_root": config.get(
            "saved_head_root", FINETUNE_RESULTS_CONFIG["saved_head_root"]
        ),
        "warm_start": bool(
            experiment.get("warm_start", config.get("warm_start", False))
        ),
        "workers": experiment.get("workers", config.get("workers", 1)),
        "blas_threads": experiment.get(
            "blas_threads", config.get("blas_threads", 1)
        ),
        "progress_every": experiment.get(
            "progress_every", config.get("progress_every", 10)
        ),
        "output": None,
        "overwrite": False,
    }

    sweep_config["embedding_type"] = None
    sweep_config["representation"] = embedding
    sweep_config["seeds"] = [int(seed) for seed in experiment["seeds"]]

    # Do not share editable list objects across experiments or with module
    # defaults.
    sweep_config["transforms"] = list(sweep_config["transforms"])
    sweep_config["ridge_lambdas"] = list(sweep_config["ridge_lambdas"])
    sweep_config["whitening_variants"] = [
        dict(variant) for variant in sweep_config["whitening_variants"]
    ]
    return sweep_config


def _validated_whitening_variants(
    values: Any,
) -> tuple[dict[str, Any], ...]:
    """Normalize configured covariance estimators."""
    variants: list[dict[str, Any]] = []
    for value in values:
        estimator = validate_whitening_estimator(value["whitening_estimator"])
        variant = {"whitening_estimator": estimator}
        class_weighted = value.get("class_weighted", False)
        if not isinstance(class_weighted, bool):
            raise ValueError("whitening variant class_weighted must be boolean.")
        variant["class_weighted"] = class_weighted
        if "whitening_relative_tolerance" in value:
            variant["whitening_relative_tolerance"] = value[
                "whitening_relative_tolerance"
            ]
        variants.append(variant)
    return tuple(variants)


def _validated_parallelism(config: Mapping[str, Any]) -> tuple[int, int]:
    """Validate process and numerical-library thread counts."""
    workers = int(config.get("workers", 1))
    blas_threads = int(config.get("blas_threads", 1))
    if min(workers, blas_threads) < 1:
        raise ValueError("workers and blas_threads must be positive.")
    return workers, blas_threads


def _validated_progress_every(config: Mapping[str, Any]) -> int:
    """Validate the per-seed progress-report interval."""
    value = int(config.get("progress_every", 10))
    if value < 1:
        raise ValueError("progress_every must be a positive integer.")
    return value


def _sweep_variants(
    bundle: EmbeddingBundle,
    job: Mapping[str, Any],
    progress_callback: Any,
) -> list[dict[str, Any]]:
    """Sweep baselines once and each configured whitening variant once."""
    common = {
        "ridge_lambdas": job["ridge_lambdas"],
        "class_balanced": bool(job["class_balanced"]),
        "whitening_relative_tolerance": job["whitening_relative_tolerance"],
        "progress_callback": progress_callback,
        "erm_head": job.get("erm_head"),
        "warm_start": bool(job["warm_start"]),
    }
    runs = sweep_bundle(
        bundle,
        transforms=job["baselines"],
        whitening_estimator="empirical",
        whitening_class_weighted=False,
        **common,
    ) if job["baselines"] else []
    if job["include_whitening"]:
        for variant in job["variants"]:
            runs.extend(
                sweep_bundle(
                    bundle,
                    transforms=("whiten",),
                    whitening_estimator=variant["whitening_estimator"],
                    whitening_class_weighted=variant["class_weighted"],
                    **{
                        **common,
                        "whitening_relative_tolerance": variant.get(
                            "whitening_relative_tolerance",
                            common["whitening_relative_tolerance"],
                        ),
                    },
                )
            )
    return runs


def _run_seed_job(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load and sweep one representation seed in a worker process."""
    return _run_seed_job_with_active_setup(job)


def _run_seed_job_with_active_setup(
    job: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run one seed with the standard weighted fitting protocol."""
    context = (
        nullcontext()
        if int(job["blas_threads"]) <= 1
        else threadpool_limits(limits=int(job["blas_threads"]))
    )
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
        if job["warm_start"]:
            job = dict(job)
            train = bundle.split("train")
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
            job["erm_head"] = SavedLinearHead.load(
                saved_head_path,
                classes=np.unique(train.y),
                input_features=train.X.shape[1],
                verify_hashes=bool(job["verify_hashes"]),
            )
        total_configurations = (
            len(job["ridge_lambdas"])
            * (
                len(job["baselines"])
                + (len(job["variants"]) if job["include_whitening"] else 0)
            )
        )
        completed_configurations = 0

        def report_progress(run: dict[str, Any]) -> None:
            nonlocal completed_configurations
            completed_configurations += 1
            interval = int(job["progress_every"])
            if (
                completed_configurations % interval != 0
                and completed_configurations != total_configurations
            ):
                return
            remaining = total_configurations - completed_configurations
            print(
                f"  seed={job['seed']} progress: "
                f"{completed_configurations}/{total_configurations} complete; "
                f"{remaining} configs left; method={run['method']} "
                f"ridge={float(run['ridge_lambda']):g}",
                flush=True,
            )

        runs = _sweep_variants(bundle, job, report_progress)
        _warn_single_iteration_fits(runs, seed=job["seed"])
        return runs


def _warn_single_iteration_fits(
    runs: Sequence[Mapping[str, Any]],
    *,
    seed: Any,
) -> None:
    """Warn once per fitted head when its solver stops after one iteration."""
    warned: set[tuple[str, float]] = set()
    for run in runs:
        fit = run.get("fit", {})
        iterations = fit.get("n_iter", [])
        if not iterations or max(int(value) for value in iterations) != 1:
            continue
        key = (str(run["method"]), float(run["ridge_lambda"]))
        if key in warned:
            continue
        warned.add(key)
        warnings.warn(
            "The logistic solver stopped after a single iteration for fine-tuned fit "
            f"seed={seed}, method={key[0]}, ridge_lambda={key[1]:g}; "
            "treat this result as potentially numerically unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )


def _run_combined_sweep(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run baselines and whitening variants, optionally in seed workers."""
    representation, seeds = _resolved_representation(dict(config))
    artifact_root = _resolved_root(config["artifact_root"])
    saved_head_root = _resolved_root(
        config.get("saved_head_root", FINETUNE_RESULTS_CONFIG["saved_head_root"])
    )
    ridge_lambdas = _validated_ridge_lambdas(config["ridge_lambdas"])
    class_balanced = bool(config["class_balanced"])
    variants = _validated_whitening_variants(config["whitening_variants"])
    workers, blas_threads = _validated_parallelism(config)
    progress_every = _validated_progress_every(config)

    requested_transforms = [
        str(transform) for transform in config["transforms"]
    ]
    baselines = [
        transform for transform in requested_transforms if transform != "whiten"
    ]
    include_whitening = "whiten" in requested_transforms
    if not baselines and not include_whitening:
        raise ValueError("At least one transform must be requested.")

    jobs = [
        {
            "artifact_root": str(artifact_root),
            "dataset": str(config["dataset"]),
            "representation": representation,
            "seed": seed,
            "verify_hashes": bool(config["verify_hashes"]),
            "warm_start": bool(config.get("warm_start", False)),
            "saved_head_root": str(saved_head_root),
            "ridge_lambdas": ridge_lambdas,
            "baselines": tuple(baselines),
            "include_whitening": include_whitening,
            "variants": variants,
            "class_balanced": class_balanced,
            "whitening_relative_tolerance": config[
                "whitening_relative_tolerance"
            ],
            "blas_threads": blas_threads,
            "progress_every": progress_every,
        }
        for seed in seeds
    ]
    if not jobs:
        raise ValueError("At least one representation seed must be requested.")

    seed_runs: list[list[dict[str, Any]] | None] = [None] * len(jobs)
    effective_workers = min(workers, len(jobs))
    if effective_workers == 1:
        for index, job in enumerate(jobs):
            print(
                f"[{index + 1}/{len(jobs)}] dataset={config['dataset']}  "
                f"representation={representation}  seed={job['seed']}",
                flush=True,
            )
            seed_runs[index] = _run_seed_job(job)
    else:
        print(
            f"Running {len(jobs)} representation seeds with "
            f"{effective_workers} workers and {blas_threads} BLAS "
            f"thread(s) per worker.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_indices = {
                executor.submit(_run_seed_job, job): index
                for index, job in enumerate(jobs)
            }
            completed = 0
            for future in as_completed(future_indices):
                index = future_indices[future]
                seed_runs[index] = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(jobs)}] completed "
                    f"dataset={config['dataset']}  "
                    f"representation={representation}  "
                    f"seed={jobs[index]['seed']}",
                    flush=True,
                )

    # Futures may complete in any order; JSON runs remain ordered by the
    # configured seed list so serial and parallel outputs are comparable.
    all_runs = [
        run
        for runs_for_seed in seed_runs
        if runs_for_seed is not None
        for run in runs_for_seed
    ]

    failed = sum(run["status"] != "complete" for run in all_runs)
    return {
        "schema_version": 1,
        "kind": "ridge_lambda_sweep",
        "status": "complete" if failed == 0 else "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": config["dataset"],
            "representation": representation,
            "seeds": list(seeds),
            "transforms": requested_transforms,
            "whitening_variants": list(variants),
            "ridge_lambdas": list(ridge_lambdas),
            "class_balanced": class_balanced,
            "whitening_relative_tolerance": config[
                "whitening_relative_tolerance"
            ],
            "verify_hashes": bool(config["verify_hashes"]),
            "artifact_root": str(artifact_root),
            "saved_head_root": str(saved_head_root),
            "warm_start": bool(config.get("warm_start", False)),
            "workers": workers,
            "effective_workers": effective_workers,
            "blas_threads": blas_threads,
            "model_choices": ["ridge_lambda"],
        },
        "run_count": len(all_runs),
        "failed_run_count": failed,
        "runs": all_runs,
    }


def _validated_experiments(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    experiments = config.get("experiments")
    if not experiments:
        raise ValueError("experiments must be a non-empty list.")

    resolved: list[dict[str, Any]] = []
    for experiment in experiments:
        dataset = str(experiment["dataset"])
        embedding = str(experiment["embedding"])
        sweep_config = _experiment_sweep_config(config, experiment)
        output = finetune_output_path(
            results_dir=config["results_dir"],
            dataset=dataset,
            embedding=embedding,
        )
        resolved.append(
            {
                "dataset": dataset,
                "embedding": embedding,
                "output": output,
                "sweep_config": sweep_config,
            }
        )
    return resolved


def _validate_fixed_split_bundles(config: Mapping[str, Any]) -> None:
    """Verify that fine-tuned representation seeds share one fixed split."""
    artifact_root = _resolved_root(config["artifact_root"])
    expected_signature: tuple[Any, ...] | None = None
    for seed in config["seeds"]:
        bundle_path = find_bundle(
            artifact_root,
            dataset=str(config["dataset"]),
            representation=str(config["representation"]),
            seed=int(seed),
        )
        manifest = json.loads(
            (bundle_path / "manifest.json").read_text(encoding="utf-8")
        )
        metadata = manifest.get("metadata", {})
        if metadata.get("randomized_train_val_split") is True:
            raise ValueError(
                "Fine-tuned results require the fixed train/validation split; "
                f"randomized split metadata found in {bundle_path}."
            )
        try:
            signature = tuple(
                manifest["splits"][split][field]["sha256"]
                for split in ("train", "val")
                for field in ("y", "g")
            )
        except KeyError as exc:
            raise ValueError(
                f"Cannot verify the fixed train/validation split in {bundle_path}."
            ) from exc
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                "Fine-tuned representation seeds do not share the same fixed "
                f"train/validation split; mismatch at {bundle_path}."
            )


_APPEND_COMPATIBILITY_KEYS = (
    "dataset",
    "representation",
    "seeds",
    "transforms",
    "whitening_variants",
    "ridge_lambdas",
    "class_balanced",
    "whitening_relative_tolerance",
    "verify_hashes",
    "artifact_root",
    "saved_head_root",
    "warm_start",
    "model_choices",
)


def _run_identity(run: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify one saved transform/ridge configuration."""
    bundle = run.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("Every appended run must contain bundle metadata.")
    return (
        bundle.get("fingerprint"),
        bundle.get("representation_seed"),
        run.get("method"),
        run.get("ridge_lambda"),
    )


def _merge_sweep_reports(
    existing: Mapping[str, Any],
    additional: Mapping[str, Any],
) -> dict[str, Any]:
    """Append disjoint runs from one scientifically compatible sweep."""
    if existing.get("schema_version") != additional.get("schema_version"):
        raise ValueError("Cannot append sweeps with different schema versions.")
    if existing.get("kind") != additional.get("kind"):
        raise ValueError("Cannot append sweeps with different result kinds.")
    existing_config = existing.get("config")
    additional_config = additional.get("config")
    if not isinstance(existing_config, Mapping) or not isinstance(
        additional_config, Mapping
    ):
        raise ValueError("Both sweeps must contain configuration metadata.")
    incompatible = [
        key
        for key in _APPEND_COMPATIBILITY_KEYS
        if existing_config.get(key) != additional_config.get(key)
    ]
    if incompatible:
        raise ValueError(
            "Cannot append scientifically incompatible sweeps; differing "
            "configuration fields: " + ", ".join(incompatible)
        )

    existing_runs = existing.get("runs")
    additional_runs = additional.get("runs")
    if not isinstance(existing_runs, list) or not isinstance(
        additional_runs, list
    ):
        raise ValueError("Both sweeps must contain a runs list.")
    seen = {_run_identity(run) for run in existing_runs}
    duplicate = [
        _run_identity(run)
        for run in additional_runs
        if _run_identity(run) in seen
    ]
    if duplicate:
        raise ValueError(
            "Refusing to append duplicate fine-tuned runs; first duplicate: "
            f"{duplicate[0]!r}"
        )

    combined_runs = [*existing_runs, *additional_runs]
    failed = sum(run.get("status") != "complete" for run in combined_runs)
    return {
        **existing,
        "status": "complete" if failed == 0 else "partial",
        "updated_at": additional.get("created_at"),
        "config": dict(existing_config),
        "run_count": len(combined_runs),
        "failed_run_count": failed,
        "runs": combined_runs,
    }


# =============================================================================
# Running standard hyperparameter sweeps
# =============================================================================

def run_finetune_sweeps(config: Mapping[str, Any]) -> list[Path]:
    """Run and save aggregate-compatible sweeps for all configured entries."""
    experiments = _validated_experiments(config)
    overwrite = bool(config.get("overwrite", False))
    append = bool(config.get("append", False))
    if overwrite and append:
        raise ValueError("overwrite and append are mutually exclusive.")
    existing = [
        experiment["output"]
        for experiment in experiments
        if experiment["output"].exists()
    ]
    if existing and not overwrite and not append:
        raise FileExistsError(
            "Refusing to overwrite existing fine-tuned results: "
            + ", ".join(str(path) for path in existing)
        )

    outputs: list[Path] = []
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[{index}/{len(experiments)}] fine-tuned ridge sweep: "
            f"dataset={experiment['dataset']} "
            f"embedding={experiment['embedding']}",
            flush=True,
        )
        _validate_fixed_split_bundles(experiment["sweep_config"])
        report = _run_combined_sweep(experiment["sweep_config"])
        report["config"]["result_family"] = "finetune"
        report["config"]["train_validation_split"] = "fixed"
        for run in report["runs"]:
            run["result_family"] = "finetune"
            run["protocol"]["train_validation_split"] = "fixed"
        if append and experiment["output"].exists():
            with experiment["output"].open("r", encoding="utf-8") as handle:
                report = _merge_sweep_reports(json.load(handle), report)
        output = save_sweep(
            report,
            output=experiment["output"],
            results_dir=config["results_dir"],
            overwrite=overwrite or append,
        )
        outputs.append(output)
        print(f"Saved: {output}", flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run fixed-split sweeps for fine-tuned representations."
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Representation-seed worker processes; 1 runs serially.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        help="BLAS/OpenMP threads available inside each worker.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        help="Print per-seed progress after this many configurations (default: 10).",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=None,
        help="Initialize each fitted head from the saved ERM head.",
    )
    parser.add_argument(
        "--dataset",
        help="Run only configured experiments for this dataset.",
    )
    parser.add_argument(
        "--embedding",
        help="Run only the configured experiment with this embedding.",
    )
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help=(
            "Atomically replace existing "
            "setup-specific fine-tuned result files."
        ),
    )
    write_mode.add_argument(
        "--append",
        action="store_true",
        default=None,
        help="Atomically append nonduplicate runs to compatible result files.",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Apply small runtime overrides without mutating editable defaults."""
    config = dict(FINETUNE_RESULTS_CONFIG)
    config["sweep_defaults"] = dict(
        FINETUNE_RESULTS_CONFIG["sweep_defaults"]
    )
    config["experiments"] = [
        dict(experiment)
        for experiment in FINETUNE_RESULTS_CONFIG["experiments"]
    ]
    if args.dataset is not None:
        config["experiments"] = [
            experiment
            for experiment in config["experiments"]
            if str(experiment["dataset"]).casefold()
            == str(args.dataset).casefold()
        ]
        if not config["experiments"]:
            raise ValueError(
                f"No configured experiments match dataset={args.dataset!r}."
            )
    if args.embedding is not None:
        config["experiments"] = [
            experiment
            for experiment in config["experiments"]
            if str(experiment["embedding"]).casefold()
            == str(args.embedding).casefold()
        ]
        if not config["experiments"]:
            raise ValueError(
                "No configured experiments match "
                f"embedding={args.embedding!r}."
            )
    if args.workers is not None:
        config["workers"] = args.workers
    if args.blas_threads is not None:
        config["blas_threads"] = args.blas_threads
    if args.progress_every is not None:
        config["progress_every"] = args.progress_every
    if args.warm_start is not None:
        config["warm_start"] = args.warm_start
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite
    if args.append is not None:
        config["append"] = args.append
    return config


def main(argv: Sequence[str] | None = None) -> None:
    run_finetune_sweeps(resolved_config(parse_args(argv)))


if __name__ == "__main__":
    main()
