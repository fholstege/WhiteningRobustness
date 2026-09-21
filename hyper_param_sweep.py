#!/usr/bin/env python3
"""Sweep ridge penalties for fixed logistic heads on frozen embeddings.

This script deliberately varies one model quantity: ``ridge_lambda``.  The
optimizer, stopping tolerance, and maximum iteration count remain fixed in
``model.py``.  Each representation transform is fitted once on train and then
reused for every ridge value, which makes a sweep substantially cheaper than
calling ``evaluate.py`` repeatedly.

Edit ``SWEEP_CONFIG`` below or override its fields from the command line.  The
output is one JSON file consumed by ``aggregate.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from data import EmbeddingBundle, SPLITS
from evaluate import (
    ROOT,
    _normalized_identifier,
    find_bundle,
)
from metric import evaluation_metrics, selection_metric_statistics
from model import (
    LogisticConfig,
    LogisticConvergenceError,
    fit_logistic,
    fit_logistic_from_reference,
    reference_head_in_transform,
)
from numerics import OPTIMIZATION_DTYPE
from whitening import (
    make_transform,
    validate_whitening_estimator,
)


# =============================================================================
# Sweep configuration
# =============================================================================

SWEEP_CONFIG = {
    "dataset": "WB",
    "embedding_type": "resnet50",
    "representation": None,         # direct manifest-name override
    "seeds": [1, 2, 3, 4, 5],      # DINO is unseeded; these are then ignored
    "transforms": ["identity", "standardize", "whiten"],
    # Covariance estimator used when the requested transform is "whiten".
    # "empirical", "ledoit-wolf", or "ledoit-wolf-nonlinear"
    "whitening_estimator": "empirical",
    # The sole hyperparameter grid. Values are relative to average log loss.
    "ridge_lambdas": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "class_balanced": False,
    # None uses the numerical-rank default documented in whitening.py.
    "whitening_relative_tolerance": None,
    "verify_hashes": True,
    "artifact_root": "artifacts/embeddings",
    "results_dir": "results/ridge_sweeps",
    "output": None,                 # optional exact JSON path
    # Existing output files are protected unless this is explicitly enabled.
    "overwrite": False,
}


# =============================================================================
# Small validation and identity helpers
# =============================================================================

def _validated_ridge_lambdas(values: Iterable[float]) -> tuple[float, ...]:
    """Return a unique, ordered sequence of finite positive ridge values."""
    ridge_lambdas = tuple(float(value) for value in values)
    if not ridge_lambdas:
        raise ValueError("At least one ridge_lambda must be requested.")
    if any(not np.isfinite(value) or value <= 0 for value in ridge_lambdas):
        raise ValueError("Every ridge_lambda must be finite and positive.")
    if len(set(ridge_lambdas)) != len(ridge_lambdas):
        raise ValueError("ridge_lambdas contains duplicate values.")
    return ridge_lambdas


def _resolved_representation(
    config: dict[str, Any],
) -> tuple[str, tuple[int | None, ...]]:
    """Resolve the manifest representation and embedding seeds to evaluate."""
    direct = config.get("representation")
    if direct:
        seeds = tuple(
            None if seed is None else int(seed) for seed in config["seeds"]
        )
        return str(direct), seeds

    embedding_type = str(config["embedding_type"]).lower()
    if embedding_type == "resnet50":
        return "resnet50", tuple(int(seed) for seed in config["seeds"])
    raise ValueError(
        "embedding_type must be 'resnet50', or set representation."
    )


def _bundle_identity(bundle: EmbeddingBundle) -> dict[str, Any]:
    """Keep enough manifest identity to detect accidental sweep mixing."""
    return {
        "path": str(bundle.root),
        "fingerprint": bundle.fingerprint,
        "dataset": bundle.manifest.get("dataset"),
        "representation": bundle.manifest.get("representation"),
        "representation_seed": bundle.manifest.get("representation_seed"),
    }


# =============================================================================
# One-bundle ridge sweep
# =============================================================================

def _completed_run_report(
    *,
    bundle: EmbeddingBundle,
    transformed: dict[str, np.ndarray],
    transform_report: dict[str, Any],
    method: str,
    requested_name: str,
    ridge_lambda: float,
    class_balanced: bool,
    fit: Any,
) -> dict[str, Any]:
    """Build one fitted ridge run from a shared transformed bundle."""
    train = bundle.split("train")
    logistic_config = LogisticConfig(
        ridge_lambda=ridge_lambda,
        class_balanced=class_balanced,
    )
    run: dict[str, Any] = {
        "status": "complete" if fit.converged else "failed",
        "bundle": _bundle_identity(bundle),
        "method": method,
        "requested_transform": requested_name,
        "ridge_lambda": ridge_lambda,
        "C": logistic_config.C,
        "sklearn_C": logistic_config.sklearn_C(train.y.size),
        "class_balanced": logistic_config.class_balanced,
        "transform": transform_report,
        "fit": {
            "fit_split": "train",
            "observations": int(train.y.size),
            "uses_group_labels": False,
            **fit.diagnostics(),
        },
        "protocol": {
            "fit_split": "train",
            "selection_split": "val",
            "group_labels_used_for_fitting": False,
            "group_labels_used_for_model_selection": False,
        },
    }
    if not fit.converged:
        run["error"] = "; ".join(fit.convergence_messages)
        run["selection"] = {}
        run["splits"] = {}
        return run

    split_reports: dict[str, dict[str, Any]] = {}
    for split_name in SPLITS:
        split = bundle.split(split_name)
        split_reports[split_name] = evaluation_metrics(
            fit.estimator,
            transformed[split_name],
            split.y,
            split.g,
        )
    run["splits"] = split_reports
    validation = bundle.split("val")
    labels, label_counts = np.unique(validation.y, return_counts=True)
    validation_metrics, validation_standard_errors = (
        selection_metric_statistics(
            fit.estimator,
            transformed["val"],
            validation.y,
        )
    )
    run["selection"] = {
        "split": "val",
        "label_counts": {
            str(int(label)): int(count)
            for label, count in zip(labels, label_counts)
        },
        "metrics": validation_metrics,
        "standard_errors": validation_standard_errors,
    }
    return run


def _failed_convergence_run_report(
    *,
    bundle: EmbeddingBundle,
    transform_report: dict[str, Any],
    method: str,
    requested_name: str,
    ridge_lambda: float,
    class_balanced: bool,
    error: LogisticConvergenceError,
) -> dict[str, Any]:
    """Serialize one unusable ridge candidate without evaluation metrics."""
    train = bundle.split("train")
    logistic_config = LogisticConfig(
        ridge_lambda=ridge_lambda,
        class_balanced=class_balanced,
    )
    attempts = [dict(attempt) for attempt in error.solver_attempts]
    return {
        "status": "failed",
        "error": str(error),
        "bundle": _bundle_identity(bundle),
        "method": method,
        "requested_transform": requested_name,
        "ridge_lambda": ridge_lambda,
        "C": logistic_config.C,
        "sklearn_C": logistic_config.sklearn_C(train.y.size),
        "class_balanced": class_balanced,
        "transform": transform_report,
        "fit": {
            "fit_split": "train",
            "observations": int(train.y.size),
            "uses_group_labels": False,
            "converged": False,
            "convergence_messages": list(error.convergence_messages),
            "solver_attempts": attempts,
            "n_iter": attempts[-1]["n_iter"],
        },
        "protocol": {
            "fit_split": "train",
            "selection_split": "val",
            "group_labels_used_for_fitting": False,
            "group_labels_used_for_model_selection": False,
        },
        "selection": {},
        "splits": {},
    }

def sweep_bundle(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    class_balanced: bool,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    erm_head: Any | None = None,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fit all transform-by-ridge combinations for one embedding bundle.

    Preprocessing and logistic fitting use train only.  The separate
    ``selection`` object contains only label-based validation metrics and has
    no group argument.  Group-aware metrics remain available under ``splits``
    for final reporting, but ``aggregate.py`` never reads them for selection.
    """
    requested_transforms = tuple(transforms)
    if not requested_transforms:
        raise ValueError("At least one transform must be requested.")
    ridge_values = _validated_ridge_lambdas(ridge_lambdas)
    if not isinstance(warm_start, bool):
        raise TypeError("warm_start must be boolean.")
    if warm_start and erm_head is None:
        raise ValueError("warm_start=True requires a fitted ERM head.")
    train = bundle.split("train")
    runs: list[dict[str, Any]] = []
    canonical_names: set[str] = set()

    for requested_name in requested_transforms:
        # This is intentionally outside the ridge loop. The same fitted
        # coordinate system must be used for every penalty being compared.
        transform = make_transform(
            requested_name,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            class_weighted=whitening_class_weighted,
        )
        transform = (
            transform.fit(train.X, train.y)
            if whitening_class_weighted
            else transform.fit(train.X)
        )
        transform_report = transform.diagnostics()
        method = str(transform_report["name"])
        if method in canonical_names:
            raise ValueError(
                f"Transform {requested_name!r} duplicates method {method!r}."
            )
        canonical_names.add(method)

        # Hold only one transform's three arrays at a time. They are reused
        # over ridge values and released when the next transform starts.
        transformed = {
            split_name: transform.transform(bundle.split(split_name).X)
            for split_name in SPLITS
        }
        # Convert train once per fitted transform and reuse the same contiguous
        # float64 optimization matrix over the complete ridge path. The
        # exact float32 transformed values are preserved by this cast. Replacing
        # the dictionary entry also releases its float32 train array before the
        # repeated fits, keeping only float32 validation/test arrays alongside
        # the float64 training matrix.
        transformed["train"] = np.asarray(
            transformed["train"], dtype=OPTIMIZATION_DTYPE, order="C"
        )
        reference_head = (
            reference_head_in_transform(erm_head, transform)
            if warm_start
            else None
        )

        for ridge_lambda in ridge_values:
            try:
                logistic_config = LogisticConfig(
                    ridge_lambda=ridge_lambda,
                    class_balanced=class_balanced,
                )
                fit = (
                    fit_logistic_from_reference(
                        transformed["train"],
                        train.y,
                        logistic_config,
                        reference_head,
                        proximity_penalty=False,
                    )
                    if warm_start
                    else fit_logistic(
                        transformed["train"], train.y, logistic_config
                    )
                )
            except LogisticConvergenceError as error:
                failed_run = _failed_convergence_run_report(
                    bundle=bundle,
                    transform_report=transform_report,
                    method=method,
                    requested_name=requested_name,
                    ridge_lambda=ridge_lambda,
                    class_balanced=class_balanced,
                    error=error,
                )
                runs.append(failed_run)
                if progress_callback is not None:
                    progress_callback(failed_run)
                continue
            runs.append(
                completed_run := _completed_run_report(
                    bundle=bundle,
                    transformed=transformed,
                    transform_report=transform_report,
                    method=method,
                    requested_name=requested_name,
                    ridge_lambda=ridge_lambda,
                    class_balanced=class_balanced,
                    fit=fit,
                )
            )
            if progress_callback is not None:
                progress_callback(completed_run)

        if not any(
            run["method"] == method and run["status"] == "complete"
            for run in runs
        ):
            raise RuntimeError(
                "No converged ridge candidate for "
                f"method={method!r} in embedding bundle {bundle.root}; "
                "failing the complete seed experiment."
            )

    return runs


def run_sweep(config: dict[str, Any]) -> dict[str, Any]:
    """Load every requested bundle and collect one portable sweep report."""
    representation, seeds = _resolved_representation(config)
    artifact_root = Path(config["artifact_root"]).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    ridge_lambdas = _validated_ridge_lambdas(config["ridge_lambdas"])

    all_runs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        bundle_path = find_bundle(
            artifact_root,
            dataset=config["dataset"],
            representation=representation,
            seed=seed,
        )
        bundle = EmbeddingBundle.load(
            bundle_path,
            verify_hashes=bool(config["verify_hashes"]),
        )
        print(
            f"[{index}/{len(seeds)}] dataset={config['dataset']}  "
            f"representation={representation}  seed={seed}"
        )
        all_runs.extend(
            sweep_bundle(
                bundle,
                transforms=config["transforms"],
                ridge_lambdas=ridge_lambdas,
                class_balanced=bool(config["class_balanced"]),
                whitening_relative_tolerance=config[
                    "whitening_relative_tolerance"
                ],
                whitening_estimator=config["whitening_estimator"],
            )
        )

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
            "transforms": list(config["transforms"]),
            "ridge_lambdas": list(ridge_lambdas),
            "class_balanced": bool(config["class_balanced"]),
            "whitening_relative_tolerance": config[
                "whitening_relative_tolerance"
            ],
            "whitening_estimator": validate_whitening_estimator(
                config["whitening_estimator"]
            ),
            "verify_hashes": bool(config["verify_hashes"]),
            "artifact_root": str(artifact_root.resolve()),
            "model_choices": ["ridge_lambda"],
        },
        "run_count": len(all_runs),
        "failed_run_count": failed,
        "runs": all_runs,
    }


# =============================================================================
# Saving and command-line overrides
# =============================================================================

def _default_output(report: dict[str, Any], results_dir: str | Path) -> Path:
    """Create a readable, collision-resistant default result filename."""
    config = report["config"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dataset = _normalized_identifier(config["dataset"])
    representation = _normalized_identifier(config["representation"])
    root = Path(results_dir).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    return root / dataset / f"sweep_{representation}_{timestamp}.json"


def save_sweep(
    report: dict[str, Any],
    *,
    output: str | Path | None,
    results_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write a sweep JSON, optionally replacing an existing result atomically."""
    if output is None:
        path = _default_output(report, results_dir)
    else:
        path = Path(output).expanduser()
        if not path.is_absolute():
            path = ROOT / path
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sweep: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if overwrite:
        # Write beside the destination and rename only after serialization
        # succeeds. Path.replace is atomic when both files share a filesystem,
        # so a failed rerun cannot leave a half-written sweep at ``path``.
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(contents)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        # Exclusive creation also protects against another process creating
        # the same output between the existence check and this write.
        with path.open("x", encoding="utf-8") as handle:
            handle.write(contents)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Sweep only ridge_lambda for fixed logistic heads."
    )
    parser.add_argument("--dataset")
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument(
        "--embedding-type",
        choices=("resnet50",),
    )
    representation.add_argument("--representation")
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        nargs="+",
        help="One or more embedding seeds. Ignored for DINO.",
    )
    parser.add_argument(
        "--transform",
        dest="transforms",
        nargs="+",
    )
    parser.add_argument(
        "--ridge-lambda",
        dest="ridge_lambdas",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--whitening-estimator",
        type=validate_whitening_estimator,
        help=(
            "Covariance estimator for whitening: empirical, ledoit-wolf, "
            "or ledoit-wolf-nonlinear."
        ),
    )
    parser.add_argument(
        "--whitening-relative-tolerance",
        type=float,
    )
    parser.add_argument(
        "--class-balanced",
        dest="class_balanced",
        action="store_true",
        default=None,
    )
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--verify-hashes",
        dest="verify_hashes",
        action="store_true",
        default=None,
    )
    verification.add_argument(
        "--skip-hash-verification",
        dest="verify_hashes",
        action="store_false",
    )
    parser.add_argument("--artifact-root")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Atomically replace --output if that sweep JSON already exists.",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line values without mutating the editable defaults."""
    config = dict(SWEEP_CONFIG)
    config["seeds"] = list(SWEEP_CONFIG["seeds"])
    config["transforms"] = list(SWEEP_CONFIG["transforms"])
    config["ridge_lambdas"] = list(SWEEP_CONFIG["ridge_lambdas"])
    for key in (
        "dataset",
        "embedding_type",
        "representation",
        "seeds",
        "transforms",
        "ridge_lambdas",
        "class_balanced",
        "whitening_estimator",
        "whitening_relative_tolerance",
        "verify_hashes",
        "artifact_root",
        "results_dir",
        "output",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.embedding_type is not None:
        config["representation"] = None
    if args.representation is not None:
        config["embedding_type"] = None
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = resolved_config(parse_args(argv))
    report = run_sweep(config)
    path = save_sweep(
        report,
        output=config["output"],
        results_dir=config["results_dir"],
        overwrite=bool(config["overwrite"]),
    )
    completed = report["run_count"] - report["failed_run_count"]
    print(
        f"Completed {completed}/{report['run_count']} configurations.\n"
        f"Saved: {path}"
    )


if __name__ == "__main__":
    main()
