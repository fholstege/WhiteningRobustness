#!/usr/bin/env python3
"""Evaluate frozen embeddings with a fixed scikit-learn logistic head.

Edit ``EVAL_CONFIG`` below or override individual values on the command line.
ERM fits on train; comparison methods use their documented validation
adaptation splits. Except for oracle DFR, group labels are reporting-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

from data import EmbeddingBundle, SavedLinearHead, SPLITS
from methods import (
    validate_fitting_method,
    sweep_afr_bundle,
    sweep_dfr_bundle,
)
from metric import evaluation_metrics
from model import LogisticConfig, fit_logistic
from whitening import (
    make_transform,
    validate_whitening_estimator,
)


ROOT = Path(__file__).resolve().parent

# =============================================================================
# Evaluation configuration
# =============================================================================

EVAL_CONFIG = {
    "dataset": "WB",
    "method": "ERM",
    "embedding_type": "resnet50",
    "representation": None,         # direct override, e.g. "bert"
    "seed": 1,
    "transforms": ["identity", "standardize", "whiten"],
    # Covariance estimator used when the requested transform is "whiten".
    # "empirical", "ledoit-wolf", or "ledoit-wolf-nonlinear"
    "whitening_estimator": "empirical",
    # None uses float32 machine epsilon. Try e.g. 1e-5 explicitly.
    "whitening_relative_tolerance": None,
    "ridge_lambda": 0.001,          # average-loss ridge; reported C = 1/lambda
    "class_balanced": False,        # ordinary ERM unless explicitly enabled
    "afr_gamma": 4.0,
    "afr_random_state": 0,
    "verify_hashes": True,
    "artifact_root": "artifacts/embeddings",
    "saved_head_root": "artifacts/last_layers",
    "output": None,                 # optional JSON path
}


# =============================================================================
# Embedding selection
# =============================================================================

def _normalized_identifier(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return {"waterbird": "wb", "waterbirds": "wb"}.get(
        normalized, normalized
    )


def find_bundle(
    artifact_root: str | Path,
    *,
    dataset: str,
    representation: str,
    seed: int | None,
) -> Path:
    """Find exactly one NumPy bundle using its manifest identity."""
    root = Path(artifact_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Embedding artifact root does not exist: {root}")

    requested_dataset = _normalized_identifier(dataset)
    requested_representation = _normalized_identifier(representation)
    matches: list[Path] = []
    available: list[str] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_dataset = manifest.get("dataset")
        manifest_representation = manifest.get("representation")
        manifest_seed = manifest.get("representation_seed")
        available.append(
            f"{manifest_dataset}/{manifest_representation}/seed={manifest_seed}"
        )
        if (
            _normalized_identifier(manifest_dataset) == requested_dataset
            and _normalized_identifier(manifest_representation)
            == requested_representation
            and manifest_seed == seed
        ):
            matches.append(manifest_path.parent)

    if matches:
        # An artifact root may contain named alternative collections such as
        # ``robustness/``. Prefer the nearest matching bundle so the default
        # collection remains usable, while pointing ``--artifact-root`` at an
        # alternative collection selects its bundles normally.
        minimum_depth = min(len(path.relative_to(root).parts) for path in matches)
        matches = [
            path
            for path in matches
            if len(path.relative_to(root).parts) == minimum_depth
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Bundle identity is ambiguous: "
            + ", ".join(str(path) for path in matches)
        )
    choices = "\n  ".join(available)
    raise FileNotFoundError(
        "No embedding bundle matches "
        f"dataset={dataset!r}, representation={representation!r}, seed={seed!r}."
        f"\nAvailable bundles:\n  {choices}"
    )


def _resolve_representation(config: dict[str, Any]) -> tuple[str, int | None]:
    direct = config.get("representation")
    if direct:
        return str(direct), config.get("seed")

    embedding_type = str(config["embedding_type"]).lower()
    if embedding_type == "resnet50":
        return "resnet50", config.get("seed")
    raise ValueError(
        "embedding_type must be 'resnet50', or set representation."
    )


# =============================================================================
# Fitting and evaluation
# =============================================================================

def evaluate(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambda: float,
    class_balanced: bool,
    method: str = "ERM",
    dfr_split_seed: int | None = None,
    afr_gamma: float = 4.0,
    afr_random_state: int = 0,
    saved_head: SavedLinearHead | None = None,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
) -> dict[str, Any]:
    """Fit each method on train and report all three splits."""
    transform_names = tuple(transforms)
    if not transform_names:
        raise ValueError("At least one transform must be requested.")

    fitting_method = validate_fitting_method(method)
    if fitting_method == "AFR":
        if saved_head is None:
            raise ValueError("AFR requires the matching saved trained head.")
        if class_balanced:
            raise ValueError(
                "AFR supplies its own sample weights; do not enable "
                "inverse-class-frequency weighting."
            )
        runs = sweep_afr_bundle(
            bundle,
            transforms=transform_names,
            ridge_lambdas=(ridge_lambda,),
            score_head=saved_head,
            afr_gammas=(afr_gamma,),
            random_state=afr_random_state,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            whitening_class_weighted=whitening_class_weighted,
        )
        failed = [run for run in runs if run["status"] != "complete"]
        if failed:
            raise RuntimeError(
                "AFR did not converge: "
                + "; ".join(str(run.get("error")) for run in failed)
            )
        first = runs[0]
        return {
            "schema_version": 1,
            "status": "complete",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "bundle": first["bundle"],
            "head": {
                "method": "AFR",
                "ridge_lambda": float(ridge_lambda),
                "C": 1.0 / float(ridge_lambda),
                "sklearn_C": first["sklearn_C"],
                "regularization_convention": "average_log_loss",
                "class_balanced": True,
                "class_balance_method": (
                    "afr_inverse_class_frequency_confidence_weight"
                ),
                "afr_gamma": float(afr_gamma),
                "afr_random_state": int(afr_random_state),
                "score_head": saved_head.diagnostics(),
            },
            "protocol": {
                "method": "AFR",
                "fit_split": "val_afr_adaptation_half",
                "evaluation_splits": list(SPLITS),
                "selection_split": "val_afr_selection_half",
                "refit_on_validation": True,
                **first["protocol"],
            },
            "evaluations": [
                {
                    "method": run["method"],
                    "method_family": "afr",
                    "requested_transform": run["requested_transform"],
                    "transform": run["transform"],
                    "fit": run["fit"],
                    "afr": run["afr"],
                    "selection": run["selection"],
                    "splits": run["splits"],
                }
                for run in runs
            ],
        }
    if fitting_method == "DFR":
        if class_balanced:
            raise ValueError(
                "DFR uses subgroup sampling; do not enable "
                "inverse-class-frequency weighting."
            )
        representation_seed = bundle.manifest.get("representation_seed")
        split_seed = (
            int(dfr_split_seed)
            if dfr_split_seed is not None
            else (
                int(representation_seed)
                if representation_seed is not None
                else 0
            )
        )
        runs = sweep_dfr_bundle(
            bundle,
            transforms=transform_names,
            ridge_lambdas=(ridge_lambda,),
            split_seed=split_seed,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            whitening_class_weighted=whitening_class_weighted,
        )
        first = runs[0]
        return {
            "schema_version": 1,
            "status": "complete",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "bundle": first["bundle"],
            "head": {
                "method": "DFR",
                "ridge_lambda": float(ridge_lambda),
                "C": 1.0 / float(ridge_lambda),
                "sklearn_C": first["sklearn_C"],
                "regularization_convention": "average_log_loss",
                "class_balanced": True,
                "class_balance_method": "subgroup_sampling",
            },
            "protocol": {
                "method": "DFR",
                "fit_split": "dfr_train_balanced",
                "evaluation_splits": list(SPLITS),
                "selection_split": "dfr_val",
                "original_train_split_used": False,
                "refit_on_validation": True,
                "group_labels_used_for_fitting": True,
                "group_labels_used_for_model_selection": True,
                "test_group_labels_used_for_model_selection": False,
                "oracle": True,
                "dfr_split_seed": split_seed,
            },
            "evaluations": [
                {
                    "method": run["method"],
                    "method_family": "dfr",
                    "requested_transform": run["requested_transform"],
                    "transform": run["transform"],
                    "fit": run["fit"],
                    "dfr": run["dfr"],
                    "selection": run["selection"],
                    "splits": run["splits"],
                }
                for run in runs
            ],
        }

    train = bundle.split("train")
    logistic_config = LogisticConfig(
        ridge_lambda=ridge_lambda,
        class_balanced=class_balanced,
    )
    evaluations = []
    for requested_name in transform_names:
        # Only whitening uses the spectral tolerance. Passing it through this
        # single factory keeps every method on the same evaluation path.
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
        X_train = transform.transform(train.X)
        fit = fit_logistic(X_train, train.y, logistic_config)
        if not fit.converged:
            messages = "; ".join(fit.convergence_messages)
            raise RuntimeError(f"{method} did not converge: {messages}")

        split_reports = {}
        for split_name in SPLITS:
            split = bundle.split(split_name)
            X = (
                X_train
                if split_name == "train"
                else transform.transform(split.X)
            )
            split_reports[split_name] = evaluation_metrics(
                fit.estimator, X, split.y, split.g
            )
        evaluations.append(
            {
                "method": method,
                "requested_transform": requested_name,
                "transform": transform_report,
                "fit": {
                    "fit_split": "train",
                    "observations": int(train.y.size),
                    "uses_group_labels": False,
                    **fit.diagnostics(),
                },
                "splits": split_reports,
            }
        )

    return {
        "schema_version": 1,
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": {
            "path": str(bundle.root),
            "fingerprint": bundle.fingerprint,
            "dataset": bundle.manifest.get("dataset"),
            "representation": bundle.manifest.get("representation"),
            "representation_seed": bundle.manifest.get("representation_seed"),
        },
        "head": {
            "method": "ERM",
            "ridge_lambda": logistic_config.ridge_lambda,
            "C": logistic_config.C,
            "sklearn_C": logistic_config.sklearn_C(train.y.size),
            "regularization_convention": "average_log_loss",
            "class_balanced": logistic_config.class_balanced,
            "class_balance_method": evaluations[0]["fit"][
                "class_balance_method"
            ],
        },
        "protocol": {
            "fit_split": "train",
            "evaluation_splits": list(SPLITS),
            "refit_on_validation": False,
            "group_labels_used_for_fitting": False,
        },
        "evaluations": evaluations,
    }


# =============================================================================
# Reporting
# =============================================================================

def format_results(report: dict[str, Any]) -> str:
    """Render one method-by-metric table for each split."""
    bundle = report["bundle"]
    head = report["head"]
    lines = [
        (
            f"dataset={bundle['dataset']}  "
            f"representation={bundle['representation']}  "
            f"seed={bundle['representation_seed']}"
        ),
        (
            f"ridge_lambda={head['ridge_lambda']:.8g}  C={head['C']:.8g}  "
            f"(sklearn_C={head['sklearn_C']:.8g})  "
            f"class_balance={head['class_balance_method']}"
            + (
                f"  afr_gamma={head['afr_gamma']:.8g}"
                f"  afr_seed={head['afr_random_state']}"
                if head.get("afr_gamma") is not None
                else ""
            )
        ),
        "",
    ]
    if report["protocol"].get("method") == "DFR":
        lines.extend(
            (
                "DFR oracle protocol: TRAIN is the group-balanced sample from",
                "the first validation half; VAL is the held-out selection half;",
                "TEST is the unchanged bundle test split.",
                "",
            )
        )
    elif report["protocol"].get("method") == "AFR":
        lines.extend(
            (
                "AFR protocol: the saved train head assigns fixed confidence",
                "and inverse-class-frequency weights on one validation half;",
                "L-BFGS fits an ERM-centered correction in transform coordinates;",
                "the other validation half selects gamma and ridge by",
                "worst-group accuracy. Test groups remain evaluation-only.",
                "",
            )
        )

    for split_name in SPLITS:
        group_ids = sorted(
            {
                int(group)
                for evaluation in report["evaluations"]
                for group in evaluation["splits"][split_name]["per_group"]
            }
        )
        headers = (
            "method",
            "overall",
            "equal-group",
            "worst-group",
            "balanced-LL",
            "worst-LL",
            *(f"g={group}" for group in group_ids),
        )
        rows = []
        for evaluation in report["evaluations"]:
            metrics = evaluation["splits"][split_name]
            per_group = metrics["per_group"]
            rows.append(
                (
                    evaluation["method"],
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['group_balanced_accuracy']:.4f}",
                    f"{metrics['worst_group_accuracy']:.4f}",
                    f"{metrics['equal_group_log_loss']:.4f}",
                    f"{metrics['worst_group_log_loss']:.4f}",
                    *(
                        (
                            f"{per_group[str(group)]['accuracy']:.4f}"
                            if str(group) in per_group
                            else "n/a"
                        )
                        for group in group_ids
                    ),
                )
            )
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers))
        ]

        def render(values: Sequence[str]) -> str:
            return "  ".join(
                value.ljust(widths[index])
                for index, value in enumerate(values)
            )

        lines.extend(
            (
                split_name.upper(),
                render(headers),
                render(tuple("-" * width for width in widths)),
                *(render(row) for row in rows),
                "",
            )
        )

    lines.extend(
        (
            "Accuracy columns are fractions; g=* columns are per-group accuracy.",
            "LL means log loss: balanced-LL weights groups equally and worst-LL is",
            "the largest group log loss. Lower LL is better.",
        )
    )
    return "\n".join(lines)


def save_results(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# =============================================================================
# Command-line overrides
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Fit fixed logistic heads on train and report train/val/test metrics."
        )
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--method",
        type=validate_fitting_method,
        help=(
            "Head-fitting protocol: ERM (default), AFR, NeuroTune, or oracle "
            "DFR."
        ),
    )
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument(
        "--embedding-type",
        choices=(
            "resnet50",
        ),
    )
    representation.add_argument(
        "--representation",
        help="Direct manifest representation name, for example bert.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dfr-split-seed",
        type=int,
        help=(
            "DFR validation split seed; defaults to the representation seed "
            "or zero for an unseeded representation."
        ),
    )
    parser.add_argument(
        "--transform",
        dest="transforms",
        nargs="+",
    )
    parser.add_argument(
        "--ridge-lambda",
        dest="ridge_lambda",
        type=float,
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
        help=(
            "Discard raw empirical covariance directions at or below this "
            "fraction of the largest raw eigenvalue before whitening or "
            "covariance shrinkage."
        ),
    )
    parser.add_argument(
        "--class-balanced",
        dest="class_balanced",
        action="store_true",
        default=None,
        help=(
            "Use mean-one inverse-class-frequency sample weights."
        ),
    )
    parser.add_argument(
        "--afr-gamma",
        type=float,
        help="Nonnegative AFR confidence-weighting strength.",
    )
    parser.add_argument("--afr-random-state", type=int)
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
    parser.add_argument("--saved-head-root")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Apply only explicitly supplied command-line overrides."""
    config = dict(EVAL_CONFIG)
    for key in (
        "dataset",
        "method",
        "embedding_type",
        "representation",
        "seed",
        "dfr_split_seed",
        "transforms",
        "whitening_estimator",
        "whitening_relative_tolerance",
        "ridge_lambda",
        "class_balanced",
        "afr_gamma",
        "afr_random_state",
        "verify_hashes",
        "artifact_root",
        "saved_head_root",
        "output",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    validate_fitting_method(config["method"])

    if args.embedding_type is not None:
        config["representation"] = None
    if args.representation is not None:
        config["embedding_type"] = None
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = resolved_config(parse_args(argv))
    representation, seed = _resolve_representation(config)
    artifact_root = Path(config["artifact_root"]).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
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
    saved_head = None
    if validate_fitting_method(config["method"]) == "AFR":
        saved_head_root = Path(config["saved_head_root"]).expanduser()
        if not saved_head_root.is_absolute():
            saved_head_root = ROOT / saved_head_root
        saved_head_path = (
            saved_head_root / str(config["dataset"]) / representation /
            ("default" if seed is None else f"seed_{int(seed):03d}")
        )
        saved_head = SavedLinearHead.load(
            saved_head_path,
            classes=np.unique(bundle.split("train").y),
            input_features=bundle.split("train").X.shape[1],
            verify_hashes=bool(config["verify_hashes"]),
        )
    report = evaluate(
        bundle,
        transforms=config["transforms"],
        ridge_lambda=float(config["ridge_lambda"]),
        class_balanced=bool(config["class_balanced"]),
        method=config["method"],
        dfr_split_seed=config.get("dfr_split_seed"),
        afr_gamma=float(config["afr_gamma"]),
        afr_random_state=int(config["afr_random_state"]),
        saved_head=saved_head,
        whitening_relative_tolerance=config[
            "whitening_relative_tolerance"
        ],
        whitening_estimator=config["whitening_estimator"],
    )
    print(format_results(report))
    if config["output"] is not None:
        print(f"\nSaved: {save_results(report, config['output'])}")


if __name__ == "__main__":
    main()
