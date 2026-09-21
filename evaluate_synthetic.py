#!/usr/bin/env python3
"""Evaluate the standard synthetic model like ``evaluate.py``.

The filename intentionally follows the requested spelling. Edit
``SYNTHETIC_CONFIG`` below or override individual settings on the command
line. Every representation transform and logistic head is fitted on the
synthetic training environment and then reused unchanged on the test
environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from metric import evaluation_metrics
from numerics import SYNTHETIC_DTYPE
from model import LogisticConfig, fit_logistic
from synthetic import SyntheticConfig, generate_train_test
from whitening import (
    make_transform,
    validate_whitening_estimator,
)


ROOT = Path(__file__).resolve().parent
SYNTHETIC_SPLITS = ("train", "test")


# =============================================================================
# Standard synthetic evaluation configuration
# =============================================================================

SYNTHETIC_CONFIG = {
    # Data-generating process. These are the standard settings previously
    # defined by SyntheticConfig in synthetic.py.
    "n_train": 1000,
    "n_test": 1000,
    "d": 1200,
    "kappa_train": 0.9,
    "kappa_test": 0.5,
    "gamma": 5.0,
    "sigma_y": 0.1,
    "sigma_a": 0.1,
    "sigma_epsilon": 20.0,
    "seed": 1,
    "exact_train_label_balance": True,
    "exact_test_label_balance": True,
    # Evaluation settings, matching the structure and defaults of evaluate.py.
    "transforms": ["identity", "standardize", "whiten"],
    # "empirical", "ledoit-wolf", or "ledoit-wolf-nonlinear"
    "whitening_estimator": "empirical",
    "whitening_relative_tolerance": None,
    "ridge_lambda": 0.001,
    "class_balanced": False,
    "output": None,  # optional JSON path
}


# =============================================================================
# Fitting and evaluation
# =============================================================================

def _synthetic_data_config(config: dict[str, Any]) -> SyntheticConfig:
    """Build the validated data configuration from the editable dictionary."""
    return SyntheticConfig(
        n_train=int(config["n_train"]),
        n_test=int(config["n_test"]),
        d=int(config["d"]),
        kappa_train=float(config["kappa_train"]),
        kappa_test=float(config["kappa_test"]),
        gamma=float(config["gamma"]),
        sigma_y=float(config["sigma_y"]),
        sigma_a=float(config["sigma_a"]),
        sigma_epsilon=float(config["sigma_epsilon"]),
        seed=int(config["seed"]),
        class_balanced=bool(config["class_balanced"]),
        exact_train_label_balance=bool(
            config["exact_train_label_balance"]
        ),
        exact_test_label_balance=bool(
            config["exact_test_label_balance"]
        ),
    )


def evaluate_synthetic(
    data_config: SyntheticConfig,
    *,
    transforms: Iterable[str],
    ridge_lambda: float,
    whitening_estimator: str = "empirical",
    whitening_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Fit every requested method on synthetic train and report train/test."""
    transform_names = tuple(transforms)
    if not transform_names:
        raise ValueError("At least one transform must be requested.")

    train, test = generate_train_test(data_config)
    split_data = {"train": train, "test": test}
    logistic_config = LogisticConfig(
        ridge_lambda=float(ridge_lambda),
        class_balanced=data_config.class_balanced,
    )

    evaluations: list[dict[str, Any]] = []
    canonical_names: set[str] = set()
    for requested_name in transform_names:
        # Preprocessing observes train only. The fitted object is then frozen
        # before either split is evaluated.
        transform = make_transform(
            requested_name,
            dtype=SYNTHETIC_DTYPE,
            whitening_estimator=whitening_estimator,
            whitening_relative_tolerance=whitening_relative_tolerance,
        ).fit(train.X)
        transform_report = transform.diagnostics()
        method = str(transform_report["name"])
        if method in canonical_names:
            raise ValueError(
                f"Transform {requested_name!r} duplicates method {method!r}."
            )
        canonical_names.add(method)

        X_train = transform.transform(train.X)
        X_test = transform.transform(test.X)
        fit = fit_logistic(X_train, train.y, logistic_config)
        if not fit.converged:
            messages = "; ".join(fit.convergence_messages)
            raise RuntimeError(f"{method} did not converge: {messages}")

        transformed = {"train": X_train, "test": X_test}
        split_reports = {
            split_name: evaluation_metrics(
                fit.estimator,
                transformed[split_name],
                split_data[split_name].y,
                split_data[split_name].groups,
                dtype=SYNTHETIC_DTYPE,
            )
            for split_name in SYNTHETIC_SPLITS
        }

        # Map the fitted head back to the original synthetic coordinates. This
        # makes the core, spurious, and noise coefficient sizes comparable
        # across identity, standardization, and whitening.
        coef_original, intercept_original = transform.inverse_head(
            fit.estimator.coef_,
            fit.estimator.intercept_,
        )
        flat_coef = np.asarray(coef_original, dtype=np.float64).reshape(
            -1, data_config.d
        )[0]
        evaluations.append(
            {
                "method": method,
                "requested_transform": requested_name,
                "transform": transform_report,
                "fit": {
                    "fit_split": "train",
                    "observations": data_config.n_train,
                    "uses_group_labels": False,
                    **fit.diagnostics(data_dtype=SYNTHETIC_DTYPE, metric_dtype=SYNTHETIC_DTYPE),
                },
                "splits": split_reports,
                "original_coordinate_coefficients": {
                    "core": float(flat_coef[0]),
                    "spurious": float(flat_coef[1]),
                    "noise_norm": float(np.linalg.norm(flat_coef[2:])),
                    "intercept": np.asarray(
                        intercept_original, dtype=np.float64
                    ).tolist(),
                },
            }
        )

    return {
        "schema_version": 1,
        "kind": "synthetic_evaluation",
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "data": asdict(data_config),
        "head": {
            "ridge_lambda": logistic_config.ridge_lambda,
            "C": logistic_config.C,
            "sklearn_C": logistic_config.sklearn_C(data_config.n_train),
            "regularization_convention": "average_log_loss",
            "class_balanced": logistic_config.class_balanced,
            "class_balance_method": (
                "inverse_class_frequency_sample_weight"
                if logistic_config.class_balanced
                else "none"
            ),
        },
        "protocol": {
            "fit_split": "train",
            "evaluation_splits": list(SYNTHETIC_SPLITS),
            "group_labels_used_for_fitting": False,
            "exact_train_label_balance": (
                data_config.exact_train_label_balance
            ),
            "exact_test_label_balance": (
                data_config.exact_test_label_balance
            ),
        },
        "evaluations": evaluations,
    }


# =============================================================================
# Terminal reporting
# =============================================================================

def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> list[str]:
    """Align a small plain-text table without another dependency."""
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    return [
        render(headers),
        render(tuple("-" * width for width in widths)),
        *(render(row) for row in rows),
    ]


def format_results(report: dict[str, Any]) -> str:
    """Render train/test metrics and original-coordinate coefficients."""
    data = report["data"]
    head = report["head"]
    lines = [
        (
            f"dataset=synthetic  seed={data['seed']}  "
            f"n_train={data['n_train']}  n_test={data['n_test']}  d={data['d']}"
        ),
        (
            f"kappa_train={data['kappa_train']:.4g}  "
            f"kappa_test={data['kappa_test']:.4g}  gamma={data['gamma']:.4g}"
        ),
        (
            f"ridge_lambda={head['ridge_lambda']:.8g}  C={head['C']:.8g}  "
            f"(sklearn_C={head['sklearn_C']:.8g})  "
            f"class_balance={head['class_balance_method']}"
        ),
        "",
    ]

    for split_name in SYNTHETIC_SPLITS:
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
        lines.extend(
            (
                split_name.upper(),
                *_render_table(headers, rows),
                "",
            )
        )

    coefficient_headers = (
        "method",
        "core",
        "spurious",
        "noise-norm",
        "intercept",
    )
    coefficient_rows = []
    for evaluation in report["evaluations"]:
        coefficient = evaluation["original_coordinate_coefficients"]
        intercept = np.asarray(coefficient["intercept"]).reshape(-1)
        coefficient_rows.append(
            (
                evaluation["method"],
                f"{coefficient['core']:.6g}",
                f"{coefficient['spurious']:.6g}",
                f"{coefficient['noise_norm']:.6g}",
                ",".join(f"{value:.6g}" for value in intercept),
            )
        )
    lines.extend(
        (
            "ORIGINAL-COORDINATE COEFFICIENTS",
            *_render_table(coefficient_headers, coefficient_rows),
            "",
            "Accuracy columns are fractions; g=* columns are per-group accuracy.",
            "LL means log loss; lower is better.",
        )
    )
    return "\n".join(lines)


def save_results(report: dict[str, Any], output: str | Path) -> Path:
    """Write an optional JSON result without overwriting an existing file."""
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing synthetic evaluation: {path}"
        )
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
            "Generate standard synthetic train/test data and evaluate fixed "
            "logistic heads."
        )
    )
    parser.add_argument("--n-train", dest="n_train", type=int)
    parser.add_argument("--n-test", dest="n_test", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument(
        "--kappa-train",
        dest="kappa_train",
        type=float,
    )
    parser.add_argument(
        "--kappa-test",
        dest="kappa_test",
        type=float,
    )
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--sigma-y", dest="sigma_y", type=float)
    parser.add_argument("--sigma-a", dest="sigma_a", type=float)
    parser.add_argument(
        "--sigma-epsilon",
        dest="sigma_epsilon",
        type=float,
    )
    parser.add_argument("--seed", type=int)
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
        help="Fit with mean-one inverse-class-frequency sample weights.",
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Apply only explicitly supplied command-line overrides."""
    config = {
        **SYNTHETIC_CONFIG,
        "transforms": list(SYNTHETIC_CONFIG["transforms"]),
    }
    for key in SYNTHETIC_CONFIG:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = resolved_config(parse_args(argv))
    data_config = _synthetic_data_config(config)
    report = evaluate_synthetic(
        data_config,
        transforms=config["transforms"],
        ridge_lambda=float(config["ridge_lambda"]),
        whitening_estimator=config["whitening_estimator"],
        whitening_relative_tolerance=config[
            "whitening_relative_tolerance"
        ],
    )
    print(format_results(report))
    if config["output"] is not None:
        print(f"\nSaved: {save_results(report, config['output'])}")


if __name__ == "__main__":
    main()
