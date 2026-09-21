#!/usr/bin/env python3
"""Sweep whitening estimators over synthetic dimension/train-sample ratios.

Edit ``SYNTHETIC_SWEEP_CONFIG`` below and run::

    python synthetic_sweep.py --workers 4 --blas-threads 1

The output contains one tidy result row per condition, simulation, and method.
Every method within a row's condition sees exactly the same generated data.
Transforms and logistic heads are fitted on train only; groups are used only
for reporting metrics.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from metric import evaluation_metrics
from numerics import SYNTHETIC_DTYPE
from model import LogisticConfig, fit_logistic
from synthetic import SyntheticConfig, generate_train_test
from whitening import (
    make_transform,
    validate_whitening_estimator,
)


ROOT = Path(__file__).resolve().parent


# =============================================================================
# Editable sweep configuration
# =============================================================================

SYNTHETIC_SWEEP_CONFIG: dict[str, Any] = {
    # Accept one sample size or a list. For every sample size, set
    # d = round(d_n_ratio * n_train) at each configured ratio.
    "n_train": [1000],
    "d_n_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    "n_test": 1000,
    "n_sim": 100,
    "base_seed": 1,
    # Independent condition/simulation jobs can run in parallel worker threads.
    "workers": 1,
    "blas_threads": 1,
    # Use exact label balance in both environments; Theorem 2 requires it for
    # the training labels.
    "exact_train_label_balance": True,
    "exact_test_label_balance": True,
    # Synthetic data-generating process.
    "kappa_train": 0.9,
    "kappa_test": 0.5,
    "gamma": 5.0,
    # Editable appendix values. gamma above selects the original Figure 2.
    # All values share the same conditions and random seeds in one JSON.
    "gamma_values": [1.0, 2.5, 5.0],
    "sigma_y": 0.1,
    "sigma_a": 0.1,
    "sigma_epsilon": 20.0,
    # Matched preprocessing comparisons.
    "baselines": ["identity", "standardize"],
    "whitening_estimators": [
        'empirical',
        'ledoit-wolf-nonlinear'
    ],
    "whitening_relative_tolerance": None,
    # Fixed prediction-head settings. scikit-learn optimizes summed logistic
    # loss, so keeping its C fixed holds the L2 coefficient unchanged across
    # every configured n_train. The shared model interface receives the
    # equivalent average-loss ridge_lambda = 1 / (n_train * sklearn_C).
    "ridge_lambda": 0.000000001,
    "class_balanced": False,
    # Store compact first-two-coordinate samples for the boundary figure.
    "plot_sample_size": 1000,
    "plot_sample_simulation": 0,
    # With d = q + 2, this selects d = 2000 and q = 1998 at n = 1000.
    "boundary_d_n_ratio": 2.0,
    # One switch skips all theorem/proposition diagnostics and validation jobs.
    "run_theory_checks": False,
    # Optional fixed-q/n Theorem 2 convergence sweep over sample sizes.
    "theorem2_validation": {
        "enabled": True,
        "q_n_ratio": 1.2,
        "n_train": [1000, 2000, 5000, 10000],
        "n_sim": 100,
        "base_seed": 10001,
        "sigma_epsilon": 5.0,
        "whitening_relative_tolerance": None,
    },
    # Appendix Proposition 2: d<n, noiseless signal coordinates, and
    # coordinate-wise N(0, 1) nuisance noise under uncentered whitening.
    "proposition2_validation": {
        "enabled": True,
        "q_n_ratio": [0.8],
        "n_train": [1000, 2000, 5000, 10000],
        "n_sim": 100,
        "base_seed": 20001,
        "sigma_y": 0.0,
        "sigma_a": 0.0,
        "noise_coordinate_std": 1.0,
    },
    # Existing files are never overwritten.
    "output": "results/synthetic_gamma_sweep.json",
    "overwrite": False,
}


# =============================================================================
# Validation and evaluation
# =============================================================================

def _validated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of an editable sweep configuration."""
    normalized = dict(config)
    raw_sample_sizes = config["n_train"]
    sample_sizes = (
        [int(value) for value in raw_sample_sizes]
        if isinstance(raw_sample_sizes, (list, tuple))
        else [int(raw_sample_sizes)]
    )
    ratios = [float(value) for value in config["d_n_ratio"]]
    if (
        not sample_sizes
        or len(sample_sizes) != len(set(sample_sizes))
        or any(n_train <= 0 for n_train in sample_sizes)
    ):
        raise ValueError("n_train must contain distinct positive sample sizes.")
    if not ratios or any(not np.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("d_n_ratio must contain finite positive values.")
    if int(config["n_test"]) <= 0:
        raise ValueError("n_test must be positive.")
    if int(config["n_sim"]) <= 0:
        raise ValueError("n_sim must be positive.")
    workers = config.get("workers", 1)
    blas_threads = config.get("blas_threads", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if (
        isinstance(blas_threads, bool)
        or not isinstance(blas_threads, int)
        or blas_threads < 1
    ):
        raise ValueError("blas_threads must be a positive integer.")
    exact_train_label_balance = bool(
        config.get("exact_train_label_balance", False)
    )
    exact_test_label_balance = bool(
        config.get("exact_test_label_balance", False)
    )
    invalid_balanced_sizes = [
        n_train for n_train in sample_sizes if n_train % 2
    ]
    if exact_train_label_balance and invalid_balanced_sizes:
        raise ValueError(
            "exact_train_label_balance requires every n_train value to be "
            f"even; invalid values: {invalid_balanced_sizes}."
        )
    if exact_test_label_balance and int(config["n_test"]) % 2:
        raise ValueError(
            "exact_test_label_balance requires n_test to be even; "
            f"received {int(config['n_test'])}."
        )
    plot_sample_size = int(config.get("plot_sample_size", 500))
    plot_sample_simulation = int(config.get("plot_sample_simulation", 0))
    if plot_sample_size <= 0:
        raise ValueError("plot_sample_size must be positive.")
    if not 0 <= plot_sample_simulation < int(config["n_sim"]):
        raise ValueError(
            "plot_sample_simulation must index one of the requested simulations."
        )
    boundary_d_n_ratio = float(
        config.get("boundary_d_n_ratio", ratios[0])
    )
    if not np.isfinite(boundary_d_n_ratio) or boundary_d_n_ratio <= 0:
        raise ValueError("boundary_d_n_ratio must be finite and positive.")
    theorem2_d_n_ratio = float(
        config.get("theorem2_d_n_ratio", max(ratios))
    )
    if not np.isfinite(theorem2_d_n_ratio) or theorem2_d_n_ratio <= 0:
        raise ValueError("theorem2_d_n_ratio must be finite and positive.")

    run_theory_checks = bool(config.get("run_theory_checks", True))
    theorem2_raw = config.get("theorem2_validation")
    if theorem2_raw is None:
        theorem2_validation = {"enabled": False}
    elif not isinstance(theorem2_raw, Mapping):
        raise TypeError("theorem2_validation must be a mapping.")
    else:
        theorem2_validation = dict(theorem2_raw)
        enabled = run_theory_checks and bool(
            theorem2_validation.get("enabled", True)
        )
        theorem2_validation["enabled"] = enabled
        if enabled:
            raw_sample_sizes = theorem2_validation["n_train"]
            if isinstance(raw_sample_sizes, (list, tuple)):
                theorem_sample_sizes = [int(value) for value in raw_sample_sizes]
            else:
                theorem_sample_sizes = [int(raw_sample_sizes)]
            raw_q_n_ratios = theorem2_validation["q_n_ratio"]
            if isinstance(raw_q_n_ratios, (list, tuple)):
                theorem_q_n_ratios = [float(value) for value in raw_q_n_ratios]
            else:
                theorem_q_n_ratios = [float(raw_q_n_ratios)]
            theorem_n_sim = int(theorem2_validation["n_sim"])
            theorem_sigma_epsilon = float(
                theorem2_validation["sigma_epsilon"]
            )
            if not theorem_sample_sizes:
                raise ValueError(
                    "theorem2_validation n_train must contain at least one "
                    "sample size."
                )
            if len(theorem_sample_sizes) != len(set(theorem_sample_sizes)):
                raise ValueError(
                    "theorem2_validation n_train contains duplicate sample sizes."
                )
            invalid_sample_sizes = [
                n for n in theorem_sample_sizes if n <= 0 or n % 2
            ]
            if invalid_sample_sizes:
                raise ValueError(
                    "Every theorem2_validation n_train value must be a positive "
                    f"even integer; invalid values: {invalid_sample_sizes}."
                )
            if not theorem_q_n_ratios or any(
                not np.isfinite(ratio) or ratio <= 1.0
                for ratio in theorem_q_n_ratios
            ):
                raise ValueError(
                    "theorem2_validation q_n_ratio must be finite and greater "
                    "than one."
                )
            if len(theorem_q_n_ratios) != 1:
                raise ValueError(
                    "theorem2_validation accepts one fixed q_n_ratio."
                )
            theorem_conditions = []
            for ratio_index, target_ratio in enumerate(theorem_q_n_ratios):
                for n_index, theorem_n in enumerate(theorem_sample_sizes):
                    q = int(round(target_ratio * theorem_n))
                    realized_q_n_ratio = q / theorem_n
                    dimension = q + 2
                    if q <= theorem_n:
                        raise ValueError(
                            "theorem2_validation q_n_ratio and n_train produce "
                            f"q/n={realized_q_n_ratio:g} for n={theorem_n}; "
                            "choose values whose rounded q satisfies q>n."
                        )
                    theorem_conditions.append(
                        {
                            "condition_id": (
                                f"theorem2_validation_ratio{ratio_index}_n{n_index}"
                            ),
                            "n_train": theorem_n,
                            "q": q,
                            "d": dimension,
                            "target_q_n_ratio": target_ratio,
                            "psi_q_over_n": realized_q_n_ratio,
                            "target_d_n_ratio": dimension / theorem_n,
                            "realized_d_n_ratio": dimension / theorem_n,
                            "regime": "theorem2_above_one",
                        }
                    )
            if theorem_n_sim <= 0:
                raise ValueError(
                    "theorem2_validation n_sim must be positive."
                )
            if (
                not np.isfinite(theorem_sigma_epsilon)
                or theorem_sigma_epsilon <= 0
            ):
                raise ValueError(
                    "theorem2_validation sigma_epsilon must be finite and positive."
                )
            if not np.isclose(float(config["sigma_y"]), float(config["sigma_a"])):
                raise ValueError(
                    "theorem2_validation requires sigma_y equal to sigma_a."
                )
            theorem2_validation.update(
                {
                    "q_n_ratio": theorem_q_n_ratios,
                    "n_train": theorem_sample_sizes,
                    "n_sim": theorem_n_sim,
                    "base_seed": int(theorem2_validation["base_seed"]),
                    "sigma_epsilon": theorem_sigma_epsilon,
                    "whitening_relative_tolerance": theorem2_validation.get(
                        "whitening_relative_tolerance"
                    ),
                    "conditions": theorem_conditions,
                }
            )

    proposition2_raw = config.get("proposition2_validation")
    if proposition2_raw is None:
        proposition2_validation = {"enabled": False}
    elif not isinstance(proposition2_raw, Mapping):
        raise TypeError("proposition2_validation must be a mapping.")
    else:
        proposition2_validation = dict(proposition2_raw)
        enabled = run_theory_checks and bool(
            proposition2_validation.get("enabled", True)
        )
        proposition2_validation["enabled"] = enabled
        if enabled:
            raw_sample_sizes = proposition2_validation["n_train"]
            proposition_sample_sizes = (
                [int(value) for value in raw_sample_sizes]
                if isinstance(raw_sample_sizes, (list, tuple))
                else [int(raw_sample_sizes)]
            )
            raw_ratios = proposition2_validation["q_n_ratio"]
            proposition_ratios = (
                [float(value) for value in raw_ratios]
                if isinstance(raw_ratios, (list, tuple))
                else [float(raw_ratios)]
            )
            if (
                not proposition_sample_sizes
                or len(proposition_sample_sizes) != len(set(proposition_sample_sizes))
                or any(n <= 0 or n % 2 for n in proposition_sample_sizes)
            ):
                raise ValueError(
                    "proposition2_validation n_train must contain distinct "
                    "positive even sample sizes."
                )
            if (
                not proposition_ratios
                or len(proposition_ratios) != len(set(proposition_ratios))
                or any(
                    not np.isfinite(ratio) or not 0.0 < ratio < 1.0
                    for ratio in proposition_ratios
                )
            ):
                raise ValueError(
                    "proposition2_validation q_n_ratio must contain distinct "
                    "finite ratios strictly between zero and one."
                )
            proposition_n_sim = int(proposition2_validation["n_sim"])
            noise_coordinate_std = float(
                proposition2_validation["noise_coordinate_std"]
            )
            if proposition_n_sim <= 0:
                raise ValueError("proposition2_validation n_sim must be positive.")
            if not np.isclose(noise_coordinate_std, 1.0):
                raise ValueError(
                    "proposition2_validation requires noise_coordinate_std=1."
                )
            if float(proposition2_validation["sigma_y"]) != 0.0 or float(
                proposition2_validation["sigma_a"]
            ) != 0.0:
                raise ValueError(
                    "proposition2_validation requires sigma_y=sigma_a=0."
                )
            proposition_conditions = []
            for ratio_index, target_ratio in enumerate(proposition_ratios):
                for n_index, proposition_n in enumerate(proposition_sample_sizes):
                    q = int(round(target_ratio * proposition_n))
                    d = q + 2
                    if q <= 0 or d >= proposition_n:
                        raise ValueError(
                            "proposition2_validation requires d=q+2<n, but "
                            f"q/n={q / proposition_n:g}, n={proposition_n}, d={d}."
                        )
                    proposition_conditions.append(
                        {
                            "condition_id": (
                                f"proposition2_validation_ratio{ratio_index}_n{n_index}"
                            ),
                            "n_train": proposition_n,
                            "q": q,
                            "d": d,
                            "target_q_n_ratio": target_ratio,
                            "psi_q_over_n": q / proposition_n,
                            "target_d_n_ratio": d / proposition_n,
                            "realized_d_n_ratio": d / proposition_n,
                            "regime": "proposition2_below_one",
                        }
                    )
            proposition2_validation.update(
                {
                    "q_n_ratio": proposition_ratios,
                    "n_train": proposition_sample_sizes,
                    "n_sim": proposition_n_sim,
                    "base_seed": int(proposition2_validation["base_seed"]),
                    "sigma_y": 0.0,
                    "sigma_a": 0.0,
                    "noise_coordinate_std": noise_coordinate_std,
                    "conditions": proposition_conditions,
                }
            )

    baselines = [str(value) for value in config["baselines"]]
    invalid_baselines = set(baselines) - {"identity", "standardize"}
    if invalid_baselines:
        raise ValueError(
            "baselines may contain only identity and standardize; got "
            f"{sorted(invalid_baselines)}."
        )
    estimators = [
        validate_whitening_estimator(str(value))
        for value in config["whitening_estimators"]
    ]
    if len(estimators) != len(set(estimators)):
        raise ValueError("whitening_estimators contains duplicate aliases.")
    if not estimators:
        raise ValueError("At least one whitening estimator is required.")
    conditions = []
    for n_train in sample_sizes:
        realized_dimensions: set[int] = set()
        for target_ratio in ratios:
            d = int(round(target_ratio * n_train))
            if d < 2:
                raise ValueError(
                    f"d/n={target_ratio:g} and n_train={n_train} produce d={d}; "
                    "the synthetic dimension must be at least two."
                )
            if d in realized_dimensions:
                raise ValueError(
                    "Two requested d/n ratios produce the same realized "
                    f"dimension d={d} at n_train={n_train}; remove one of them."
                )
            realized_dimensions.add(d)
            conditions.append(
                {
                    "condition_id": len(conditions),
                    "target_d_n_ratio": target_ratio,
                    "realized_d_n_ratio": d / n_train,
                    "n_train": n_train,
                    "d": d,
                }
            )

    ridge_lambda = config.get("ridge_lambda")
    if ridge_lambda is not None:
        ridge_lambda = float(ridge_lambda)
        if not np.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
            raise ValueError("ridge_lambda must be finite and strictly positive.")
        sklearn_c = None
    else:
        sklearn_c = float(config["sklearn_C"])
        if not np.isfinite(sklearn_c) or sklearn_c <= 0.0:
            raise ValueError("sklearn_C must be finite and strictly positive.")
        ridge_lambda = None

    normalized.update(
        {
            "n_train": sample_sizes,
            "d_n_ratio": ratios,
            "n_test": int(config["n_test"]),
            "n_sim": int(config["n_sim"]),
            "base_seed": int(config["base_seed"]),
            "workers": workers,
            "blas_threads": blas_threads,
            "exact_train_label_balance": exact_train_label_balance,
            "exact_test_label_balance": exact_test_label_balance,
            "baselines": baselines,
            "whitening_estimators": estimators,
            "ridge_lambda": ridge_lambda,
            "sklearn_C": sklearn_c,
            "class_balanced": bool(config["class_balanced"]),
            "plot_sample_size": plot_sample_size,
            "plot_sample_simulation": plot_sample_simulation,
            "boundary_d_n_ratio": boundary_d_n_ratio,
            "theorem2_d_n_ratio": theorem2_d_n_ratio,
            "run_theory_checks": run_theory_checks,
            "theorem2_validation": theorem2_validation,
            "proposition2_validation": proposition2_validation,
            "boundary_condition_id": min(
                conditions,
                key=lambda condition: abs(
                    condition["realized_d_n_ratio"] - boundary_d_n_ratio
                ),
            )["condition_id"],
            "conditions": conditions,
        }
    )
    theorem2_conditions = [
        condition
        for condition in conditions
        if (condition["d"] - 2) / condition["n_train"] > 1.0
    ]
    normalized["theorem2_condition_id"] = (
        min(
            theorem2_conditions,
            key=lambda condition: abs(
                condition["realized_d_n_ratio"] - theorem2_d_n_ratio
            ),
        )["condition_id"]
        if run_theory_checks and theorem2_conditions
        else None
    )
    if normalized["ridge_lambda"] is not None:
        for n_train in sample_sizes:
            LogisticConfig(
                ridge_lambda=normalized["ridge_lambda"],
                class_balanced=normalized["class_balanced"],
            )
    else:
        if (
            not np.isfinite(normalized["sklearn_C"])
            or normalized["sklearn_C"] <= 0.0
        ):
            raise ValueError("sklearn_C must be finite and strictly positive.")
        # Reuse the model's scientific validation for the equivalent
        # average-loss ridge at every configured sample size.
        for n_train in sample_sizes:
            LogisticConfig(
                ridge_lambda=1.0 / (n_train * normalized["sklearn_C"]),
                class_balanced=normalized["class_balanced"],
            )
    return normalized


def _method_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Construct nonduplicated baseline and whitening method specifications."""
    specs = [
        {
            "transform": baseline,
            "whitening_estimator": None,
        }
        for baseline in config["baselines"]
    ]
    specs.extend(
        {
            "transform": "whiten",
            "whitening_estimator": estimator,
        }
        for estimator in config["whitening_estimators"]
    )
    return specs


def _configured_value_count(value: Any) -> int:
    """Return the number of configured values, accepting a scalar."""
    return len(value) if isinstance(value, (list, tuple)) else 1


def _plot_split_payload(data: Any, max_points: int) -> dict[str, Any]:
    """Store a compact, deterministic sample for first-two-coordinate plots."""
    count = min(int(max_points), int(data.X.shape[0]))
    indices = np.arange(count, dtype=np.int64)
    return {
        "indices": indices.tolist(),
        "first_two_coordinates": np.asarray(
            data.X[indices, :2], dtype=np.float64
        ).tolist(),
        "labels": np.asarray(data.y[indices], dtype=np.int64).tolist(),
        "groups": np.asarray(data.groups[indices], dtype=np.int64).tolist(),
    }


def _theorem2_diagnostics(
    *,
    train: Any,
    transformed_train: np.ndarray,
    transform: Any,
    data_config: SyntheticConfig,
    method_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the empirical max-margin quantities appearing in Theorem 2."""
    is_empirical_whitening = (
        method_spec["transform"] == "whiten"
        and method_spec["whitening_estimator"] == "empirical"
    )
    if not is_empirical_whitening:
        return None

    n = int(data_config.n_train)
    q = int(data_config.d - 2)
    psi = q / n
    signed_labels = 2.0 * np.asarray(train.y, dtype=np.float64) - 1.0
    diagnostics = transform.diagnostics()
    prerequisites = {
        "equal_signal_noise_scales": bool(
            np.isclose(data_config.sigma_y, data_config.sigma_a)
        ),
        "exact_label_balance": bool(np.isclose(signed_labels.sum(), 0.0)),
        "q_at_least_n_minus_one": bool(q >= n - 1),
        "whitened_rank_n_minus_one": bool(
            diagnostics.get("retained_rank") == n - 1
        ),
    }
    applicable = all(prerequisites.values()) and psi > 1.0
    result: dict[str, Any] = {
        "applicable": applicable,
        "prerequisites": prerequisites,
        "psi_q_over_n": float(psi),
    }
    if not applicable:
        return result

    # With exact label balance and full centered row rank, the empirically
    # whitened Gram matrix is (n - 1) H. Hence this vector attains unit margin
    # for every training observation and is the hard-margin solution.
    whitened_coefficients = (
        transformed_train.T @ signed_labels / float(n - 1)
    )
    margins = signed_labels * (
        transformed_train @ whitened_coefficients
    )
    original, intercept = transform.inverse_head(
        whitened_coefficients[None, :],
        np.zeros(1, dtype=np.float64),
    )
    flat = np.asarray(original, dtype=np.float64).reshape(-1, data_config.d)[0]

    sigma_squared = float(data_config.sigma_y**2)
    rho_train = float(2.0 * data_config.kappa_train - 1.0)
    gamma_capital = 1.0 + sigma_squared
    denominator = gamma_capital**2 - rho_train**2
    tau_squared = (
        sigma_squared * (gamma_capital - rho_train**2) / denominator
    )
    sigma_signal = np.asarray(
        [
            [
                gamma_capital,
                data_config.gamma * rho_train,
            ],
            [
                data_config.gamma * rho_train,
                data_config.gamma**2 * gamma_capital,
            ],
        ],
        dtype=np.float64,
    )
    beta_signal = np.asarray(
        [1.0, data_config.gamma * rho_train], dtype=np.float64
    )
    alpha_psi = float(
        psi / (data_config.sigma_epsilon**2 * (psi - 1.0))
    )
    finite_n_delta = float(1.0 / (n * alpha_psi))
    finite_n_signal = np.linalg.solve(
        sigma_signal + finite_n_delta * np.eye(2),
        beta_signal,
    )
    finite_n_residual = float(
        1.0
        - 2.0 * finite_n_signal @ beta_signal
        + finite_n_signal @ sigma_signal @ finite_n_signal
    )
    finite_n_noise_squared_over_n = float(
        alpha_psi * finite_n_residual
    )
    finite_noise_contribution = float(
        data_config.sigma_epsilon**2
        / psi
        * np.dot(flat[2:], flat[2:])
        / n
    )
    result.update(
        {
            "finite_sample": {
                "core": float(flat[0]),
                "spurious": float(flat[1]),
                "noise_norm": float(np.linalg.norm(flat[2:])),
                "noise_squared_over_n": float(
                    np.dot(flat[2:], flat[2:]) / n
                ),
                "noise_prediction_variance": finite_noise_contribution,
                "intercept": np.asarray(
                    intercept, dtype=np.float64
                ).tolist(),
                "minimum_training_margin": float(margins.min()),
                "maximum_training_margin": float(margins.max()),
                "maximum_uniform_margin_error": float(
                    np.max(np.abs(margins - 1.0))
                ),
            },
            "limit": {
                "core": float(
                    (gamma_capital - rho_train**2) / denominator
                ),
                "spurious": float(
                    rho_train
                    * sigma_squared
                    / (data_config.gamma * denominator)
                ),
                "noise_squared_over_n": float(
                    psi
                    * tau_squared
                    / (data_config.sigma_epsilon**2 * (psi - 1.0))
                ),
                "noise_prediction_variance": float(
                    tau_squared / (psi - 1.0)
                ),
                "Gamma": float(gamma_capital),
                "rho_train": rho_train,
                "tau_squared": float(tau_squared),
            },
            "finite_n_approximation": {
                "core": float(finite_n_signal[0]),
                "spurious": float(finite_n_signal[1]),
                "noise_squared_over_n": finite_n_noise_squared_over_n,
                "noise_prediction_variance": float(
                    data_config.sigma_epsilon**2
                    / psi
                    * finite_n_noise_squared_over_n
                ),
                "delta_n": finite_n_delta,
                "alpha_psi": alpha_psi,
            },
        }
    )
    return result


def _appendix_below_one_diagnostics(
    *,
    train: Any,
    data_config: SyntheticConfig,
) -> dict[str, Any]:
    """Validate the appendix's exact core-only solution for q/n < 1."""
    n = int(data_config.n_train)
    q = int(data_config.d - 2)
    psi = q / n
    signed_labels = 2.0 * np.asarray(train.y, dtype=np.float64) - 1.0
    coefficients = np.zeros(data_config.d, dtype=np.float64)
    coefficients[0] = 1.0
    scores = np.asarray(train.X, dtype=np.float64) @ coefficients
    margins = signed_labels * scores
    maximum_margin_error = float(np.max(np.abs(margins - 1.0)))
    if maximum_margin_error > 1e-12:
        raise RuntimeError(
            "Appendix validation failed: the core-only classifier does not "
            "have uniform unit margins."
        )
    noise_variance = float(data_config.sigma_epsilon**2 / q)
    if not np.isclose(noise_variance, 1.0):
        raise RuntimeError(
            "Appendix validation requires unit-variance nuisance coordinates."
        )
    common_coefficients = {
        "core": 1.0,
        "spurious": 0.0,
        "noise_norm": 0.0,
        "noise_squared_over_n": 0.0,
        "noise_prediction_variance": 0.0,
    }
    return {
        "applicable": True,
        "regime": "appendix_below_one",
        "solution_source": "appendix_core_only_proposition",
        "psi_q_over_n": float(psi),
        "prerequisites": {
            "q_over_n_below_one": bool(psi < 1.0),
            "d_below_n": bool(data_config.d < n),
            "uncentered_second_moment_whitening": True,
            "noiseless_core_and_spurious_coordinates": bool(
                data_config.sigma_y == 0.0 and data_config.sigma_a == 0.0
            ),
            "unit_noise_coordinate_variance": True,
            "uniform_unit_margins": True,
        },
        "finite_sample": {
            **common_coefficients,
            "intercept": [0.0],
            "minimum_training_margin": float(margins.min()),
            "maximum_training_margin": float(margins.max()),
            "maximum_uniform_margin_error": maximum_margin_error,
            "uncentered_whitened_objective": float(
                0.5 * np.dot(scores, scores) / (n - 1)
            ),
        },
        "limit": dict(common_coefficients),
    }


def _evaluate_method(
    *,
    train: Any,
    test: Any,
    data_config: SyntheticConfig,
    logistic_config: LogisticConfig,
    method_spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit and evaluate one method on an already generated data pair."""
    estimator = method_spec["whitening_estimator"] or "empirical"
    transform = make_transform(
        str(method_spec["transform"]),
        dtype=SYNTHETIC_DTYPE,
        whitening_estimator=str(estimator),
        whitening_relative_tolerance=config[
            "whitening_relative_tolerance"
        ],
    ).fit(train.X)
    X_train = transform.transform(train.X)
    X_test = transform.transform(test.X)
    fit = fit_logistic(X_train, train.y, logistic_config)
    if not fit.converged:
        messages = "; ".join(fit.convergence_messages)
        method = transform.diagnostics()["name"]
        raise RuntimeError(
            f"{method} did not converge for seed={data_config.seed}, "
            f"n_train={data_config.n_train}, d={data_config.d}: {messages}"
        )

    coefficients, intercept = transform.inverse_head(
        fit.estimator.coef_,
        fit.estimator.intercept_,
    )
    flat = np.asarray(coefficients, dtype=np.float64).reshape(
        -1, data_config.d
    )[0]
    transform_diagnostics = transform.diagnostics()
    result = {
        "method": str(transform_diagnostics["name"]),
        "whitening_estimator": method_spec["whitening_estimator"],
        "transform": transform_diagnostics,
        "fit": fit.diagnostics(data_dtype=SYNTHETIC_DTYPE, metric_dtype=SYNTHETIC_DTYPE),
        "train": evaluation_metrics(
            fit.estimator, X_train, train.y, train.groups, dtype=SYNTHETIC_DTYPE
        ),
        "test": evaluation_metrics(
            fit.estimator, X_test, test.y, test.groups, dtype=SYNTHETIC_DTYPE
        ),
        "original_coordinate_coefficients": {
            "core": float(flat[0]),
            "spurious": float(flat[1]),
            "noise_norm": float(np.linalg.norm(flat[2:])),
            "noise_squared_over_n": float(np.dot(flat[2:], flat[2:]) / data_config.n_train),
            "intercept": np.asarray(intercept, dtype=np.float64).tolist(),
        },
    }
    if config["run_theory_checks"]:
        theorem2 = _theorem2_diagnostics(
            train=train,
            transformed_train=X_train,
            transform=transform,
            data_config=data_config,
            method_spec=method_spec,
        )
        if theorem2 is not None:
            result["theorem2"] = theorem2
    return result


def _run_sweep_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all methods for one matched condition/simulation dataset."""
    resolved = job["config"]
    condition = dict(job["condition"])
    simulation = int(job["simulation"])
    seed = int(resolved["base_seed"]) + simulation

    data_config = SyntheticConfig(
        n_train=int(condition["n_train"]),
        n_test=int(resolved["n_test"]),
        d=int(condition["d"]),
        kappa_train=float(resolved["kappa_train"]),
        kappa_test=float(resolved["kappa_test"]),
        gamma=float(resolved["gamma"]),
        sigma_y=float(resolved["sigma_y"]),
        sigma_a=float(resolved["sigma_a"]),
        sigma_epsilon=float(resolved["sigma_epsilon"]),
        seed=seed,
        class_balanced=bool(resolved["class_balanced"]),
        exact_train_label_balance=bool(
            resolved["exact_train_label_balance"]
        ),
        exact_test_label_balance=bool(
            resolved["exact_test_label_balance"]
        ),
    )
    train, test = generate_train_test(data_config)
    if resolved.get("ridge_lambda") is not None:
        ridge_lambda = float(resolved["ridge_lambda"])
    else:
        ridge_lambda = 1.0 / (
            data_config.n_train * float(resolved["sklearn_C"])
        )
    logistic_config = LogisticConfig(
        ridge_lambda=ridge_lambda,
        class_balanced=bool(resolved["class_balanced"]),
    )
    rows = []
    for method_spec in _method_specs(resolved):
        evaluation = _evaluate_method(
            train=train,
            test=test,
            data_config=data_config,
            logistic_config=logistic_config,
            method_spec=method_spec,
            config=resolved,
        )
        rows.append(
            {
                **condition,
                "simulation": simulation,
                "seed": seed,
                **evaluation,
            }
        )

    boundary_sample = None
    if simulation == int(resolved["plot_sample_simulation"]):
        boundary_sample = {
            **condition,
            "simulation": simulation,
            "seed": seed,
            "train": _plot_split_payload(
                train, int(resolved["plot_sample_size"])
            ),
            "test": _plot_split_payload(
                test, int(resolved["plot_sample_size"])
            ),
        }

    return {
        "condition": condition,
        "simulation": simulation,
        "rows": rows,
        "boundary_sample": boundary_sample,
    }


def _run_theorem2_validation_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run one dedicated empirical-whitening hard-margin diagnostic."""
    resolved = job["config"]
    theorem_config = resolved["theorem2_validation"]
    theorem_condition = dict(job["condition"])
    simulation = int(job["simulation"])
    seed = int(theorem_config["base_seed"]) + simulation
    n_train = int(theorem_condition["n_train"])
    d = int(theorem_condition["d"])
    condition = {
        **theorem_condition,
        "phase": "theorem2",
    }
    data_config = SyntheticConfig(
        n_train=n_train,
        n_test=2,
        d=d,
        kappa_train=float(resolved["kappa_train"]),
        kappa_test=float(resolved["kappa_test"]),
        gamma=float(resolved["gamma"]),
        sigma_y=float(resolved["sigma_y"]),
        sigma_a=float(resolved["sigma_a"]),
        sigma_epsilon=float(theorem_config["sigma_epsilon"]),
        seed=seed,
        class_balanced=False,
        exact_train_label_balance=True,
        exact_test_label_balance=True,
    )
    train, _ = generate_train_test(data_config)
    transform = make_transform(
        "whiten",
        dtype=SYNTHETIC_DTYPE,
        whitening_estimator="empirical",
        whitening_relative_tolerance=theorem_config[
            "whitening_relative_tolerance"
        ],
    ).fit(train.X)
    transformed_train = transform.transform(train.X)
    theorem = _theorem2_diagnostics(
        train=train,
        transformed_train=transformed_train,
        transform=transform,
        data_config=data_config,
        method_spec={
            "transform": "whiten",
            "whitening_estimator": "empirical",
        },
    )
    if theorem is None or theorem.get("applicable") is not True:
        raise RuntimeError(
            "Dedicated Theorem 2 configuration did not satisfy its prerequisites."
        )
    row = {
        **condition,
        "simulation": simulation,
        "seed": seed,
        "sigma_epsilon": float(data_config.sigma_epsilon),
        "theorem2": theorem,
    }
    return {
        "condition": condition,
        "simulation": simulation,
        "rows": [row],
        "boundary_sample": None,
    }


def _run_proposition2_validation_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Run one dedicated below-one appendix validation diagnostic."""
    resolved = job["config"]
    proposition_config = resolved["proposition2_validation"]
    proposition_condition = dict(job["condition"])
    simulation = int(job["simulation"])
    seed = int(proposition_config["base_seed"]) + simulation
    q = int(proposition_condition["q"])
    condition = {
        **proposition_condition,
        "phase": "proposition2",
    }
    data_config = SyntheticConfig(
        n_train=int(condition["n_train"]),
        n_test=2,
        d=int(condition["d"]),
        kappa_train=float(resolved["kappa_train"]),
        kappa_test=float(resolved["kappa_test"]),
        gamma=float(resolved["gamma"]),
        sigma_y=float(proposition_config["sigma_y"]),
        sigma_a=float(proposition_config["sigma_a"]),
        sigma_epsilon=float(
            proposition_config["noise_coordinate_std"] * np.sqrt(q)
        ),
        seed=seed,
        class_balanced=False,
        exact_train_label_balance=True,
        exact_test_label_balance=True,
    )
    train, _ = generate_train_test(data_config)
    proposition = _appendix_below_one_diagnostics(
        train=train,
        data_config=data_config,
    )
    row = {
        **condition,
        "simulation": simulation,
        "seed": seed,
        "sigma_epsilon": float(data_config.sigma_epsilon),
        "noise_coordinate_std": float(
            proposition_config["noise_coordinate_std"]
        ),
        "proposition2": proposition,
    }
    return {
        "condition": condition,
        "simulation": simulation,
        "rows": [row],
        "boundary_sample": None,
    }


def _execute_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    runner: Callable[[Mapping[str, Any]], dict[str, Any]],
    workers: int,
    blas_threads: int,
    progress: Callable[[int, int, Mapping[str, Any], int], None] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Run independent numerical jobs and restore configured output order."""
    total = len(jobs)
    if total == 0:
        return [], 0
    effective_workers = min(int(workers), total)
    outputs: list[dict[str, Any] | None] = [None] * total
    if int(blas_threads) <= 1:
        executor_context = nullcontext()
    else:
        executor_context = threadpool_limits(limits=int(blas_threads))

    with executor_context:
        if effective_workers == 1:
            for index, job in enumerate(jobs):
                output = runner(job)
                outputs[index] = output
                if progress is not None:
                    progress(
                        index + 1,
                        total,
                        output["condition"],
                        output["simulation"],
                    )
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_indices = {
                    executor.submit(runner, job): index
                    for index, job in enumerate(jobs)
                }
                completed = 0
                for future in as_completed(future_indices):
                    index = future_indices[future]
                    output = future.result()
                    outputs[index] = output
                    completed += 1
                    if progress is not None:
                        progress(
                            completed,
                            total,
                            output["condition"],
                            output["simulation"],
                        )
    return [output for output in outputs if output is not None], effective_workers


def run_sweep(
    config: Mapping[str, Any],
    *,
    progress: Callable[[int, int, Mapping[str, Any], int], None] | None = None,
) -> dict[str, Any]:
    """Run a matched Monte Carlo sweep and return a JSON-serializable report."""
    if "gamma_values" in config:
        gammas = [float(value) for value in config["gamma_values"]]
        if (not gammas or len(set(gammas)) != len(gammas)
                or any(not np.isfinite(value) or value <= 0 for value in gammas)):
            raise ValueError("gamma_values must contain distinct finite positive values.")
        if float(config["gamma"]) not in gammas:
            raise ValueError("gamma_values must include gamma, the original Figure 2 value.")
        if config.get("run_theory_checks", False):
            raise ValueError("The gamma sweep requires run_theory_checks=False.")
        reports = []
        for gamma in gammas:
            gamma_config = dict(config, gamma=gamma, run_theory_checks=False)
            gamma_config.pop("gamma_values")
            def gamma_progress(completed, total, condition, simulation):
                if progress is not None:
                    progress(completed, total, dict(condition, gamma=gamma), simulation)
            reports.append(run_sweep(gamma_config, progress=gamma_progress))
        report = next(r for r in reports if r["config"]["gamma"] == float(config["gamma"]))
        report["gamma_values"] = gammas
        report["gamma_sweep"] = [r for r in reports if r is not report]
        return report
    resolved = _validated_config(config)
    jobs = [
        {
            "condition": condition,
            "simulation": simulation,
            "config": resolved,
        }
        for condition in resolved["conditions"]
        for simulation in range(resolved["n_sim"])
    ]
    # The numerical work happens in compiled libraries that release the GIL.
    # Apply one global BLAS/OpenMP limit while worker threads run so each job
    # does not create another full numerical thread pool.
    completed_outputs, effective_workers = _execute_jobs(
        jobs,
        runner=_run_sweep_job,
        workers=int(resolved["workers"]),
        blas_threads=int(resolved["blas_threads"]),
        progress=progress,
    )

    # Futures complete nondeterministically. Flatten in configured job order so
    # serial and parallel reports have identical result and sample ordering.
    results = [
        row for output in completed_outputs for row in output["rows"]
    ]
    decision_boundary_samples = [
        output["boundary_sample"]
        for output in completed_outputs
        if output["boundary_sample"] is not None
    ]

    theorem_config = resolved["theorem2_validation"]
    theorem_outputs: list[dict[str, Any]] = []
    theorem_effective_workers = 0
    if theorem_config["enabled"]:
        theorem_jobs = [
            {
                "config": resolved,
                "condition": condition,
                "simulation": simulation,
            }
            for condition in theorem_config["conditions"]
            for simulation in range(int(theorem_config["n_sim"]))
        ]
        theorem_outputs, theorem_effective_workers = _execute_jobs(
            theorem_jobs,
            runner=_run_theorem2_validation_job,
            workers=int(resolved["workers"]),
            blas_threads=int(resolved["blas_threads"]),
            progress=progress,
        )
    theorem_rows = [
        row for output in theorem_outputs for row in output["rows"]
    ]

    proposition_config = resolved["proposition2_validation"]
    proposition_outputs: list[dict[str, Any]] = []
    proposition_effective_workers = 0
    if proposition_config["enabled"]:
        proposition_jobs = [
            {
                "config": resolved,
                "condition": condition,
                "simulation": simulation,
            }
            for condition in proposition_config["conditions"]
            for simulation in range(int(proposition_config["n_sim"]))
        ]
        proposition_outputs, proposition_effective_workers = _execute_jobs(
            proposition_jobs,
            runner=_run_proposition2_validation_job,
            workers=int(resolved["workers"]),
            blas_threads=int(resolved["blas_threads"]),
            progress=progress,
        )
    proposition_rows = [
        row for output in proposition_outputs for row in output["rows"]
    ]

    stored_config = {
        key: value
        for key, value in resolved.items()
        if key not in {"conditions", "output", "overwrite"}
    }
    return {
        "schema_version": 13,
        "kind": "synthetic_whitening_estimator_sweep",
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "config": stored_config,
        "conditions": resolved["conditions"],
        "protocol": {
            "numerical_dtype": np.dtype(SYNTHETIC_DTYPE).name,
            "fit_split": "train",
            "evaluation_splits": ["train", "test"],
            "group_labels_used_for_fitting": False,
            "matched_methods_within_simulation": True,
            "paired_seeds_across_conditions": True,
            "fixed_n_train_across_conditions": len(resolved["n_train"]) == 1,
            "paired_seeds_across_sample_sizes": True,
            "sklearn_C_fixed_across_sample_sizes": (
                resolved.get("ridge_lambda") is None
                and resolved.get("sklearn_C") is not None
            ),
            "regularization_convention": (
                "average_log_loss"
                if resolved.get("ridge_lambda") is not None
                else "fixed_summed_log_loss_C"
            ),
            "exact_train_label_balance": resolved[
                "exact_train_label_balance"
            ],
            "exact_test_label_balance": resolved[
                "exact_test_label_balance"
            ],
            "theory_checks_enabled": resolved["run_theory_checks"],
            "theorem2_diagnostics_use_hard_margin_solution": resolved[
                "run_theory_checks"
            ],
            "theorem2_validation_paired_seeds_across_conditions": True,
            "theorem2_validation_fixed_q_n_ratio": True,
            "proposition2_validation_paired_seeds_across_conditions": True,
            "proposition2_validation_regime": (
                "uncentered_noiseless_signal_unit_noise"
            ),
            "workers": resolved["workers"],
            "effective_workers": effective_workers,
            "blas_threads_per_worker": resolved["blas_threads"],
        },
        "plot_data": {
            "boundary_condition_id": resolved["boundary_condition_id"],
            "boundary_simulation": resolved["plot_sample_simulation"],
            "theorem2_condition_id": resolved["theorem2_condition_id"],
            "decision_boundary_samples": decision_boundary_samples,
        },
        "theorem2_validation": {
            "config": theorem_config,
            "effective_workers": theorem_effective_workers,
            "results": theorem_rows,
        },
        "proposition2_validation": {
            "config": proposition_config,
            "effective_workers": proposition_effective_workers,
            "results": proposition_rows,
        },
        "results": results,
    }


# =============================================================================
# Output and minimal command-line overrides
# =============================================================================

def save_results(
    report: Mapping[str, Any],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write JSON, replacing an existing result only when requested."""
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing synthetic sweep: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if overwrite:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(contents)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(contents)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep whitening estimators over synthetic d/n ratios. Edit "
            "SYNTHETIC_SWEEP_CONFIG for scientific settings."
        )
    )
    parser.add_argument("--n-sim", type=int)
    parser.add_argument("--n-train", type=int, nargs="+")
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--d-n-ratio", type=float, nargs="+")
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--gamma", type=float, help="Spurious feature scale only.")
    parser.add_argument("--gammas", type=float, nargs="+", help="Override editable gamma_values.")
    parser.add_argument(
        "--skip-theory-checks", action="store_true",
        help="Run only the main performance sweep, without theory validation.",
    )
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        help=(
            "Average-log-loss ridge penalty. This is the scientific "
            "regularization parameter used by the repository."
        ),
    )
    parser.add_argument(
        "--sklearn-C",
        type=float,
        help=(
            "Legacy compatibility alias for a fixed summed-log-loss C. "
            "When provided without --ridge-lambda, the sweep converts it to "
            "the equivalent ridge_lambda for each training sample size."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Parallel condition/simulation workers; 1 runs serially.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        help="BLAS/OpenMP threads available inside each worker.",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Atomically replace an existing synthetic sweep JSON.",
    )
    return parser.parse_args(argv)


def _print_progress(
    completed: int,
    total: int,
    condition: Mapping[str, Any],
    simulation: int,
) -> None:
    phase = condition.get("phase", "main")
    print(
        f"[{completed}/{total}] phase={phase} "
        f"condition={condition['condition_id']} "
        f"n={condition['n_train']} d={condition['d']} "
        f"gamma={condition.get('gamma', 'configured')} "
        f"simulation={simulation + 1}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = dict(SYNTHETIC_SWEEP_CONFIG)
    if args.n_sim is not None:
        config["n_sim"] = args.n_sim
    if args.n_train is not None:
        config["n_train"] = args.n_train
    if args.n_test is not None:
        config["n_test"] = args.n_test
    if args.d_n_ratio is not None:
        config["d_n_ratio"] = args.d_n_ratio
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if args.gamma is not None:
        config["gamma"] = args.gamma
        config["gamma_values"] = [args.gamma]
    if args.gammas is not None:
        config["gamma_values"] = args.gammas
    if args.skip_theory_checks:
        config["run_theory_checks"] = False
    if args.ridge_lambda is not None:
        config["ridge_lambda"] = args.ridge_lambda
    if args.sklearn_C is not None:
        config["sklearn_C"] = args.sklearn_C
    if args.workers is not None:
        config["workers"] = args.workers
    if args.blas_threads is not None:
        config["blas_threads"] = args.blas_threads
    if args.output is not None:
        config["output"] = args.output
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite
    output_path = Path(config["output"]).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if output_path.exists() and not config.get("overwrite", False):
        raise FileExistsError(
            f"Refusing to rerun over existing results: {output_path}. "
            "Choose --output or explicitly pass --overwrite."
        )
    if int(config.get("workers", 1)) > 1:
        job_count = (
            _configured_value_count(config["n_train"])
            * len(config["d_n_ratio"])
            * int(config["n_sim"])
            * len(config.get("gamma_values", [config["gamma"]]))
        )
        run_theory_checks = bool(config.get("run_theory_checks", True))
        theorem_config = config.get("theorem2_validation", {})
        theorem_job_count = (
            _configured_value_count(theorem_config["n_train"])
            * _configured_value_count(theorem_config["q_n_ratio"])
            * int(theorem_config.get("n_sim", 0))
            if run_theory_checks and theorem_config.get("enabled", False)
            else 0
        )
        proposition_config = config.get("proposition2_validation", {})
        proposition_job_count = (
            _configured_value_count(proposition_config["n_train"])
            * _configured_value_count(proposition_config["q_n_ratio"])
            * int(proposition_config.get("n_sim", 0))
            if run_theory_checks and proposition_config.get("enabled", False)
            else 0
        )
        print(
            f"Running {job_count} matched main jobs and "
            f"{theorem_job_count} theorem-validation jobs and "
            f"{proposition_job_count} proposition-validation jobs with up to "
            f"{config['workers']} workers and {config.get('blas_threads', 1)} "
            "BLAS thread(s) per worker.",
            flush=True,
        )
    report = run_sweep(config, progress=_print_progress)
    output = save_results(
        report,
        config["output"],
        overwrite=bool(config.get("overwrite", False)),
    )
    print(
        f"Saved {sum(len(r['results']) for r in [report, *report.get('gamma_sweep', [])])} main result rows and "
        f"{len(report['theorem2_validation']['results'])} theorem rows and "
        f"{len(report['proposition2_validation']['results'])} proposition rows "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
