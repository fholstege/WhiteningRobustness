"""NumPy synthetic experiments matching the chapter's data-generating process.

This file is separate from real-data evaluation so changes to the synthetic
model cannot silently alter how frozen embedding bundles are evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from model import LogisticConfig, fit_logistic
from metric import evaluation_metrics
from numerics import SYNTHETIC_DTYPE
from whitening import make_transform


# =============================================================================
# Synthetic data configuration and containers
# =============================================================================

@dataclass(frozen=True)
class SyntheticConfig:
    n_train: int = 1000
    n_test: int = 1000
    d: int = 1200
    kappa_train: float = 0.9
    kappa_test: float = 0.5
    gamma: float = 5.0
    sigma_y: float = 0.1
    sigma_a: float = 0.1
    sigma_epsilon: float = 20.0
    seed: int = 1
    class_balanced: bool = False
    exact_train_label_balance: bool = True
    exact_test_label_balance: bool = True

    def __post_init__(self) -> None:
        if self.n_train <= 0 or self.n_test <= 0:
            raise ValueError("Synthetic sample sizes must be positive.")
        if self.d < 2:
            raise ValueError("Synthetic dimension must be at least two.")
        if not 0 <= self.kappa_train <= 1 or not 0 <= self.kappa_test <= 1:
            raise ValueError("Environment agreement probabilities must lie in [0, 1].")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if min(self.sigma_y, self.sigma_a, self.sigma_epsilon) < 0:
            raise ValueError("Synthetic standard deviations must be nonnegative.")
        if self.exact_train_label_balance and self.n_train % 2:
            raise ValueError(
                "Exact synthetic train-label balance requires even n_train."
            )
        if self.exact_test_label_balance and self.n_test % 2:
            raise ValueError(
                "Exact synthetic test-label balance requires even n_test."
            )


@dataclass(frozen=True)
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def generate_environment(
    *,
    n: int,
    d: int,
    kappa: float,
    gamma: float,
    sigma_y: float,
    sigma_a: float,
    sigma_epsilon: float,
    rng: np.random.Generator,
    exact_label_balance: bool = False,
) -> SyntheticData:
    """Draw one environment with a core and a spuriously correlated feature."""
    # y and a use {-1, +1} while features are generated. Labels/groups are
    # converted to zero-based integers only at the return boundary.
    if exact_label_balance:
        if n % 2:
            raise ValueError("Exact label balance requires an even sample size.")
        y_signed = np.concatenate(
            (np.full(n // 2, -1.0), np.full(n // 2, 1.0))
        )
        y_signed = rng.permutation(y_signed)
    else:
        y_signed = rng.choice(np.array([-1.0, 1.0]), size=n)
    # The environment parameter kappa controls only agreement between y and a.
    agrees = rng.random(n) < kappa
    a_signed = np.where(agrees, y_signed, -y_signed)
    core = rng.normal(y_signed, sigma_y)
    spurious = gamma * rng.normal(a_signed, sigma_a)
    # Remaining coordinates contain independent isotropic noise. Dividing by
    # sqrt(q) keeps their total expected energy comparable as d changes.
    q = d - 2
    if q:
        epsilon = rng.normal(
            0.0, sigma_epsilon / np.sqrt(q), size=(n, q)
        )
        X = np.column_stack((core, spurious, epsilon))
    else:
        X = np.column_stack((core, spurious))
    y = (y_signed > 0).astype(np.int64)
    attribute = (a_signed > 0).astype(np.int64)
    groups = (2 * y + attribute).astype(np.int64)
    return SyntheticData(
        X=X.astype(SYNTHETIC_DTYPE), y=y, groups=groups
    )


def generate_train_test(config: SyntheticConfig) -> tuple[SyntheticData, SyntheticData]:
    """Generate reproducible train and test environments from separate streams."""
    train_rng = np.random.default_rng(config.seed)
    test_rng = np.random.default_rng(config.seed + 1)
    # Only kappa and sample size differ across the two environments.
    common = {
        "d": config.d,
        "gamma": config.gamma,
        "sigma_y": config.sigma_y,
        "sigma_a": config.sigma_a,
        "sigma_epsilon": config.sigma_epsilon,
    }
    train = generate_environment(
        n=config.n_train,
        kappa=config.kappa_train,
        rng=train_rng,
        exact_label_balance=config.exact_train_label_balance,
        **common,
    )
    test = generate_environment(
        n=config.n_test,
        kappa=config.kappa_test,
        rng=test_rng,
        exact_label_balance=config.exact_test_label_balance,
        **common,
    )
    return train, test


def run_synthetic_path(
    config: SyntheticConfig,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
) -> dict[str, Any]:
    """Evaluate transforms and ridge values on one generated train/test pair."""
    train, test = generate_train_test(config)
    results = []
    for transform_name in transforms:
        # Fit preprocessing on synthetic train and reuse it on synthetic test.
        transform = make_transform(transform_name, dtype=SYNTHETIC_DTYPE).fit(train.X)
        X_train = transform.transform(train.X)
        X_test = transform.transform(test.X)
        for ridge_lambda in ridge_lambdas:
            fit = fit_logistic(
                X_train,
                train.y,
                LogisticConfig(
                    float(ridge_lambda), config.class_balanced
                ),
            )
            if not fit.converged:
                # Preserve failed settings in the result rather than silently
                # dropping them from a ridge path.
                results.append(
                    {
                        "transform": transform_name,
                        "ridge_lambda": float(ridge_lambda),
                        "C": float(1.0 / ridge_lambda),
                        **fit.diagnostics(data_dtype=SYNTHETIC_DTYPE, metric_dtype=SYNTHETIC_DTYPE),
                        "test": None,
                        "coefficients": None,
                    }
                )
                continue
            # Map coefficients back so core/spurious magnitudes are comparable
            # across identity, standardization, and whitening.
            coef, intercept = transform.inverse_head(
                fit.estimator.coef_, fit.estimator.intercept_
            )
            flat = coef.reshape(-1, coef.shape[-1])[0]
            results.append(
                {
                    "transform": transform_name,
                    "ridge_lambda": float(ridge_lambda),
                    "C": float(1.0 / ridge_lambda),
                    **fit.diagnostics(data_dtype=SYNTHETIC_DTYPE, metric_dtype=SYNTHETIC_DTYPE),
                    "transform_diagnostics": transform.diagnostics(),
                    "test": evaluation_metrics(
                        fit.estimator, X_test, test.y, test.groups, dtype=SYNTHETIC_DTYPE
                    ),
                    "coefficients": {
                        "core": float(flat[0]),
                        "spurious": float(flat[1]),
                        "noise_norm": float(np.linalg.norm(flat[2:])),
                        "intercept": np.asarray(intercept).tolist(),
                    },
                }
            )
    return {
        "schema_version": 1,
        "config": asdict(config),
        "results": results,
    }
