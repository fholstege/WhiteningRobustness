"""Fixed scikit-learn logistic-regression fitting.

Class-balanced fits use mean-one inverse-class-frequency sample weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from numerics import (
    METRIC_DTYPE,
    MODEL_PARAMETER_DTYPE,
    OPTIMIZATION_DTYPE,
    PREPROCESSING_DTYPE,
    STORED_EMBEDDING_DTYPE,
)

BINARY_SOLVER = "liblinear"
MULTICLASS_SOLVER = "lbfgs"
MAX_ITER = 2000
TOLERANCE = 1e-6
# One initial fit and one retry after a convergence warning. L-BFGS can warm
# start the retry; liblinear does not support a useful warm restart.
MAX_LBFGS_ATTEMPTS = 2


class OffsetLogisticRegression(LogisticRegression):
    """Logistic regression with a fixed training-logit offset.

    This is deliberately a small, dense-array L-BFGS extension of sklearn's
    estimator API.  The learned ``coef_`` and ``intercept_`` parameterize the
    correction to the supplied offset.  Consequently, ordinary sklearn
    prediction methods work directly when ``offset=None``; callers using an
    offset must add the reference head to the fitted correction before using
    the estimator for prediction on new observations.

    As in sklearn's L-BFGS logistic regression, the intercept is not included
    in the L2 penalty.  ``C`` has exactly sklearn's weighted-loss convention:
    the L2 strength relative to average loss is ``1 / (C * sum_weight)``.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        offset: np.ndarray | None = None,
        initial_coef: np.ndarray | None = None,
        initial_intercept: np.ndarray | None = None,
    ) -> "OffsetLogisticRegression":
        if self.solver != "lbfgs":
            raise ValueError("OffsetLogisticRegression requires solver='lbfgs'.")
        if self.penalty not in ("l2", "deprecated") or self.l1_ratio not in (
            None,
            0,
            0.0,
        ):
            raise ValueError(
                "OffsetLogisticRegression supports only L2 regularization."
            )
        if self.dual:
            raise ValueError("OffsetLogisticRegression does not support dual=True.")
        if self.class_weight is not None:
            raise ValueError(
                "Pass explicit sample_weight instead of class_weight to "
                "OffsetLogisticRegression."
            )

        X = np.asarray(X, dtype=OPTIMIZATION_DTYPE, order="C")
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size:
            raise ValueError("X and y have incompatible shapes.")
        if y.size == 0 or not np.isfinite(X).all():
            raise ValueError("Offset logistic regression requires finite non-empty data.")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Logistic regression requires at least two classes.")
        n_samples, n_features = X.shape
        n_outputs = 1 if classes.size == 2 else classes.size

        if sample_weight is None:
            weights = np.ones(n_samples, dtype=OPTIMIZATION_DTYPE)
        else:
            weights = np.asarray(sample_weight, dtype=OPTIMIZATION_DTYPE)
            if weights.shape != (n_samples,):
                raise ValueError(
                    "sample_weight must have shape "
                    f"({n_samples},), got {weights.shape}."
                )
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("sample_weight must contain finite nonnegative values.")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("sample_weight must have positive sum.")

        expected_offset_shape = (n_samples,) if n_outputs == 1 else (
            n_samples,
            n_outputs,
        )
        if offset is None:
            fixed_offset = np.zeros(expected_offset_shape, dtype=OPTIMIZATION_DTYPE)
        else:
            fixed_offset = np.asarray(offset, dtype=OPTIMIZATION_DTYPE)
            if n_outputs == 1 and fixed_offset.shape == (n_samples, 1):
                fixed_offset = fixed_offset[:, 0]
            if fixed_offset.shape != expected_offset_shape:
                raise ValueError(
                    "offset must have shape "
                    f"{expected_offset_shape}, got {fixed_offset.shape}."
                )
            if not np.isfinite(fixed_offset).all():
                raise ValueError("offset must contain only finite values.")

        coef_shape = (n_outputs, n_features)
        intercept_shape = (n_outputs,)
        if initial_coef is None:
            if self.warm_start and hasattr(self, "coef_"):
                coef0 = np.asarray(self.coef_, dtype=OPTIMIZATION_DTYPE)
            else:
                coef0 = np.zeros(coef_shape, dtype=OPTIMIZATION_DTYPE)
        else:
            coef0 = np.asarray(initial_coef, dtype=OPTIMIZATION_DTYPE)
        if coef0.shape != coef_shape or not np.isfinite(coef0).all():
            raise ValueError(
                f"initial_coef must have finite shape {coef_shape}, got {coef0.shape}."
            )
        if initial_intercept is None:
            if self.fit_intercept and self.warm_start and hasattr(self, "intercept_"):
                intercept0 = np.asarray(self.intercept_, dtype=OPTIMIZATION_DTYPE)
            else:
                intercept0 = np.zeros(intercept_shape, dtype=OPTIMIZATION_DTYPE)
        else:
            intercept0 = np.asarray(initial_intercept, dtype=OPTIMIZATION_DTYPE)
        if intercept0.shape != intercept_shape or not np.isfinite(intercept0).all():
            raise ValueError(
                "initial_intercept must have finite shape "
                f"{intercept_shape}, got {intercept0.shape}."
            )
        if not self.fit_intercept and np.any(intercept0):
            raise ValueError("initial_intercept must be zero when fit_intercept=False.")

        l2_strength = 1.0 / (float(self.C) * weight_sum)
        include_intercept = int(bool(self.fit_intercept))
        n_dof = n_features + include_intercept
        encoded = np.searchsorted(classes, y)

        if n_outputs == 1:
            target = (encoded == 1).astype(OPTIMIZATION_DTYPE)
            start = np.concatenate(
                (coef0[0], intercept0 if include_intercept else np.empty(0))
            )

            def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
                coef = parameters[:n_features]
                intercept = parameters[-1] if include_intercept else 0.0
                scores = X @ coef + intercept + fixed_offset
                point_loss = np.logaddexp(0.0, scores) - target * scores
                residual = weights * (expit(scores) - target) / weight_sum
                loss = float(weights @ point_loss / weight_sum)
                loss += 0.5 * l2_strength * float(coef @ coef)
                gradient_coef = X.T @ residual + l2_strength * coef
                gradient = (
                    np.concatenate((gradient_coef, [float(residual.sum())]))
                    if include_intercept
                    else gradient_coef
                )
                return loss, gradient

        else:
            start_matrix = (
                np.column_stack((coef0, intercept0))
                if include_intercept
                else coef0
            )
            start = np.asarray(start_matrix, order="F").ravel(order="F")

            def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
                matrix = parameters.reshape((n_outputs, n_dof), order="F")
                coef = matrix[:, :n_features]
                intercept = matrix[:, -1] if include_intercept else 0.0
                scores = X @ coef.T + intercept + fixed_offset
                log_normalizer = logsumexp(scores, axis=1)
                point_loss = log_normalizer - scores[np.arange(n_samples), encoded]
                probabilities = np.exp(scores - log_normalizer[:, None])
                probabilities[np.arange(n_samples), encoded] -= 1.0
                residual = probabilities * (weights[:, None] / weight_sum)
                loss = float(weights @ point_loss / weight_sum)
                loss += 0.5 * l2_strength * float(np.square(coef).sum())
                gradient_coef = residual.T @ X + l2_strength * coef
                gradient_matrix = (
                    np.column_stack((gradient_coef, residual.sum(axis=0)))
                    if include_intercept
                    else gradient_coef
                )
                return loss, np.asarray(gradient_matrix, order="F").ravel(order="F")

        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(self.max_iter),
                "maxls": 50,
                "gtol": float(self.tol),
                "ftol": 64 * np.finfo(float).eps,
            },
        )
        fitted = (
            result.x.reshape((n_outputs, n_dof), order="F")
            if n_outputs > 1
            else result.x[None, :]
        )
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.optimization_initial_coef_ = coef0.copy()
        self.optimization_initial_intercept_ = intercept0.copy()
        self.coef_ = np.asarray(fitted[:, :n_features], dtype=MODEL_PARAMETER_DTYPE)
        self.intercept_ = np.asarray(
            fitted[:, -1] if include_intercept else np.zeros(n_outputs),
            dtype=MODEL_PARAMETER_DTYPE,
        )
        self.n_iter_ = np.asarray(
            [min(int(result.nit), int(self.max_iter))], dtype=np.int32
        )
        self.offset_used_ = offset is not None
        self.optimization_result_ = result
        if not result.success:
            warnings.warn(
                "lbfgs failed to converge after "
                f"{result.nit} iteration(s): {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self


class LogisticConvergenceError(RuntimeError):
    """Logistic regression failed after every permitted solver attempt."""

    def __init__(
        self,
        message: str,
        *,
        solver_attempts: Sequence[dict[str, object]],
        convergence_messages: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.solver_attempts = tuple(dict(value) for value in solver_attempts)
        self.convergence_messages = tuple(convergence_messages)


# =============================================================================
# Logistic configuration and fitted diagnostics
# =============================================================================
@dataclass(frozen=True)
class LogisticConfig:
    """The scientific head configuration.

    Optimizer details are fixed module constants. Only the ridge coefficient
    varies across primary experiments.
    """

    ridge_lambda: float
    class_balanced: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be finite and strictly positive.")

    @property
    def C(self) -> float:
        """Inverse average-loss ridge coefficient, useful for reporting."""
        return 1.0 / self.ridge_lambda

    def sklearn_C(self, n_samples: int) -> float:
        """Convert average-loss ridge to scikit-learn's C parameter.

        Scikit-learn uses a summed-log-loss convention. Therefore
        ``C = 1 / (n_samples * ridge_lambda)`` makes ``ridge_lambda`` the
        coefficient relative to average log loss.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        summed_loss_ridge = float(n_samples) * self.ridge_lambda
        optimization_max = float(np.finfo(OPTIMIZATION_DTYPE).max)
        sklearn_c = 1.0 / summed_loss_ridge
        if summed_loss_ridge > optimization_max:
            warnings.warn(
                "n_train * ridge_lambda exceeds the largest finite "
                f"{np.dtype(OPTIMIZATION_DTYPE).name} "
                f"value: {n_samples} * {self.ridge_lambda:g} = "
                f"{summed_loss_ridge:.8g} > {optimization_max:.8g}. This "
                "product cannot be represented in the optimization dtype; "
                "the resulting "
                f"sklearn_C={sklearn_c:.8g} may be numerically unsafe in "
                f"{np.dtype(OPTIMIZATION_DTYPE).name}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return sklearn_c


@dataclass
class LogisticFit:
    estimator: LogisticRegression
    converged: bool
    convergence_messages: tuple[str, ...]
    class_priors: np.ndarray
    sklearn_C: float
    ridge_lambda: float
    solver_attempts: tuple[dict[str, object], ...]
    class_balance_method: str = "none"
    sample_weight_summary: dict[str, float] | None = None

    @property
    def n_iter(self) -> list[int]:
        return np.asarray(self.estimator.n_iter_, dtype=int).tolist()

    def diagnostics(
        self, *, data_dtype=PREPROCESSING_DTYPE, metric_dtype=METRIC_DTYPE
    ) -> dict[str, object]:
        return {
            "solver": self.estimator.solver,
            "primary_solver": self.estimator.solver,
            "warm_start": bool(self.estimator.warm_start),
            "solver_attempts": [dict(attempt) for attempt in self.solver_attempts],
            "max_iter": MAX_ITER,
            "tol": TOLERANCE,
            "converged": self.converged,
            "convergence_messages": list(self.convergence_messages),
            "n_iter": self.n_iter,
            "class_balance_method": self.class_balance_method,
            "sample_weighting": (
                (
                    "inverse_class_frequency_mean_one"
                    if self.class_balance_method
                    == "inverse_class_frequency_sample_weight"
                    else self.class_balance_method
                )
                if self.sample_weight_summary is not None
                else "sklearn_default_uniform"
            ),
            "sample_weight_summary": self.sample_weight_summary,
            "regularization_convention": "average_log_loss",
            "ridge_lambda": self.ridge_lambda,
            "C": 1.0 / self.ridge_lambda,
            "sklearn_C": self.sklearn_C,
            "class_priors": self.class_priors.tolist(),
            "numerical_dtype": np.dtype(data_dtype).name,
            "stored_embedding_dtype": np.dtype(data_dtype).name,
            "preprocessing_dtype": np.dtype(data_dtype).name,
            "optimization_dtype": np.dtype(OPTIMIZATION_DTYPE).name,
            "parameter_dtype": np.dtype(MODEL_PARAMETER_DTYPE).name,
            "metric_dtype": np.dtype(metric_dtype).name,
            "fixed_logit_offset": bool(
                getattr(self.estimator, "offset_used_", False)
            ),
            "initialization": getattr(
                self.estimator, "initialization_source_", "zero_or_solver_default"
            ),
            "proximity_penalty": bool(
                getattr(self.estimator, "proximity_penalty_", False)
            ),
        }


def _class_priors(
    y: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Calculate empirical training priors in estimator class order."""
    counts = np.asarray(
        [(y == label).sum() for label in classes],
        dtype=MODEL_PARAMETER_DTYPE,
    )
    if np.any(counts == 0):
        raise ValueError("Every fitted class must have positive training mass.")
    return counts / counts.sum()


def _class_balanced_sample_weights(
    y: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Give every class equal total weight while preserving mean weight one."""
    counts = np.asarray([(y == label).sum() for label in classes], dtype=float)
    if np.any(counts == 0):
        raise ValueError("Every fitted class must have positive training mass.")
    per_class = y.size / (classes.size * counts)
    class_to_weight = {
        label: per_class[index] for index, label in enumerate(classes)
    }
    weights = np.asarray(
        [class_to_weight[label] for label in y], dtype=OPTIMIZATION_DTYPE
    )
    if not np.isclose(weights.mean(), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Class-balanced sample weights must have mean one.")
    return weights


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    config: LogisticConfig,
    *,
    callbacks: Sequence[Any] = (),
    sample_weight: np.ndarray | None = None,
    sample_weight_method: str | None = None,
    fixed_offset: np.ndarray | None = None,
    initial_coef: np.ndarray | None = None,
    initial_intercept: np.ndarray | None = None,
    force_lbfgs: bool = False,
) -> LogisticFit:
    """Fit the fixed logistic estimator and retain convergence diagnostics."""
    X = np.asarray(X, dtype=OPTIMIZATION_DTYPE, order="C")
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y have incompatible shapes.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if np.unique(y).size < 2:
        raise ValueError("Logistic regression requires at least two classes.")

    classes = np.unique(y)
    custom_lbfgs = bool(
        force_lbfgs
        or fixed_offset is not None
        or initial_coef is not None
        or initial_intercept is not None
    )
    solver = (
        MULTICLASS_SOLVER
        if custom_lbfgs
        else (
            BINARY_SOLVER if classes.size == 2 else MULTICLASS_SOLVER
        )
    )
    if sample_weight is not None and config.class_balanced:
        raise ValueError(
            "Explicit sample_weight cannot be combined with class_balanced=True."
        )
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=OPTIMIZATION_DTYPE)
        if sample_weight.shape != (y.size,):
            raise ValueError(
                "sample_weight must have shape (n_samples,), "
                f"expected ({y.size},) but got {sample_weight.shape}."
            )
        if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
            raise ValueError("sample_weight must contain finite nonnegative values.")
        mean_weight = float(sample_weight.mean())
        if mean_weight <= 0:
            raise ValueError("sample_weight must have positive mean.")
        sample_weight = sample_weight / mean_weight
        if not sample_weight_method:
            sample_weight_method = "explicit_mean_one_sample_weight"
    elif config.class_balanced:
        sample_weight = _class_balanced_sample_weights(y, classes)
        sample_weight_method = "inverse_class_frequency_sample_weight"
    elif sample_weight_method is not None:
        raise ValueError("sample_weight_method requires explicit sample_weight.")

    # Scale scikit-learn's C by n so ridge_lambda multiplies average loss.
    # Class-balancing weights have mean one, hence sum to n and preserve this
    # exact regularization convention.
    sklearn_C = config.sklearn_C(y.size)

    estimator_class = OffsetLogisticRegression if custom_lbfgs else LogisticRegression
    estimator = estimator_class(
        penalty="l2",
        C=sklearn_C,
        fit_intercept=True,
        solver=solver,
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=0,
        warm_start=solver != "liblinear",
    )
    if callbacks and custom_lbfgs:
        raise ValueError("Callbacks are not supported by OffsetLogisticRegression.")
    if callbacks:
        estimator.set_callbacks(*callbacks)

    solver_attempts: list[dict[str, object]] = []
    final_messages: tuple[str, ...] = ()
    for attempt_number in range(1, MAX_LBFGS_ATTEMPTS + 1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            special_fit_kwargs: dict[str, Any] = {}
            if custom_lbfgs:
                special_fit_kwargs["offset"] = fixed_offset
                if attempt_number == 1:
                    special_fit_kwargs["initial_coef"] = initial_coef
                    special_fit_kwargs["initial_intercept"] = initial_intercept
            if sample_weight is None:
                estimator.fit(X, y, **special_fit_kwargs)
            else:
                estimator.fit(
                    X,
                    y,
                    sample_weight=sample_weight,
                    **special_fit_kwargs,
                )
        final_messages = tuple(
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        )
        solver_attempts.append(
            {
                "attempt": attempt_number,
                "solver": solver,
                "warm_started": attempt_number > 1 and solver != "liblinear",
                "converged": not final_messages,
                "convergence_messages": list(final_messages),
                "n_iter": np.asarray(estimator.n_iter_, dtype=int).tolist(),
            }
        )
        if not final_messages:
            break
    else:
        retry_description = (
            "one warm restart" if solver != "liblinear" else "one retry"
        )
        message = (
            f"Logistic regression did not converge with solver={solver!r} after "
            f"{MAX_LBFGS_ATTEMPTS} attempts ({retry_description}): "
            + "; ".join(final_messages)
        )
        raise LogisticConvergenceError(
            message,
            solver_attempts=solver_attempts,
            convergence_messages=final_messages,
        )

    estimator.coef_ = np.asarray(
        estimator.coef_, dtype=MODEL_PARAMETER_DTYPE
    )
    estimator.intercept_ = np.asarray(
        estimator.intercept_, dtype=MODEL_PARAMETER_DTYPE
    )
    priors = _class_priors(y, np.asarray(estimator.classes_))
    sample_weight_summary = (
        {
            "minimum": float(sample_weight.min()),
            "maximum": float(sample_weight.max()),
            "mean": float(sample_weight.mean()),
            "sum": float(sample_weight.sum()),
        }
        if sample_weight is not None
        else None
    )
    return LogisticFit(
        estimator=estimator,
        converged=True,
        convergence_messages=(),
        class_priors=priors,
        sklearn_C=sklearn_C,
        ridge_lambda=config.ridge_lambda,
        solver_attempts=tuple(solver_attempts),
        class_balance_method=(
            str(sample_weight_method)
            if sample_weight is not None
            else "none"
        ),
        sample_weight_summary=sample_weight_summary,
    )


def fit_logistic_from_reference(
    X: np.ndarray,
    y: np.ndarray,
    config: LogisticConfig,
    reference_head: Any,
    *,
    proximity_penalty: bool,
    sample_weight: np.ndarray | None = None,
    sample_weight_method: str | None = None,
) -> LogisticFit:
    """Fit with L-BFGS from, or L2-regularized toward, a reference head.

    With ``proximity_penalty=False``, this is ordinary zero-centered ridge
    logistic regression initialized at the supplied head.  With
    ``proximity_penalty=True``, L-BFGS optimizes a zero-initialized correction
    using the reference logits as a fixed offset; adding the correction back
    to the reference parameters gives the exact ERM-centered ridge objective.
    """
    X = np.asarray(X, dtype=OPTIMIZATION_DTYPE, order="C")
    y = np.asarray(y).reshape(-1)
    if not hasattr(reference_head, "classes_"):
        raise ValueError("reference_head must be a fitted classifier.")
    classes = np.asarray(reference_head.classes_)
    if not np.array_equal(classes, np.unique(y)):
        raise ValueError("reference_head classes differ from fitting labels.")
    reference_coef = np.asarray(
        reference_head.coef_, dtype=OPTIMIZATION_DTYPE
    )
    reference_intercept = np.asarray(
        reference_head.intercept_, dtype=OPTIMIZATION_DTYPE
    )
    expected_outputs = 1 if classes.size == 2 else classes.size
    if reference_coef.shape != (expected_outputs, X.shape[1]):
        raise ValueError("reference_head coefficient dimension differs from X.")
    if reference_intercept.shape != (expected_outputs,):
        raise ValueError("reference_head intercept shape is invalid.")

    natural_solver = (
        BINARY_SOLVER if classes.size == 2 else MULTICLASS_SOLVER
    )
    if not proximity_penalty and natural_solver == "liblinear":
        warnings.warn(
            "ERM warm start was requested but is ignored because "
            "solver='liblinear' does not support warm-start initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fit_logistic(
            X,
            y,
            config,
            sample_weight=sample_weight,
            sample_weight_method=sample_weight_method,
        )

    if proximity_penalty:
        fixed_offset = X @ reference_coef.T + reference_intercept
        if expected_outputs == 1:
            fixed_offset = fixed_offset[:, 0]
        initial_coef = np.zeros_like(reference_coef)
        initial_intercept = np.zeros_like(reference_intercept)
    else:
        fixed_offset = None
        initial_coef = reference_coef
        initial_intercept = reference_intercept

    fit = fit_logistic(
        X,
        y,
        config,
        sample_weight=sample_weight,
        sample_weight_method=sample_weight_method,
        fixed_offset=fixed_offset,
        initial_coef=initial_coef,
        initial_intercept=initial_intercept,
        force_lbfgs=True,
    )
    fit.estimator.initialization_source_ = "original_erm_head"
    fit.estimator.proximity_penalty_ = bool(proximity_penalty)
    fit.estimator.reference_coef_ = reference_coef.copy()
    fit.estimator.reference_intercept_ = reference_intercept.copy()
    if proximity_penalty:
        fit.estimator.correction_coef_ = np.asarray(fit.estimator.coef_).copy()
        fit.estimator.correction_intercept_ = np.asarray(
            fit.estimator.intercept_
        ).copy()
        fit.estimator.coef_ = np.asarray(
            fit.estimator.coef_ + reference_coef,
            dtype=MODEL_PARAMETER_DTYPE,
        )
        fit.estimator.intercept_ = np.asarray(
            fit.estimator.intercept_ + reference_intercept,
            dtype=MODEL_PARAMETER_DTYPE,
        )
    return fit


def reference_head_in_transform(
    reference_head: Any,
    transform: Any,
    *,
    feature_indices: np.ndarray | None = None,
) -> Any:
    """Express a raw-coordinate reference head in fitted transform coordinates.

    ``feature_indices`` applies a fixed raw-coordinate mask before the
    transform, as required by methods such as NeuroTune.
    """
    if not all(
        hasattr(reference_head, attribute)
        for attribute in ("classes_", "coef_", "intercept_")
    ):
        raise ValueError("reference_head must be a fitted linear classifier.")
    coef = np.asarray(reference_head.coef_, dtype=OPTIMIZATION_DTYPE)
    if feature_indices is not None:
        indices = np.asarray(feature_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("feature_indices must be one-dimensional integers.")
        if np.any(indices < 0) or np.any(indices >= coef.shape[1]):
            raise ValueError("feature_indices contains an out-of-range coordinate.")
        coef = coef[:, indices]
    intercept = np.asarray(
        reference_head.intercept_, dtype=OPTIMIZATION_DTYPE
    ).copy()
    if hasattr(transform, "basis_"):
        transformed_coef = (
            coef @ np.asarray(transform.basis_, dtype=OPTIMIZATION_DTYPE)
        ) * np.asarray(transform.scales_, dtype=OPTIMIZATION_DTYPE)[None, :]
        transformed_intercept = intercept + coef @ np.asarray(
            transform.mean_, dtype=OPTIMIZATION_DTYPE
        )
    elif hasattr(transform, "scale_"):
        transformed_coef = coef * np.asarray(
            transform.scale_, dtype=OPTIMIZATION_DTYPE
        )[None, :]
        transformed_intercept = intercept + coef @ np.asarray(
            transform.mean_, dtype=OPTIMIZATION_DTYPE
        )
    else:
        transformed_coef = coef.copy()
        transformed_intercept = intercept
    return SimpleNamespace(
        classes_=np.asarray(reference_head.classes_).copy(),
        coef_=transformed_coef,
        intercept_=transformed_intercept,
        n_features_in_=int(transformed_coef.shape[1]),
    )
