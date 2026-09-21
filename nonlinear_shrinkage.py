"""Analytical nonlinear shrinkage of observed covariance eigenvalues.

This module implements the eigenvalue formulas of Ledoit and Wolf (2020),
"Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices".
It follows the authors' reference Matlab implementation and the small
MIT-licensed ``nonlinshrink`` Python translation, but deliberately returns
eigenvalues rather than materializing a covariance matrix.

For ``dimension > effective_sample_size``, the published estimator also
assigns one common positive eigenvalue to every sample-null direction.  The
calling whitening code records that reference value but does not use it:
this repository whitens only the raw empirically observed covariance support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numerics import PREPROCESSING_DTYPE, SPECTRAL_ESTIMATION_DTYPE


_SQRT_FIVE = float(np.sqrt(5.0))
_DENSITY_CONSTANT = float(3.0 / (4.0 * _SQRT_FIVE))
_HILBERT_LINEAR_CONSTANT = float(-3.0 / (10.0 * np.pi))
_HILBERT_LOG_CONSTANT = float(3.0 / (4.0 * _SQRT_FIVE * np.pi))
DEFAULT_HILBERT_RELATIVE_ERROR = 1e-12
_MINIMUM_ASYMPTOTIC_ARGUMENT = 10.0
_ASYMPTOTIC_TERM_COUNT = 10
_ASYMPTOTIC_COEFFICIENTS = tuple(
    3.0 * 5.0**power / ((2 * power + 1) * (2 * power + 3))
    for power in range(_ASYMPTOTIC_TERM_COUNT)
)


def validate_hilbert_relative_error(value: float) -> float:
    """Validate the target error used to select the Hilbert formula."""
    value = float(value)
    epsilon = float(np.finfo(SPECTRAL_ESTIMATION_DTYPE).eps)
    if not np.isfinite(value) or not 64.0 * epsilon <= value < 1.0:
        raise ValueError(
            "nonlinear Hilbert relative error must be finite and in "
            f"[{64.0 * epsilon:.8g}, 1)."
        )
    return value


def _hilbert_asymptotic_switch(relative_error: float) -> float:
    """Return the argument where direct cancellation reaches the target."""
    relative_error = validate_hilbert_relative_error(relative_error)
    epsilon = float(np.finfo(SPECTRAL_ESTIMATION_DTYPE).eps)
    return max(
        _MINIMUM_ASYMPTOTIC_ARGUMENT,
        float(np.sqrt(relative_error / epsilon)),
    )


def _hilbert_transform_kernel(
    standardized: np.ndarray,
    *,
    relative_error: float,
) -> tuple[np.ndarray, int, float]:
    """Evaluate the Hilbert kernel without large-argument cancellation.

    The direct formula subtracts two terms of order ``abs(x)`` to recover a
    result of order ``1 / abs(x)``.  Its relative roundoff therefore grows
    approximately as ``epsilon * x**2``.  Once that estimate exceeds the
    configured target, evaluate the algebraically equivalent inverse-power
    expansion instead.
    """
    values = np.asarray(standardized, dtype=SPECTRAL_ESTIMATION_DTYPE)
    absolute = np.abs(values)
    switch = _hilbert_asymptotic_switch(relative_error)
    use_asymptotic = absolute > SPECTRAL_ESTIMATION_DTYPE(switch)
    kernel = np.empty_like(values)

    direct = ~use_asymptotic
    if np.any(direct):
        x = values[direct]
        absolute_x = absolute[direct]
        sqrt_five = SPECTRAL_ESTIMATION_DTYPE(_SQRT_FIVE)
        log_ratio = np.empty_like(x)
        inside = absolute_x < sqrt_five
        outside = absolute_x > sqrt_five
        boundary = ~(inside | outside)

        # These atanh forms are the same logarithm as the published formula,
        # but avoid first rounding a ratio very close to one.
        log_ratio[inside] = -SPECTRAL_ESTIMATION_DTYPE(2.0) * np.arctanh(
            x[inside] / sqrt_five
        )
        log_ratio[outside] = -SPECTRAL_ESTIMATION_DTYPE(2.0) * np.arctanh(
            sqrt_five / x[outside]
        )
        log_ratio[boundary] = SPECTRAL_ESTIMATION_DTYPE(0.0)

        quadratic = SPECTRAL_ESTIMATION_DTYPE(1.0) - (
            x * x / SPECTRAL_ESTIMATION_DTYPE(5.0)
        )
        direct_kernel = (
            SPECTRAL_ESTIMATION_DTYPE(_HILBERT_LINEAR_CONSTANT) * x
            + SPECTRAL_ESTIMATION_DTYPE(_HILBERT_LOG_CONSTANT)
            * quadratic
            * log_ratio
        )
        # At |x| = sqrt(5), quadratic * log_ratio has limiting value zero.
        direct_kernel[boundary] = (
            SPECTRAL_ESTIMATION_DTYPE(_HILBERT_LINEAR_CONSTANT)
            * x[boundary]
        )
        kernel[direct] = direct_kernel

    if np.any(use_asymptotic):
        x = values[use_asymptotic]
        inverse_squared = SPECTRAL_ESTIMATION_DTYPE(1.0) / (x * x)
        polynomial = np.full_like(x, _ASYMPTOTIC_COEFFICIENTS[-1])
        for coefficient in reversed(_ASYMPTOTIC_COEFFICIENTS[:-1]):
            polynomial = (
                SPECTRAL_ESTIMATION_DTYPE(coefficient)
                + inverse_squared * polynomial
            )
        kernel[use_asymptotic] = -polynomial / (
            SPECTRAL_ESTIMATION_DTYPE(np.pi) * x
        )

    if not np.isfinite(kernel).all():
        raise ValueError(
            "Analytical nonlinear shrinkage produced a non-finite "
            "Hilbert-transform kernel."
        )
    return kernel, int(np.count_nonzero(use_asymptotic)), switch


def _spectral_density_and_hilbert_transform(
    eigenvalues_ascending: np.ndarray,
    *,
    bandwidth: float,
    hilbert_relative_error: float,
    block_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate the positive spectral density and its Hilbert transform.

    Blocking the target eigenvalues avoids the several full ``r x r``
    workspaces created by the compact reference implementation.  The formula
    itself is unchanged: every target eigenvalue is compared with the complete
    positive observed spectrum.
    """
    hilbert_relative_error = validate_hilbert_relative_error(
        hilbert_relative_error
    )
    eigenvalues = np.asarray(
        eigenvalues_ascending, dtype=SPECTRAL_ESTIMATION_DTYPE
    )
    count = int(eigenvalues.size)
    density = np.empty(count, dtype=SPECTRAL_ESTIMATION_DTYPE)
    hilbert = np.empty(count, dtype=SPECTRAL_ESTIMATION_DTYPE)
    source = eigenvalues[None, :]
    local_bandwidth = (
        SPECTRAL_ESTIMATION_DTYPE(bandwidth) * source
    ).astype(SPECTRAL_ESTIMATION_DTYPE, copy=False)
    asymptotic_pair_count = 0
    maximum_absolute_argument = 0.0
    asymptotic_switch = _hilbert_asymptotic_switch(hilbert_relative_error)

    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        target = eigenvalues[start:stop, None]
        standardized = (target - source) / local_bandwidth
        maximum_absolute_argument = max(
            maximum_absolute_argument,
            float(np.max(np.abs(standardized))),
        )
        quadratic = SPECTRAL_ESTIMATION_DTYPE(1.0) - (
            standardized * standardized / SPECTRAL_ESTIMATION_DTYPE(5.0)
        )

        kernel = np.maximum(quadratic, SPECTRAL_ESTIMATION_DTYPE(0.0))
        density[start:stop] = SPECTRAL_ESTIMATION_DTYPE(
            _DENSITY_CONSTANT
        ) * np.mean(
            kernel / local_bandwidth,
            axis=1,
            dtype=SPECTRAL_ESTIMATION_DTYPE,
        )

        (
            hilbert_kernel,
            block_asymptotic_count,
            block_switch,
        ) = _hilbert_transform_kernel(
            standardized,
            relative_error=hilbert_relative_error,
        )
        asymptotic_pair_count += block_asymptotic_count
        asymptotic_switch = block_switch
        hilbert[start:stop] = np.mean(
            hilbert_kernel / local_bandwidth,
            axis=1,
            dtype=SPECTRAL_ESTIMATION_DTYPE,
        )

    pair_count = count * count
    diagnostics = {
        "relative_error_target": hilbert_relative_error,
        "estimated_cancellation_switch": asymptotic_switch,
        "asymptotic_term_count": _ASYMPTOTIC_TERM_COUNT,
        "direct_pair_count": pair_count - asymptotic_pair_count,
        "asymptotic_pair_count": asymptotic_pair_count,
        "maximum_absolute_standardized_argument": maximum_absolute_argument,
        "spectral_estimation_dtype": np.dtype(
            SPECTRAL_ESTIMATION_DTYPE
        ).name,
    }
    return density, hilbert, diagnostics


def analytical_nonlinear_shrinkage_eigenvalues(
    observed_eigenvalues: np.ndarray,
    *,
    effective_sample_size: int,
    dimension: int,
    output_dtype: Any = PREPROCESSING_DTYPE,
    hilbert_relative_error: float = DEFAULT_HILBERT_RELATIVE_ERROR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Ledoit--Wolf cleaned eigenvalues on the observed support.

    Parameters
    ----------
    observed_eigenvalues:
        Strictly positive raw sample-covariance eigenvalues in descending
        order.  Numerical-null directions must already have been removed.
    effective_sample_size:
        Covariance degrees of freedom.  For centered observations this is
        ``n_samples - 1``.
    dimension:
        Ambient feature dimension.  It determines whether the nonsingular or
        singular Ledoit--Wolf formula applies; reducing the eigendecomposition
        to a Gram matrix does not change this statistical dimension.

    Returns
    -------
    cleaned_eigenvalues, diagnostics
        Cleaned positive eigenvalues in the same descending-eigenvector order,
        plus JSON-serializable formula diagnostics.  In the singular case the
        reference null-space eigenvalue is diagnostic only.
    """
    hilbert_relative_error = validate_hilbert_relative_error(
        hilbert_relative_error
    )
    eigenvalues_descending = np.asarray(
        observed_eigenvalues, dtype=SPECTRAL_ESTIMATION_DTYPE
    )
    if eigenvalues_descending.ndim != 1 or eigenvalues_descending.size == 0:
        raise ValueError(
            "Analytical nonlinear shrinkage requires a non-empty positive "
            "observed spectrum."
        )
    if (
        not np.isfinite(eigenvalues_descending).all()
        or np.any(eigenvalues_descending <= 0)
    ):
        raise ValueError(
            "Analytical nonlinear shrinkage requires finite, strictly "
            "positive observed eigenvalues."
        )
    if effective_sample_size < 12:
        raise ValueError(
            "Analytical nonlinear shrinkage requires at least 12 effective "
            "observations after subtracting covariance degrees of freedom."
        )
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    positive_count = int(eigenvalues_descending.size)
    maximum_positive_count = min(dimension, effective_sample_size)
    if positive_count > maximum_positive_count:
        raise ValueError(
            "Observed covariance rank exceeds min(dimension, "
            "effective_sample_size)."
        )

    eigenvalues = eigenvalues_descending[::-1].copy()
    bandwidth = float(effective_sample_size ** (-1.0 / 3.0))
    density, hilbert, hilbert_diagnostics = (
        _spectral_density_and_hilbert_transform(
            eigenvalues,
            bandwidth=bandwidth,
            hilbert_relative_error=hilbert_relative_error,
        )
    )

    pi = SPECTRAL_ESTIMATION_DTYPE(np.pi)
    if dimension <= effective_sample_size:
        # An explicit raw tolerance can lower the retained support dimension.
        # In the nonsingular branch, shrinkage is fitted inside that selected
        # support, matching the repository's treatment of scalar LW shrinkage.
        statistical_dimension = positive_count
        concentration = SPECTRAL_ESTIMATION_DTYPE(
            statistical_dimension / effective_sample_size
        )
        imaginary = pi * concentration * eigenvalues * density
        real = (
            SPECTRAL_ESTIMATION_DTYPE(1.0)
            - concentration
            - pi * concentration * eigenvalues * hilbert
        )
        denominator = imaginary * imaginary + real * real
        cleaned = eigenvalues / denominator
        formula_case = "nonsingular"
        reference_null_eigenvalue = None
    else:
        # Appendix C / equations (C.4)--(C.8).  The positive-direction
        # correction is computed from the normalized positive spectrum.  The
        # common null-space eigenvalue is recorded but never used for
        # whitening in this repository.
        statistical_dimension = dimension
        concentration = SPECTRAL_ESTIMATION_DTYPE(
            dimension / effective_sample_size
        )
        denominator = (
            pi
            * pi
            * eigenvalues
            * eigenvalues
            * (density * density + hilbert * hilbert)
        )
        cleaned = eigenvalues / denominator
        formula_case = "singular"

        h = SPECTRAL_ESTIMATION_DTYPE(bandwidth)
        h_squared = h * h
        sqrt_five = SPECTRAL_ESTIMATION_DTYPE(_SQRT_FIVE)
        hilbert_at_zero = (
            SPECTRAL_ESTIMATION_DTYPE(1.0) / pi
            * (
                SPECTRAL_ESTIMATION_DTYPE(3.0)
                / (SPECTRAL_ESTIMATION_DTYPE(10.0) * h_squared)
                + SPECTRAL_ESTIMATION_DTYPE(3.0)
                / (SPECTRAL_ESTIMATION_DTYPE(4.0) * sqrt_five * h)
                * (
                    SPECTRAL_ESTIMATION_DTYPE(1.0)
                    - SPECTRAL_ESTIMATION_DTYPE(1.0)
                    / (SPECTRAL_ESTIMATION_DTYPE(5.0) * h_squared)
                )
                * np.log(
                    (SPECTRAL_ESTIMATION_DTYPE(1.0) + sqrt_five * h)
                    / (SPECTRAL_ESTIMATION_DTYPE(1.0) - sqrt_five * h)
                )
            )
            * np.mean(
                SPECTRAL_ESTIMATION_DTYPE(1.0) / eigenvalues,
                dtype=SPECTRAL_ESTIMATION_DTYPE,
            )
        )
        reference_null_eigenvalue = float(
            SPECTRAL_ESTIMATION_DTYPE(1.0)
            / (
                pi
                * SPECTRAL_ESTIMATION_DTYPE(
                    (dimension - effective_sample_size)
                    / effective_sample_size
                )
                * hilbert_at_zero
            )
        )

    if not np.isfinite(cleaned).all() or np.any(cleaned <= 0):
        raise ValueError(
            "Analytical nonlinear shrinkage returned a non-finite or "
            "non-positive covariance eigenvalue on the observed support."
        )
    if (
        reference_null_eigenvalue is not None
        and (
            not np.isfinite(reference_null_eigenvalue)
            or reference_null_eigenvalue <= 0
        )
    ):
        raise ValueError(
            "Analytical nonlinear shrinkage returned an invalid reference "
            "null-space eigenvalue."
        )

    diagnostics: dict[str, Any] = {
        "formula": "ledoit_wolf_analytical_nonlinear_2020",
        "formula_case": formula_case,
        "bandwidth": bandwidth,
        "effective_sample_size": int(effective_sample_size),
        "ambient_dimension": int(dimension),
        "statistical_dimension": int(statistical_dimension),
        "concentration_ratio": float(concentration),
        "positive_spectrum_count": positive_count,
        "hilbert_kernel": hilbert_diagnostics,
        "reference_null_eigenvalue": reference_null_eigenvalue,
        "reference_null_eigenvalue_used": False,
        "null_space_policy": "discard_raw_numerical_null_directions",
    }
    return (
        np.asarray(cleaned[::-1], dtype=output_dtype),
        diagnostics,
    )
