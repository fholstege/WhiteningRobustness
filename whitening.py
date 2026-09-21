"""Public whitening transforms and transform factory."""

from __future__ import annotations

from typing import Any

from numerics import PREPROCESSING_DTYPE
from whitening_core import (
    IdentityTransform,
    RepresentationTransform,
    StandardizeTransform,
    WhiteningTransform,
    validate_whitening_estimator,
)

# Target relative roundoff for deciding when the analytical nonlinear
# Hilbert kernel switches from its direct formula to its large-argument
# inverse-power expansion. Lower values switch earlier. This is validated
# against the float64 spectral-estimation precision in numerics.py.
NONLINEAR_HILBERT_RELATIVE_ERROR = 1e-12


def make_transform(
    name: str,
    *,
    dtype: Any = PREPROCESSING_DTYPE,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    class_weighted: bool = False,
) -> RepresentationTransform:
    """Construct one canonical representation transform."""
    if name == "identity":
        return IdentityTransform(dtype=dtype)
    if name == "standardize":
        return StandardizeTransform(dtype=dtype)

    explicit = {
        "whiten_ledoit_wolf": ("ledoit-wolf", None),
        "whiten_ledoit_wolf_nonlinear": (
            "ledoit-wolf-nonlinear",
        ),
    }
    if name == "whiten":
        estimator = validate_whitening_estimator(whitening_estimator)
    elif name in explicit:
        estimator = explicit[name][0]
    else:
        raise ValueError(f"Unknown representation transform: {name!r}.")
    return WhiteningTransform(
        dtype=dtype,
        estimator=estimator,
        class_weighted=class_weighted,
        relative_tolerance=whitening_relative_tolerance,
        nonlinear_hilbert_relative_error=(
            NONLINEAR_HILBERT_RELATIVE_ERROR
        ),
    )
