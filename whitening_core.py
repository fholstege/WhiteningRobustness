"""Identity, standardization, and covariance-whitening transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.covariance import ledoit_wolf

from nonlinear_shrinkage import (
    DEFAULT_HILBERT_RELATIVE_ERROR,
    analytical_nonlinear_shrinkage_eigenvalues,
    validate_hilbert_relative_error,
)
from numerics import MODEL_PARAMETER_DTYPE, PREPROCESSING_DTYPE

# Backward-compatible internal name used by the MP whitener. Its value is
# configured centrally in numerics.py.
NUMERICAL_DTYPE = PREPROCESSING_DTYPE


def _validated_matrix(
    X: np.ndarray, *, name: str = "X", dtype: Any = NUMERICAL_DTYPE
) -> np.ndarray:
    """Return a finite matrix in the requested preprocessing precision."""
    dtype = np.dtype(dtype).type
    if dtype not in (np.float32, np.float64):
        raise ValueError("Preprocessing dtype must be float32 or float64.")
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite values.")
    return X


class RepresentationTransform(Protocol):
    """The small interface shared by all representation transforms."""

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "RepresentationTransform": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def inverse_head(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def diagnostics(self) -> dict[str, Any]: ...


WHITENING_ESTIMATORS = (
    "empirical",
    "ledoit-wolf",
    "ledoit-wolf-nonlinear",
)


def validate_whitening_estimator(value: str) -> str:
    """Accept one canonical covariance-estimator name."""
    if value not in WHITENING_ESTIMATORS:
        raise ValueError(
            "whitening_estimator must be 'empirical', 'ledoit-wolf', or "
            "'ledoit-wolf-nonlinear'."
        )
    return value


@dataclass
class IdentityTransform:
    """Pass features through unchanged while retaining their fitted dimension."""

    dtype: Any = field(default=NUMERICAL_DTYPE, kw_only=True)

    n_features_in_: int | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "IdentityTransform":
        X = _validated_matrix(X, dtype=self.dtype)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _validated_matrix(X, dtype=self.dtype)
        self._check_dimension(X)
        return np.asarray(X)

    def inverse_head(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(coef, dtype=MODEL_PARAMETER_DTYPE).copy(),
            np.asarray(intercept, dtype=MODEL_PARAMETER_DTYPE).copy(),
        )

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "name": "identity",
            "input_features": self.n_features_in_,
            "output_features": self.n_features_in_,
            "numerical_dtype": np.dtype(self.dtype).name,
        }

    def _require_fitted(self) -> None:
        if self.n_features_in_ is None:
            raise RuntimeError("Transform has not been fitted.")

    def _check_dimension(self, X: np.ndarray) -> None:
        self._require_fitted()
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from fitted data.")


@dataclass
class StandardizeTransform:
    """Coordinate-wise population standardization fitted on training data."""

    dtype: Any = field(default=NUMERICAL_DTYPE, kw_only=True)

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    constant_features_: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "StandardizeTransform":
        X = _validated_matrix(X, dtype=self.dtype)
        self.dtype = np.dtype(self.dtype).type
        # These statistics are computed on train and reused unchanged elsewhere.
        self.mean_ = X.mean(axis=0, dtype=self.dtype)
        std = X.std(axis=0, ddof=0, dtype=self.dtype)
        threshold = np.finfo(NUMERICAL_DTYPE).eps
        self.constant_features_ = np.flatnonzero(std <= threshold)
        self.scale_ = np.where(
            std > threshold, std, self.dtype(1.0)
        ).astype(self.dtype, copy=False)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _validated_matrix(X, dtype=self.dtype)
        self._check_dimension(X)
        transformed = X - self.mean_
        transformed /= self.scale_
        return transformed

    def inverse_head(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        coef = np.asarray(coef, dtype=MODEL_PARAMETER_DTYPE)
        intercept = np.asarray(intercept, dtype=MODEL_PARAMETER_DTYPE)
        coef_original = coef / self.scale_[None, :]
        intercept_original = intercept - coef_original @ self.mean_
        return coef_original, intercept_original

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "name": "standardize",
            "input_features": int(self.mean_.size),
            "output_features": int(self.mean_.size),
            "constant_feature_count": int(self.constant_features_.size),
            "variance_ddof": 0,
            "numerical_dtype": np.dtype(self.dtype).name,
        }

    def _require_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Transform has not been fitted.")

    def _check_dimension(self, X: np.ndarray) -> None:
        self._require_fitted()
        if X.shape[1] != self.mean_.size:
            raise ValueError("Feature dimension differs from fitted data.")



@dataclass
class WhiteningTransform:
    """Empirical, linear, or analytical nonlinear covariance whitening.

    ``relative_tolerance`` identifies the raw empirical covariance support. A
    raw eigenvalue is kept exactly when

        eigenvalue > relative_tolerance * largest_eigenvalue.

    When no value is supplied, the relative tolerance is float32 machine
    epsilon. Multiplying float32 epsilon by the much larger observation count
    would turn numerical-rank protection into aggressive spectral truncation
    on datasets such as CelebA. This support is selected before fitting every
    covariance estimator. Shrinkage therefore cannot revive an empirical null
    direction, and no second cutoff is applied to the positive eigenvalues
    returned within the retained support.
    """

    dtype: Any = field(default=NUMERICAL_DTYPE, kw_only=True)

    estimator: str = "empirical"
    class_weighted: bool = False
    ridge: float = 0.0
    ddof: int = 1
    relative_tolerance: float | None = None
    nonlinear_hilbert_relative_error: float = (
        DEFAULT_HILBERT_RELATIVE_ERROR
    )
    mean_: np.ndarray | None = None
    basis_: np.ndarray | None = None
    eigenvalues_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    tolerance_: float | None = None
    rank_: int | None = None
    shrinkage_: float | None = None
    decomposition_: str | None = None
    n_samples_fit_: int | None = None
    covariance_shape_: tuple[int, int] | None = None
    effective_relative_tolerance_: float | None = None
    candidate_eigenvalue_count_: int | None = None
    discarded_eigenvalue_count_: int | None = None
    pre_shrinkage_rank_: int | None = None
    pre_shrinkage_discarded_count_: int | None = None
    nonlinear_shrinkage_diagnostics_: dict[str, Any] | None = None
    numerical_rank_: int | None = None
    class_weight_summary_: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.estimator = validate_whitening_estimator(self.estimator)
        if self.ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        if self.ddof < 0:
            raise ValueError("ddof must be nonnegative.")
        if self.relative_tolerance is not None and self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be nonnegative.")
        self.nonlinear_hilbert_relative_error = (
            validate_hilbert_relative_error(
                self.nonlinear_hilbert_relative_error
            )
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "WhiteningTransform":
        self.dtype = np.dtype(self.dtype).type
        X = _validated_matrix(X, dtype=self.dtype)
        n, d = X.shape
        if self.class_weighted:
            if y is None:
                raise ValueError(
                    "class_weighted whitening requires labels y when fitting."
                )
            y = np.asarray(y).reshape(-1)
            if y.size != n:
                raise ValueError("y must contain one label per whitening row.")
            classes, inverse, counts = np.unique(
                y, return_inverse=True, return_counts=True
            )
            if classes.size < 2:
                raise ValueError(
                    "class_weighted whitening requires at least two classes."
                )
            weights = n / (classes.size * counts[inverse])
            sum_weights = float(weights.sum())
            denominator = sum_weights - float(weights @ weights) / sum_weights
            self.class_weight_summary_ = {
                "enabled": True,
                "classes": classes.tolist(),
                "class_counts": {
                    str(label): int(count)
                    for label, count in zip(classes, counts)
                },
                "minimum": float(weights.min()),
                "maximum": float(weights.max()),
                "mean": float(weights.mean()),
                "sum": sum_weights,
                "effective_sample_size": float(
                    sum_weights**2 / float(weights @ weights)
                ),
            }
        else:
            weights = None
            denominator = n - self.ddof
            self.class_weight_summary_ = {"enabled": False}
        if denominator <= 0:
            raise ValueError(f"Need n > ddof, received n={n}, ddof={self.ddof}.")

        self.n_samples_fit_ = n
        self.mean_ = (
            X.mean(axis=0, dtype=self.dtype)
            if weights is None
            else np.average(X, axis=0, weights=weights).astype(
                self.dtype, copy=False
            )
        )
        centered = X - self.mean_
        covariance_rows = (
            centered
            if weights is None
            else centered * np.sqrt(weights)[:, None]
        )

        # First identify the empirical support, independently of the requested
        # covariance estimator. When n < d, work through the n x n Gram matrix
        # and lift its retained eigenvectors into feature space. This prevents
        # either shrinkage estimator from assigning variance to sample-null or
        # representation-null directions.
        if n < d:
            raw_covariance = covariance_rows @ covariance_rows.T / denominator
            raw_eigenvalues, raw_vectors = np.linalg.eigh(
                0.5 * (raw_covariance + raw_covariance.T)
            )
            raw_decomposition = "sample_covariance_eigh"
            raw_covariance_shape = raw_covariance.shape
            raw_vectors_are_in_sample_space = True
        else:
            raw_covariance = covariance_rows.T @ covariance_rows / denominator
            raw_eigenvalues, raw_vectors = np.linalg.eigh(
                0.5 * (raw_covariance + raw_covariance.T)
            )
            raw_decomposition = "feature_covariance_eigh"
            raw_covariance_shape = raw_covariance.shape
            raw_vectors_are_in_sample_space = False

        raw_order = np.argsort(raw_eigenvalues)[::-1]
        raw_eigenvalues = np.asarray(
            raw_eigenvalues[raw_order], dtype=self.dtype
        )
        raw_vectors = np.asarray(
            raw_vectors[:, raw_order], dtype=self.dtype
        )
        maximum = float(raw_eigenvalues[0]) if raw_eigenvalues.size else 0.0
        if maximum <= 0:
            raise ValueError("Training covariance has rank zero.")
        # Keep support selection unchanged when arithmetic uses float64.
        relative = (
            self.relative_tolerance
            if self.relative_tolerance is not None
            else np.finfo(NUMERICAL_DTYPE).eps
        )
        self.effective_relative_tolerance_ = float(relative)
        self.tolerance_ = float(relative * maximum)

        numerical_keep = raw_eigenvalues > self.tolerance_
        if not np.any(numerical_keep):
            raise ValueError(
                "No raw covariance eigenvalues exceed the rank tolerance."
            )
        numerical_eigenvalues = raw_eigenvalues[numerical_keep]
        self.numerical_rank_ = int(numerical_eigenvalues.size)
        raw_keep = numerical_keep.copy()
        retained_raw_eigenvalues = raw_eigenvalues[raw_keep]
        if raw_vectors_are_in_sample_space:
            sample_vectors = raw_vectors[:, raw_keep]
            denominators = np.sqrt(
                denominator * retained_raw_eigenvalues
            )
            raw_basis = (
                covariance_rows.T @ sample_vectors
            ) / denominators[None, :]
        else:
            raw_basis = raw_vectors[:, raw_keep]
        raw_rank = int(retained_raw_eigenvalues.size)
        raw_candidate_count = int(raw_eigenvalues.size)
        self.nonlinear_shrinkage_diagnostics_ = None

        # The raw covariance/eigendecomposition workspaces are no longer
        # needed once the support basis has been lifted. Releasing them before
        # a shrinkage fit is important for the large n >= d bundles.
        del raw_covariance, raw_eigenvalues, raw_vectors

        if self.estimator == "empirical":
            self.eigenvalues_ = retained_raw_eigenvalues
            self.basis_ = raw_basis
            self.shrinkage_ = None
            self.decomposition_ = raw_decomposition
            self.covariance_shape_ = raw_covariance_shape
            self.candidate_eigenvalue_count_ = raw_candidate_count
            self.discarded_eigenvalue_count_ = int((~raw_keep).sum())
        elif self.estimator == "ledoit-wolf":
            # Fit shrinkage only in the empirically observed subspace. The
            # fitted estimator supplies a soft eigenvalue floor inside this
            # support; every positive returned eigenvalue is retained.
            self.pre_shrinkage_rank_ = raw_rank
            self.pre_shrinkage_discarded_count_ = d - raw_rank
            # With full raw support, an orthogonal rotation is unnecessary:
            # Ledoit--Wolf is rotation-equivariant, so fitting directly on
            # centered avoids a second n-by-d float32 allocation. A projected
            # score matrix is still required when an explicit raw cutoff has
            # removed feature directions.
            estimator_input = (
                covariance_rows
                if raw_rank == d
                else covariance_rows @ raw_basis
            )
            estimator_basis_lift = (
                None if raw_rank == d else raw_basis
            )
            covariance, shrinkage = ledoit_wolf(
                estimator_input, assume_centered=True
            )
            covariance = np.asarray(covariance, dtype=self.dtype)
            if self.ddof:
                covariance *= n / denominator
            self.shrinkage_ = float(shrinkage)
            estimator_suffix = "ledoit_wolf"
            if not np.isfinite(covariance).all():
                raise ValueError(
                    f"{self.estimator} returned a non-finite covariance "
                    "inside the raw empirical support."
                )
            eigenvalues, estimator_basis = np.linalg.eigh(
                0.5 * (covariance + covariance.T)
            )
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.asarray(
                eigenvalues[order], dtype=self.dtype
            )
            estimator_basis = np.asarray(
                estimator_basis[:, order], dtype=self.dtype
            )
            if eigenvalues.size != raw_rank or np.any(eigenvalues <= 0):
                raise ValueError(
                    f"{self.estimator} returned a non-positive covariance "
                    "eigenvalue inside the raw empirical support."
                )
            self.eigenvalues_ = eigenvalues
            self.basis_ = (
                estimator_basis
                if estimator_basis_lift is None
                else estimator_basis_lift @ estimator_basis
            )
            self.decomposition_ = (
                f"projected_feature_covariance_eigh_{estimator_suffix}"
                if raw_rank < d
                else f"feature_covariance_eigh_{estimator_suffix}"
            )
            self.covariance_shape_ = covariance.shape
            self.candidate_eigenvalue_count_ = raw_rank
            self.discarded_eigenvalue_count_ = 0
        else:
            # Analytical nonlinear shrinkage changes only the raw positive
            # eigenvalues, so the empirical eigenvectors remain unchanged.
            # In the singular d > (n - ddof) case, the published formula also
            # estimates a common variance for sample-null directions.  The
            # helper records that value, but this transform deliberately omits
            # it and retains only the raw observed covariance support.
            self.pre_shrinkage_rank_ = raw_rank
            self.pre_shrinkage_discarded_count_ = d - raw_rank
            (
                self.eigenvalues_,
                self.nonlinear_shrinkage_diagnostics_,
            ) = analytical_nonlinear_shrinkage_eigenvalues(
                retained_raw_eigenvalues,
                # Weighted covariance changes the estimated moment, but the
                # nonlinear-shrinkage aspect ratio still counts independent
                # rows.  Substituting Kish effective sample size can make it
                # smaller than the observed covariance rank when n < d,
                # where the analytical formula is undefined.
                effective_sample_size=n - self.ddof,
                dimension=d,
                output_dtype=self.dtype,
                hilbert_relative_error=(
                    self.nonlinear_hilbert_relative_error
                ),
            )
            self.basis_ = raw_basis
            self.shrinkage_ = None
            self.decomposition_ = (
                f"{raw_decomposition}_analytical_nonlinear_shrinkage"
            )
            self.covariance_shape_ = raw_covariance_shape
            self.candidate_eigenvalue_count_ = raw_rank
            self.discarded_eigenvalue_count_ = 0

        # Whitening divides retained coordinates by sqrt(eigenvalue + ridge).
        # The optional ridge stabilizes retained directions only.
        self.scales_ = np.sqrt(
            self.eigenvalues_ + self.dtype(self.ridge)
        ).astype(self.dtype, copy=False)
        self.rank_ = raw_rank
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _validated_matrix(X, dtype=self.dtype)
        self._check_dimension(X)
        # Projection removes discarded/null directions. Empirical scales make
        # the fitted sample covariance exactly identity; shrinkage scales
        # whiten with respect to the corresponding covariance estimate.
        transformed = (X - self.mean_) @ self.basis_
        transformed /= self.scales_
        return transformed

    def inverse_head(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        coef = np.asarray(coef, dtype=MODEL_PARAMETER_DTYPE)
        intercept = np.asarray(intercept, dtype=MODEL_PARAMETER_DTYPE)
        if coef.ndim != 2 or coef.shape[1] != self.rank_:
            raise ValueError("Head coefficient dimension does not match whitening rank.")
        # Undo projection/scaling so logits can be evaluated in original
        # embedding coordinates without changing predictions.
        coef_original = (
            self.basis_ @ (coef.T / self.scales_[:, None])
        ).T
        intercept_original = intercept - coef_original @ self.mean_
        return coef_original, intercept_original

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        method_names = {
            "empirical": "whiten",
            "ledoit-wolf": "whiten_ledoit_wolf",
            "ledoit-wolf-nonlinear": "whiten_ledoit_wolf_nonlinear",
        }
        method_name = method_names[self.estimator]
        if self.class_weighted:
            method_name += "_class_weighted"
        return {
            "name": method_name,
            "estimator": self.estimator,
            "mode": "covariance_whitening",
            "numerical_rank": self.numerical_rank_,
            "ridge": self.ridge,
            "covariance_ddof": self.ddof,
            "class_weighted": bool(self.class_weighted),
            "class_weight_summary": self.class_weight_summary_,
            "input_features": int(self.mean_.size),
            "output_features": self.rank_,
            "retained_rank": self.rank_,
            "candidate_eigenvalue_count": self.candidate_eigenvalue_count_,
            "discarded_eigenvalue_count": self.discarded_eigenvalue_count_,
            "pre_shrinkage_rank": self.pre_shrinkage_rank_,
            "pre_shrinkage_discarded_count": (
                self.pre_shrinkage_discarded_count_
            ),
            "configured_relative_tolerance": self.relative_tolerance,
            "nonlinear_hilbert_relative_error": (
                self.nonlinear_hilbert_relative_error
                if self.estimator == "ledoit-wolf-nonlinear"
                else None
            ),
            "effective_relative_tolerance": self.effective_relative_tolerance_,
            "rank_tolerance": self.tolerance_,
            "minimum_retained_eigenvalue": float(np.min(self.eigenvalues_)),
            "maximum_eigenvalue": float(np.max(self.eigenvalues_)),
            "shrinkage": self.shrinkage_,
            "shrinkage_kind": {
                "empirical": "none",
                "ledoit-wolf": "linear_scalar",
                "ledoit-wolf-nonlinear": "analytical_nonlinear_spectral",
            }[self.estimator],
            "nonlinear_shrinkage": self.nonlinear_shrinkage_diagnostics_,
            "decomposition": self.decomposition_,
            "covariance_shape": list(self.covariance_shape_),
            "fit_observations": self.n_samples_fit_,
            "numerical_dtype": np.dtype(self.dtype).name,
        }

    def _require_fitted(self) -> None:
        if (
            self.mean_ is None
            or self.basis_ is None
            or self.eigenvalues_ is None
            or self.scales_ is None
            or self.rank_ is None
        ):
            raise RuntimeError("Transform has not been fitted.")

    def _check_dimension(self, X: np.ndarray) -> None:
        self._require_fitted()
        if X.shape[1] != self.mean_.size:
            raise ValueError("Feature dimension differs from fitted data.")
