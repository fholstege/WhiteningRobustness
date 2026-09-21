"""Comparison methods built on frozen representation bundles.

Deep Feature Reweighting (DFR) deliberately uses validation group labels. The
original validation split is divided into a group-stratified fitting pool and
selection half. The fitting pool is then sampled to equal group counts before
the representation transform and ordinary ridge-logistic head are fitted.

Automatic Feature Reweighting (AFR) uses the saved trained downstream head to
assign fixed confidence-based weights on a held-out adaptation split. L-BFGS
then fits an ERM-centered correction with those weights, without group labels.

NeuroTune uses a label-stratified first half of validation to identify
coordinates whose activation magnitude is larger on errors than on correct
examples of the same class. It fits the masked head on the other validation
half and selects threshold and ridge by exact class-balanced accuracy on the
identification half. The paper's group-free SFit criterion is retained as a
reported diagnostic and optional aggregation rule.

Held-out ERM (HO_ERM) is the matched label-only comparison: it fits a
transform and class-balanced logistic head on one label-stratified validation
half and selects ridge on the untouched half. Group labels are never used.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from data import EmbeddingBundle, SavedLinearHead
from metric import evaluation_metrics, selection_metric_statistics
from model import (
    LogisticConfig,
    LogisticFit,
    fit_logistic,
    fit_logistic_from_reference,
    reference_head_in_transform as _reference_head_in_transform,
)
from whitening import make_transform


FITTING_METHODS = ("ERM", "DFR", "AFR", "NEUROTUNE")


def validate_fitting_method(value: str) -> str:
    if value not in FITTING_METHODS:
        raise ValueError(
            "method must be ERM, DFR, AFR, or NEUROTUNE."
        )
    return value


# =============================================================================
# Shared validation helpers and AFR weighting
# =============================================================================

def _validate_method_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"{method} X and y have incompatible shapes.")
    if y.size == 0:
        raise ValueError(f"{method} requires non-empty data.")
    if not np.isfinite(X).all():
        raise ValueError(f"{method} X contains non-finite values.")
    return X, y


def compute_afr_weights(
    score_head: Any,
    X: np.ndarray,
    y: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean-one AFR weights based on fixed correct-class confidence.

    AFR assigns observation ``i`` an unnormalised weight
    ``exp(-gamma * p(y_i | x_i)) / n_{y_i}``. The saved head supplies the
    probabilities and the class counts are calculated only on the adaptation
    split. Group labels never enter this calculation.
    """
    X, y = _validate_method_arrays(X, y, method="AFR")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("AFR gamma must be finite and nonnegative.")
    if not hasattr(score_head, "classes_"):
        raise ValueError("AFR score_head must be a fitted classifier.")
    classes = np.asarray(score_head.classes_)
    if classes.ndim != 1 or classes.size < 2 or not np.isin(y, classes).all():
        raise ValueError("AFR labels must belong to score_head.classes_.")
    probabilities = np.asarray(score_head.predict_proba(X), dtype=np.float64)
    if probabilities.shape != (y.size, classes.size):
        raise ValueError("AFR requires one probability column per score-head class.")
    if not np.isfinite(probabilities).all():
        raise ValueError("AFR score-head probabilities contain non-finite values.")
    class_indices = np.argmax(y[:, None] == classes[None, :], axis=1)
    correct_class_probability = probabilities[np.arange(y.size), class_indices]
    counts = np.bincount(class_indices, minlength=classes.size).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("AFR adaptation data must contain every score-head class.")
    log_weights = -gamma * correct_class_probability - np.log(counts[class_indices])
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    weights /= weights.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("AFR produced invalid sample weights.")
    class_weight_totals = np.bincount(
        class_indices, weights=weights, minlength=classes.size
    )
    return weights, {
        "paper_method": "AFR",
        "gamma": gamma,
        "classes": classes.tolist(),
        "class_counts": {
            str(label.item() if hasattr(label, "item") else label): int(count)
            for label, count in zip(classes, counts)
        },
        "correct_class_probability": {
            "minimum": float(correct_class_probability.min()),
            "maximum": float(correct_class_probability.max()),
            "mean": float(correct_class_probability.mean()),
            "sha256": sha256(
                np.asarray(correct_class_probability, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "sample_weight": {
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "mean": float(weights.mean()),
            "sum": float(weights.sum()),
            "effective_sample_size": float(
                weights.sum() ** 2 / np.square(weights).sum()
            ),
            "class_totals": {
                str(label.item() if hasattr(label, "item") else label): float(total)
                for label, total in zip(classes, class_weight_totals)
            },
            "sha256": sha256(
                np.asarray(weights, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "uses_group_labels": False,
        "proximity_penalty": True,
    }


def afr_method_name(transform_name: str) -> str:
    """Prefix a canonical fitted transform name with the AFR family."""
    normalized = str(transform_name).strip().lower().replace("-", "_")
    return f"afr_{normalized}"


def _validated_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("DFR random_state must be an integer.")
    seed = int(value)
    if seed < 0:
        raise ValueError("DFR random_state must be nonnegative.")
    return seed


def _index_hash(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    return sha256(values.tobytes(order="C")).hexdigest()


def _group_counts(groups: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    selected = groups[np.asarray(indices, dtype=np.int64)]
    return {
        str(int(group)): int(np.sum(selected == group))
        for group in np.unique(groups)
    }


def _validate_partition(partition: "DFRPartition", groups: np.ndarray) -> None:
    """Reject externally supplied partitions that violate the DFR protocol."""
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if partition.observations != groups.size:
        raise ValueError("DFR partition does not match validation length.")
    index_sets = {
        "fit pool": np.asarray(partition.fit_pool_indices),
        "balanced train": np.asarray(partition.balanced_train_indices),
        "selection": np.asarray(partition.selection_indices),
    }
    for name, indices in index_sets.items():
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"DFR {name} indices must be a one-dimensional integer array.")
        if indices.size == 0:
            raise ValueError(f"DFR {name} must be non-empty.")
        if np.any(indices < 0) or np.any(indices >= groups.size):
            raise ValueError(f"DFR {name} contains an out-of-range index.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"DFR {name} contains duplicate observations.")

    fit_pool = index_sets["fit pool"]
    balanced = index_sets["balanced train"]
    selection = index_sets["selection"]
    if abs(int(fit_pool.size) - int(selection.size)) > 1:
        raise ValueError("DFR validation halves must differ in size by at most one.")
    if np.intersect1d(fit_pool, selection).size:
        raise ValueError("DFR fit and selection halves overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((fit_pool, selection))),
        np.arange(groups.size, dtype=np.int64),
    ):
        raise ValueError("DFR halves must cover validation exactly once.")
    if not np.all(np.isin(balanced, fit_pool)):
        raise ValueError("Balanced DFR observations must come from the fit pool.")

    expected_groups = np.unique(groups)
    for name, indices in (("fit pool", fit_pool), ("selection", selection)):
        if not np.array_equal(np.unique(groups[indices]), expected_groups):
            raise ValueError(f"DFR {name} does not contain every group.")
    balanced_counts = np.asarray(
        [np.sum(groups[balanced] == group) for group in expected_groups]
    )
    if np.any(balanced_counts <= 0) or np.unique(balanced_counts).size != 1:
        raise ValueError(
            "Balanced DFR training must contain the same positive count from "
            "every group."
        )


@dataclass(frozen=True)
class DFRPartition:
    """Indices defining one reproducible DFR fit/selection protocol."""

    random_state: int
    observations: int
    fit_pool_indices: np.ndarray
    balanced_train_indices: np.ndarray
    selection_indices: np.ndarray

    def diagnostics(self, groups: np.ndarray) -> dict[str, Any]:
        groups = np.asarray(groups, dtype=np.int64).reshape(-1)
        if groups.size != self.observations:
            raise ValueError("DFR diagnostics groups do not match the partition.")
        balanced_counts = _group_counts(groups, self.balanced_train_indices)
        return {
            "source_split": "val",
            "random_state": self.random_state,
            "rng_derivation": "numpy_seed_sequence_two_child_streams",
            "splitter": "StratifiedShuffleSplit",
            "stratification": "group",
            "test_size": 0.5,
            "observations": self.observations,
            "fit_pool_observations": int(self.fit_pool_indices.size),
            "balanced_train_observations": int(
                self.balanced_train_indices.size
            ),
            "selection_observations": int(self.selection_indices.size),
            "unused_fit_pool_observations": int(
                self.fit_pool_indices.size - self.balanced_train_indices.size
            ),
            "fit_pool_group_counts": _group_counts(
                groups, self.fit_pool_indices
            ),
            "balanced_train_group_counts": balanced_counts,
            "selection_group_counts": _group_counts(
                groups, self.selection_indices
            ),
            "samples_per_group": min(balanced_counts.values()),
            "index_sha256": {
                "fit_pool": _index_hash(self.fit_pool_indices),
                "balanced_train": _index_hash(
                    self.balanced_train_indices
                ),
                "selection": _index_hash(self.selection_indices),
            },
        }


def make_dfr_partition(groups: np.ndarray, *, random_state: int) -> DFRPartition:
    """Split validation by group and balance the DFR fitting pool."""
    seed = _validated_seed(random_state)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if groups.size == 0:
        raise ValueError("DFR requires a non-empty validation group vector.")
    if np.any(groups < 0):
        raise ValueError("DFR group identifiers must be nonnegative.")
    unique_groups, counts = np.unique(groups, return_counts=True)
    if unique_groups.size < 2:
        raise ValueError("DFR requires at least two validation groups.")
    if np.any(counts < 2):
        sparse = unique_groups[counts < 2].tolist()
        raise ValueError(
            "Every DFR group needs at least two validation observations; "
            f"insufficient groups: {sparse}."
        )

    split_sequence, sample_sequence = np.random.SeedSequence(seed).spawn(2)
    split_state = int(split_sequence.generate_state(1, dtype=np.uint32)[0])
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=split_state,
    )
    dummy = np.zeros((groups.size, 1), dtype=np.uint8)
    fit_pool, selection = next(splitter.split(dummy, groups))
    fit_pool = np.asarray(fit_pool, dtype=np.int64)
    selection = np.asarray(selection, dtype=np.int64)

    expected_groups = set(int(group) for group in unique_groups)
    for name, indices in (("fit pool", fit_pool), ("selection", selection)):
        observed_groups = set(int(group) for group in np.unique(groups[indices]))
        if observed_groups != expected_groups:
            raise ValueError(f"DFR {name} does not contain every group.")
    if np.intersect1d(fit_pool, selection).size:
        raise RuntimeError("DFR fit and selection halves overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((fit_pool, selection))),
        np.arange(groups.size, dtype=np.int64),
    ):
        raise RuntimeError("DFR halves do not cover validation exactly once.")

    fit_counts = np.asarray(
        [np.sum(groups[fit_pool] == group) for group in unique_groups],
        dtype=np.int64,
    )
    samples_per_group = int(fit_counts.min())
    if samples_per_group <= 0:
        raise ValueError("DFR fitting pool contains an empty group.")
    sample_rng = np.random.default_rng(sample_sequence)
    balanced_parts = []
    for group in unique_groups:
        candidates = fit_pool[groups[fit_pool] == group]
        balanced_parts.append(
            sample_rng.choice(
                candidates,
                size=samples_per_group,
                replace=False,
            )
        )
    balanced = np.concatenate(balanced_parts).astype(np.int64, copy=False)
    balanced = sample_rng.permutation(balanced)
    if np.unique(balanced).size != balanced.size:
        raise RuntimeError("DFR subgroup sampling selected duplicate observations.")

    partition = DFRPartition(
        random_state=seed,
        observations=int(groups.size),
        fit_pool_indices=fit_pool,
        balanced_train_indices=balanced,
        selection_indices=selection,
    )
    _validate_partition(partition, groups)
    return partition


class SUBGClassifier:
    """Group-subsampled transform plus ordinary ridge-logistic classifier."""

    def __init__(
        self,
        *,
        ridge_lambda: float,
        transform_name: str = "identity",
        random_state: int = 0,
        whitening_relative_tolerance: float | None = None,
        whitening_estimator: str = "empirical",
    ) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.transform_name = str(transform_name)
        self.random_state = _validated_seed(random_state)
        self.whitening_relative_tolerance = whitening_relative_tolerance
        self.whitening_estimator = str(whitening_estimator)
        self.partition_: DFRPartition | None = None
        self.transform_: Any | None = None
        self.fit_: LogisticFit | None = None

    @staticmethod
    def _validate_arrays(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        groups = np.asarray(groups, dtype=np.int64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size != groups.size:
            raise ValueError("DFR X, y, and groups have incompatible shapes.")
        if not np.isfinite(X).all():
            raise ValueError("DFR X contains non-finite values.")
        return X, y, groups

    @classmethod
    def fit_ridge_path(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        ridge_lambdas: Iterable[float],
        transform_name: str,
        random_state: int,
        partition: DFRPartition | None = None,
        whitening_relative_tolerance: float | None = None,
        whitening_estimator: str = "empirical",
    ) -> list["SUBGClassifier"]:
        """Fit one balanced transform and reuse it over a ridge path."""
        X, y, groups = cls._validate_arrays(X, y, groups)
        ridge_values = tuple(float(value) for value in ridge_lambdas)
        if not ridge_values:
            raise ValueError("DFR requires at least one ridge_lambda.")
        if len(set(ridge_values)) != len(ridge_values):
            raise ValueError("DFR ridge_lambdas contains duplicates.")
        for value in ridge_values:
            LogisticConfig(ridge_lambda=value)

        partition = partition or make_dfr_partition(
            groups,
            random_state=random_state,
        )
        _validate_partition(partition, groups)
        balanced = partition.balanced_train_indices
        selection = partition.selection_indices
        expected_classes = np.unique(y)
        if expected_classes.size < 2:
            raise ValueError("DFR requires at least two validation classes.")
        for name, indices in (
            ("fit pool", partition.fit_pool_indices),
            ("balanced training sample", balanced),
            ("selection half", selection),
        ):
            if not np.array_equal(np.unique(y[indices]), expected_classes):
                raise ValueError(f"DFR {name} must contain every class.")

        transform = make_transform(
            transform_name,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
        ).fit(X[balanced])
        transformed = transform.transform(X[balanced])
        classifiers: list[SUBGClassifier] = []
        for ridge_lambda in ridge_values:
            fit = fit_logistic(
                transformed,
                y[balanced],
                LogisticConfig(
                    ridge_lambda=ridge_lambda,
                    class_balanced=False,
                ),
            )
            if not fit.converged:
                messages = "; ".join(fit.convergence_messages)
                raise RuntimeError(
                    f"DFR {transform_name} ridge={ridge_lambda:g} did not "
                    f"converge: {messages}"
                )
            classifier = cls(
                ridge_lambda=ridge_lambda,
                transform_name=transform_name,
                random_state=random_state,
                whitening_relative_tolerance=whitening_relative_tolerance,
                whitening_estimator=whitening_estimator,
            )
            classifier.partition_ = partition
            classifier.transform_ = transform
            classifier.fit_ = fit
            classifiers.append(classifier)
        return classifiers

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        partition: DFRPartition | None = None,
    ) -> "SUBGClassifier":
        fitted = self.fit_ridge_path(
            X,
            y,
            groups,
            ridge_lambdas=(self.ridge_lambda,),
            transform_name=self.transform_name,
            random_state=self.random_state,
            partition=partition,
            whitening_relative_tolerance=self.whitening_relative_tolerance,
            whitening_estimator=self.whitening_estimator,
        )[0]
        self.partition_ = fitted.partition_
        self.transform_ = fitted.transform_
        self.fit_ = fitted.fit_
        return self

    def _require_fitted(self) -> None:
        if self.partition_ is None or self.transform_ is None or self.fit_ is None:
            raise RuntimeError("SUBGClassifier has not been fitted.")

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.fit_.estimator.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.fit_.estimator.predict(self.transform_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.fit_.estimator.predict_proba(self.transform_.transform(X))

    def transform_diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        return dict(self.transform_.diagnostics())

    def fit_diagnostics(self, groups: np.ndarray) -> dict[str, Any]:
        self._require_fitted()
        return {
            "fit_split": "dfr_train_balanced",
            "uses_group_labels": True,
            "class_balance_method": "subgroup_sampling",
            "transform_fit_split": "dfr_train_balanced",
            "transform_fit_observations": int(len(self.partition_.balanced_train_indices)),
            "partition": self.partition_.diagnostics(groups),
            **self.fit_.diagnostics(),
        }


class DFRAveragedClassifier(SUBGClassifier):
    """A single linear head obtained by averaging in original coordinates."""

    def __init__(self, members: list[SUBGClassifier]):
        first = members[0]
        super().__init__(ridge_lambda=first.ridge_lambda,
                         transform_name=first.transform_name,
                         random_state=first.random_state)
        self.members_ = members
        self.partition_ = first.partition_
        self.fit_ = copy.deepcopy(first.fit_)
        heads = [member.transform_.inverse_head(
            member.fit_.estimator.coef_, member.fit_.estimator.intercept_
        ) for member in members]
        self.fit_.estimator.coef_ = np.mean([head[0] for head in heads], axis=0)
        self.fit_.estimator.intercept_ = np.mean([head[1] for head in heads], axis=0)
        self.fit_.estimator.n_features_in_ = self.fit_.estimator.coef_.shape[1]
        self.transform_ = first.transform_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit_.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fit_.estimator.predict_proba(X)

    def transform_diagnostics(self) -> dict[str, Any]:
        return {**self.members_[0].transform_diagnostics(),
                "subsample_count": len(self.members_),
                "member_transforms": [m.transform_diagnostics() for m in self.members_]}

    def fit_diagnostics(self, groups: np.ndarray) -> dict[str, Any]:
        members = [m.fit_diagnostics(groups) for m in self.members_]
        return {**members[0], "subsample_count": len(members),
                "aggregation": "mean_original_coordinate_coefficients_and_intercepts",
                "n_iter": [n for m in members for n in m["n_iter"]],
                "members": members}


def fit_dfr_averaged_path(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
    partition: DFRPartition, ridge_lambdas: Iterable[float],
    transform_name: str, subsample_count: int = 10,
    verbose: bool = False,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    erm_head: Any | None = None,
    warm_start: bool = False,
) -> list[DFRAveragedClassifier]:
    """Fit matched balanced draws once per transform, reusing them over ridge."""
    if isinstance(subsample_count, bool) or int(subsample_count) != subsample_count or subsample_count < 1:
        raise ValueError("DFR subsample_count must be a positive integer.")
    ridges = tuple(float(r) for r in ridge_lambdas)
    if not ridges or len(set(ridges)) != len(ridges):
        raise ValueError("DFR requires a nonempty ridge grid without duplicates.")
    for ridge in ridges:
        LogisticConfig(ridge_lambda=ridge)
    if not isinstance(warm_start, bool):
        raise TypeError("warm_start must be boolean.")
    if warm_start and erm_head is None:
        raise ValueError("DFR warm_start=True requires a fitted ERM head.")
    pool = partition.fit_pool_indices
    group_ids, counts = np.unique(groups[pool], return_counts=True)
    count = int(counts.min())
    rng = np.random.default_rng(np.random.SeedSequence([partition.random_state, 104729]))
    paths = [[] for _ in ridges]
    for draw in range(int(subsample_count)):
        if verbose:
            print(f"  DFR {transform_name}: fitting subsample {draw + 1}/{subsample_count} "
                  f"({len(ridges)} ridge values)", flush=True)
        indices = (partition.balanced_train_indices if draw == 0 else
                   rng.permutation(np.concatenate([
                       rng.choice(pool[groups[pool] == g], size=count, replace=False)
                       for g in group_ids])))
        if not np.array_equal(np.unique(y[indices]), np.unique(y)):
            raise ValueError("Every DFR balanced subsample must contain every class.")
        transform = make_transform(
            transform_name, whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            class_weighted=whitening_class_weighted,
        )
        transform = (
            transform.fit(X[indices], y[indices])
            if whitening_class_weighted
            else transform.fit(X[indices])
        )
        transformed = transform.transform(X[indices])
        reference_head = (
            _reference_head_in_transform(erm_head, transform)
            if warm_start
            else None
        )
        member_partition = DFRPartition(
            random_state=partition.random_state, observations=partition.observations,
            fit_pool_indices=pool, balanced_train_indices=indices,
            selection_indices=partition.selection_indices)
        for path, ridge in zip(paths, ridges):
            logistic_config = LogisticConfig(
                ridge_lambda=ridge, class_balanced=False
            )
            fit = (
                fit_logistic_from_reference(
                    transformed,
                    y[indices],
                    logistic_config,
                    reference_head,
                    proximity_penalty=False,
                )
                if warm_start
                else fit_logistic(transformed, y[indices], logistic_config)
            )
            if not fit.converged:
                raise RuntimeError(f"DFR subsample {draw} ridge={ridge:g} did not converge: " +
                                   "; ".join(fit.convergence_messages))
            member = SUBGClassifier(ridge_lambda=ridge, transform_name=transform_name,
                                    random_state=partition.random_state)
            member.partition_, member.transform_, member.fit_ = member_partition, transform, fit
            path.append(member)
    return [DFRAveragedClassifier(path) for path in paths]


def dfr_method_name(transform_name: str) -> str:
    """Prefix a canonical fitted transform name with the DFR family."""
    normalized = str(transform_name).strip().lower().replace("-", "_")
    return f"dfr_{normalized}"


def _bundle_identity(bundle: EmbeddingBundle) -> dict[str, Any]:
    return {
        "path": str(bundle.root),
        "fingerprint": bundle.fingerprint,
        "dataset": bundle.manifest.get("dataset"),
        "representation": bundle.manifest.get("representation"),
        "representation_seed": bundle.manifest.get("representation_seed"),
    }


def identify_neurotune_coordinates(
    selector_head: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify the union of NeuroTune coordinates on raw representations.

    For each target class, the score of a coordinate is the median absolute
    activation among incorrectly classified examples minus the corresponding
    median among correctly classified examples. A coordinate is removed when
    this score is strictly greater than ``threshold`` for any target class.
    """
    X, y = _validate_method_arrays(X, y, method="NeuroTune")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("NeuroTune threshold must be finite.")
    if not hasattr(selector_head, "classes_") or not hasattr(
        selector_head, "predict"
    ):
        raise ValueError("NeuroTune selector_head must be a fitted classifier.")
    classes = np.asarray(selector_head.classes_)
    if classes.ndim != 1 or classes.size < 2 or not np.isin(y, classes).all():
        raise ValueError("NeuroTune labels must belong to selector_head.classes_.")
    if int(getattr(selector_head, "n_features_in_", -1)) != X.shape[1]:
        raise ValueError(
            "NeuroTune selector-head feature dimension differs from the "
            "identification representations."
        )

    predictions = np.asarray(selector_head.predict(X)).reshape(-1)
    if predictions.shape != y.shape:
        raise ValueError("NeuroTune selector predictions do not align with labels.")
    deltas, per_class = _neurotune_activation_deltas(
        X, y, predictions, classes=classes, require_both_outcomes=True
    )
    removed = np.zeros(X.shape[1], dtype=bool)
    for label, delta in zip(classes, deltas):
        label_key = str(label.item() if hasattr(label, "item") else label)
        selected = np.asarray(delta > threshold, dtype=bool)
        removed |= selected
        per_class[label_key].update({
            "coordinates_above_threshold": int(selected.sum()),
            "delta_min": float(delta.min()),
            "delta_median": float(np.median(delta)),
            "delta_max": float(delta.max()),
        })

    removed_indices = np.flatnonzero(removed).astype(np.int64, copy=False)
    retained_indices = np.flatnonzero(~removed).astype(np.int64, copy=False)
    if retained_indices.size == 0:
        raise ValueError(
            "NeuroTune removed every representation coordinate; increase the "
            "threshold before fitting the final classifier."
        )
    return retained_indices, {
        "paper_method": "NeuroTune",
        "identification_split": "val",
        "selector_source": "saved_trained_last_layer",
        "selector_transform": "identity_raw_embeddings",
        "threshold": threshold,
        "threshold_rule": "delta_strictly_greater_than_threshold",
        "activation_values": "absolute_raw_coordinates",
        "union_across_classes": True,
        "original_coordinates": int(X.shape[1]),
        "removed_coordinates": int(removed_indices.size),
        "removed_fraction": float(removed_indices.size / X.shape[1]),
        "retained_coordinates": int(retained_indices.size),
        "removed_coordinate_indices": removed_indices.tolist(),
        "removed_coordinate_sha256": sha256(
            removed_indices.tobytes(order="C")
        ).hexdigest(),
        "per_class": per_class,
        "uses_group_labels": False,
    }


def _neurotune_activation_deltas(
    X: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    classes: np.ndarray,
    require_both_outcomes: bool,
) -> tuple[list[np.ndarray], dict[str, dict[str, Any]]]:
    """Return paper-defined per-class activation-magnitude differences."""
    predictions = np.asarray(predictions).reshape(-1)
    if predictions.shape != y.shape:
        raise ValueError("NeuroTune predictions do not align with labels.")
    magnitudes = np.abs(X)
    deltas: list[np.ndarray] = []
    per_class: dict[str, dict[str, Any]] = {}
    for label in classes:
        label_mask = y == label
        correct = label_mask & (predictions == y)
        incorrect = label_mask & (predictions != y)
        correct_count = int(correct.sum())
        incorrect_count = int(incorrect.sum())
        label_key = str(label.item() if hasattr(label, "item") else label)
        if require_both_outcomes and (correct_count == 0 or incorrect_count == 0):
            missing = "correct" if correct_count == 0 else "incorrect"
            raise ValueError(
                "NeuroTune identification requires at least one correct and "
                f"one incorrect validation prediction per class; class "
                f"{label_key!r} has no {missing} predictions."
            )
        correct_median = (
            np.median(magnitudes[correct], axis=0)
            if correct_count
            else np.zeros(X.shape[1], dtype=X.dtype)
        )
        incorrect_median = (
            np.median(magnitudes[incorrect], axis=0)
            if incorrect_count
            else correct_median
        )
        delta = np.asarray(incorrect_median - correct_median)
        deltas.append(delta)
        per_class[label_key] = {
            "correct_predictions": correct_count,
            "incorrect_predictions": incorrect_count,
        }
    return deltas, per_class


def _neurotune_sfit(
    estimator: Any,
    transform: Any,
    retained: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    transformed_X: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Calculate the paper's label-only spuriousness fitness score."""
    predictions = estimator.predict(
        transform.transform(X[:, retained]) if transformed_X is None else transformed_X
    )
    classes = np.asarray(estimator.classes_)
    deltas, per_class = _neurotune_activation_deltas(
        X,
        y,
        predictions,
        classes=classes,
        require_both_outcomes=False,
    )
    class_scores: dict[str, float] = {}
    for label, delta in zip(classes, deltas):
        label_key = str(label.item() if hasattr(label, "item") else label)
        class_score = float(np.abs(delta).sum(dtype=np.float64))
        class_scores[label_key] = class_score
        per_class[label_key]["absolute_delta_sum"] = class_score
    score = float(sum(class_scores.values()))
    return score, {
        "name": "sfit",
        "definition": "sum_classes_and_coordinates_absolute_spuriousness_score",
        "value": score,
        "per_class": per_class,
        "uses_group_labels": False,
    }


def neurotune_method_name(transform_name: str) -> str:
    """Prefix a canonical fitted transform name with the NeuroTune family."""
    normalized = str(transform_name).strip().lower().replace("-", "_")
    return f"neurotune_{normalized}"


def ho_erm_method_name(transform_name: str) -> str:
    """Prefix a canonical fitted transform name with the HO_ERM family."""
    normalized = str(transform_name).strip().lower().replace("-", "_")
    return f"ho_erm_{normalized}"


def _ho_erm_validation_partition(
    y: np.ndarray, *, random_state: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return reproducible label-stratified fitting and selection halves."""
    y = np.asarray(y).reshape(-1)
    seed = _validated_seed(random_state)
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        raise ValueError("HO_ERM requires at least two validation classes.")
    if np.any(counts < 2):
        raise ValueError(
            "HO_ERM needs at least two validation examples per class to form "
            "label-stratified fitting and selection halves."
        )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed
    )
    indices = np.arange(y.size, dtype=np.int64)
    fit_indices, selection_indices = next(splitter.split(indices, y))
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    selection_indices = np.asarray(selection_indices, dtype=np.int64)
    return fit_indices, selection_indices, {
        "source_split": "val",
        "random_state": seed,
        "splitter": "StratifiedShuffleSplit",
        "stratification": "label",
        "test_size": 0.5,
        "observations": int(y.size),
        "fit_observations": int(fit_indices.size),
        "selection_observations": int(selection_indices.size),
        "fit_index_sha256": _index_hash(fit_indices),
        "selection_index_sha256": _index_hash(selection_indices),
        "uses_group_labels": False,
    }


def sweep_ho_erm_bundle(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    split_seed: int = 0,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    erm_head: Any | None = None,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fit class-balanced ERM on one validation half and tune on the other."""
    requested_transforms = tuple(str(name) for name in transforms)
    if not requested_transforms:
        raise ValueError("HO_ERM requires at least one transform.")
    ridge_values = tuple(float(value) for value in ridge_lambdas)
    if not ridge_values or len(set(ridge_values)) != len(ridge_values):
        raise ValueError("HO_ERM requires unique, non-empty ridge_lambdas.")
    for value in ridge_values:
        LogisticConfig(ridge_lambda=value)
    if not isinstance(warm_start, bool):
        raise TypeError("warm_start must be boolean.")
    if warm_start and erm_head is None:
        raise ValueError("HO_ERM warm_start=True requires a fitted ERM head.")

    validation = bundle.split("val")
    fit_indices, selection_indices, partition = _ho_erm_validation_partition(
        validation.y, random_state=split_seed
    )
    fit_y = validation.y[fit_indices]
    selection_y = validation.y[selection_indices]
    labels, counts = np.unique(selection_y, return_counts=True)

    runs: list[dict[str, Any]] = []
    method_names: set[str] = set()
    for requested_name in requested_transforms:
        transform = make_transform(
            requested_name,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            class_weighted=whitening_class_weighted,
        )
        fit_X = validation.X[fit_indices]
        transform = (
            transform.fit(fit_X, fit_y)
            if whitening_class_weighted
            else transform.fit(fit_X)
        )
        transform_report = transform.diagnostics()
        method = ho_erm_method_name(transform_report["name"])
        if method in method_names:
            raise ValueError(
                f"HO_ERM transform {requested_name!r} duplicates method "
                f"{method!r}."
            )
        method_names.add(method)
        transformed = {
            split_name: transform.transform(bundle.split(split_name).X)
            for split_name in ("train", "val", "test")
        }
        transformed_fit = transformed["val"][fit_indices]
        transformed_selection = transformed["val"][selection_indices]
        reference_head = (
            _reference_head_in_transform(erm_head, transform)
            if warm_start
            else None
        )

        for ridge_lambda in ridge_values:
            logistic_config = LogisticConfig(
                ridge_lambda=ridge_lambda,
                class_balanced=True,
            )
            fit = (
                fit_logistic_from_reference(
                    transformed_fit,
                    fit_y,
                    logistic_config,
                    reference_head,
                    proximity_penalty=False,
                )
                if warm_start
                else fit_logistic(transformed_fit, fit_y, logistic_config)
            )
            if not fit.converged:
                raise RuntimeError(
                    f"HO_ERM {requested_name} ridge={ridge_lambda:g} did not "
                    "converge: " + "; ".join(fit.convergence_messages)
                )
            metrics, standard_errors = selection_metric_statistics(
                fit.estimator, transformed_selection, selection_y
            )
            run: dict[str, Any] = {
                "status": "complete",
                "bundle": _bundle_identity(bundle),
                "method": method,
                "method_family": "ho_erm",
                "fitting_method": "HO_ERM",
                "requested_transform": requested_name,
                "ridge_lambda": ridge_lambda,
                "C": 1.0 / ridge_lambda,
                "sklearn_C": fit.sklearn_C,
                "class_balanced": True,
                "class_balance_method": fit.class_balance_method,
                "transform": transform_report,
                "fit": {
                    "fit_split": "val_subset1",
                    "observations": int(fit_indices.size),
                    "uses_group_labels": False,
                    "transform_fit_split": "val_subset1",
                    "transform_fit_observations": int(fit_indices.size),
                    **fit.diagnostics(),
                },
                "protocol": {
                    "ho_erm_protocol_version": 1,
                    "validation_partition": partition,
                    "fit_split": "val_subset1",
                    "selection_split": "val_subset2",
                    "test_split": "test",
                    "original_train_split_used": False,
                    "group_labels_used_for_fitting": False,
                    "group_labels_used_for_model_selection": False,
                    "test_group_labels_used_for_model_selection": False,
                    "hyperparameter_selection": "class_balanced_accuracy",
                    "oracle": False,
                },
                "selection": {
                    "split": "val_subset2",
                    "observations": int(selection_indices.size),
                    "uses_group_labels": False,
                    "oracle": False,
                    "label_counts": {
                        str(label.item() if hasattr(label, "item") else label): int(count)
                        for label, count in zip(labels, counts)
                    },
                    "metrics": metrics,
                    "standard_errors": standard_errors,
                    "reason": (
                        "Select ridge by exact class-balanced accuracy on the "
                        "untouched validation half."
                    ),
                },
                "splits": {
                    split_name: evaluation_metrics(
                        fit.estimator,
                        transformed[split_name],
                        bundle.split(split_name).y,
                        bundle.split(split_name).g,
                    )
                    for split_name in ("train", "val", "test")
                },
            }
            runs.append(run)
            if progress_callback is not None:
                progress_callback(run)
    return runs


def _neurotune_validation_partition(
    y: np.ndarray, *, random_state: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return label-stratified identification and selection validation halves."""
    y = np.asarray(y).reshape(-1)
    seed = _validated_seed(random_state)
    labels, counts = np.unique(y, return_counts=True)
    if np.any(counts < 4):
        raise ValueError(
            "NeuroTune needs at least four validation examples per class to "
            "form label-stratified identification and selection halves."
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    indices = np.arange(y.size, dtype=np.int64)
    identification, selection = next(splitter.split(indices, y))
    identification = np.asarray(identification, dtype=np.int64)
    selection = np.asarray(selection, dtype=np.int64)
    return identification, selection, {
        "source_split": "val",
        "random_state": seed,
        "splitter": "StratifiedShuffleSplit",
        "stratification": "label",
        "test_size": 0.5,
        "identification_observations": int(identification.size),
        "selection_observations": int(selection.size),
        "identification_index_sha256": _index_hash(identification),
        "selection_index_sha256": _index_hash(selection),
        "uses_group_labels": False,
    }


def _sweep_neurotune_bundle_single_threshold(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    selector_head: SavedLinearHead,
    threshold: float = 0.0,
    class_balanced: bool = False,
    split_seed: int = 0,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Identify on V1, fit on disjoint V2, and select by balanced accuracy."""
    requested_transforms = tuple(str(name) for name in transforms)
    if not requested_transforms:
        raise ValueError("NeuroTune requires at least one transform.")
    ridge_values = tuple(float(value) for value in ridge_lambdas)
    if not ridge_values or len(set(ridge_values)) != len(ridge_values):
        raise ValueError("NeuroTune requires unique, non-empty ridge_lambdas.")
    for value in ridge_values:
        LogisticConfig(ridge_lambda=value)
    if not isinstance(warm_start, bool):
        raise TypeError("warm_start must be boolean.")

    train = bundle.split("train")
    validation = bundle.split("val")
    if not np.array_equal(np.asarray(selector_head.classes_), np.unique(train.y)):
        raise ValueError("NeuroTune saved-head classes differ from training labels.")
    identification_indices, tuning_indices, partition = _neurotune_validation_partition(
        validation.y, random_state=split_seed
    )
    retained, identification = identify_neurotune_coordinates(
        selector_head,
        validation.X[identification_indices],
        validation.y[identification_indices],
        threshold=threshold,
    )
    identification["identification_split"] = "val_subset1"
    labels, counts = np.unique(
        validation.y[identification_indices], return_counts=True
    )

    runs: list[dict[str, Any]] = []
    method_names: set[str] = set()
    for requested_name in requested_transforms:
        transform = make_transform(
            requested_name,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            class_weighted=whitening_class_weighted,
        )
        tuning_X = validation.X[tuning_indices][:, retained]
        transform = (
            transform.fit(tuning_X, validation.y[tuning_indices])
            if whitening_class_weighted
            else transform.fit(tuning_X)
        )
        transform_report = transform.diagnostics()
        method = neurotune_method_name(transform_report["name"])
        if method in method_names:
            raise ValueError(
                f"NeuroTune transform {requested_name!r} duplicates method "
                f"{method!r}."
            )
        method_names.add(method)
        reference_head = (
            _reference_head_in_transform(
                selector_head, transform, feature_indices=retained
            )
            if warm_start
            else None
        )
        # These inputs depend on the mask and transform, never on ridge.
        transformed_tuning = transform.transform(tuning_X)
        identification_X = validation.X[identification_indices]
        transformed_identification = transform.transform(identification_X[:, retained])
        transformed = {
            split_name: transform.transform(bundle.split(split_name).X[:, retained])
            for split_name in ("train", "val", "test")
        }
        for ridge_lambda in ridge_values:
            logistic_config = LogisticConfig(
                ridge_lambda=ridge_lambda,
                class_balanced=bool(class_balanced),
            )
            fit = (
                fit_logistic_from_reference(
                    transformed_tuning,
                    validation.y[tuning_indices],
                    logistic_config,
                    reference_head,
                    proximity_penalty=False,
                )
                if warm_start
                else fit_logistic(
                    transformed_tuning,
                    validation.y[tuning_indices],
                    logistic_config,
                )
            )
            if not fit.converged:
                raise RuntimeError(
                    f"NeuroTune {requested_name} ridge={ridge_lambda:g} did "
                    "not converge: " + "; ".join(fit.convergence_messages)
                )
            metrics, standard_errors = selection_metric_statistics(
                fit.estimator,
                transformed_identification,
                validation.y[identification_indices],
            )
            sfit, sfit_report = _neurotune_sfit(
                fit.estimator,
                transform,
                retained,
                identification_X,
                validation.y[identification_indices],
                transformed_X=transformed_identification,
            )
            metrics["sfit"] = sfit
            run: dict[str, Any] = {
                "status": "complete",
                "bundle": _bundle_identity(bundle),
                "method": method,
                "method_family": "neurotune",
                "fitting_method": "NEUROTUNE",
                "requested_transform": requested_name,
                "ridge_lambda": ridge_lambda,
                "C": 1.0 / ridge_lambda,
                "sklearn_C": fit.sklearn_C,
                "class_balanced": bool(class_balanced),
                "class_balance_method": fit.class_balance_method,
                "transform": transform_report,
                "fit": {
                    "fit_split": "val_subset2",
                    "observations": int(tuning_indices.size),
                    "uses_group_labels": False,
                    "transform_fit_split": "val_subset2",
                    "transform_fit_observations": int(tuning_indices.size),
                    "input_coordinates_after_removal": int(retained.size),
                    "selector_head": selector_head.diagnostics(),
                    **fit.diagnostics(),
                },
                "neurotune": identification,
                "sfit": sfit_report,
                "protocol": {
                    "neurotune_protocol_version": 5,
                    "validation_partition": partition,
                    "identification_split": "val_subset1",
                    "fit_split": "val_subset2",
                    "selection_split": "val_subset1",
                    "test_split": "test",
                    "identification_and_tuning_are_disjoint": True,
                    "mask_fixed_after_identification": True,
                    "operation_order": "remove_then_transform_then_fit",
                    "selector_source": "saved_trained_last_layer",
                    "selector_transform": "identity_raw_embeddings",
                    "group_labels_used_for_identification": False,
                    "group_labels_used_for_fitting": False,
                    "group_labels_used_for_model_selection": False,
                    "test_group_labels_used_for_model_selection": False,
                    "hyperparameter_selection": "class_balanced_accuracy",
                    "oracle": False,
                },
                "selection": {
                    "split": "val_subset1",
                    "observations": int(identification_indices.size),
                    "uses_group_labels": False,
                    "oracle": False,
                    "label_counts": {
                        str(label.item() if hasattr(label, "item") else label): int(count)
                        for label, count in zip(labels, counts)
                    },
                    "metrics": metrics,
                    "standard_errors": standard_errors,
                    "reason": (
                        "Select threshold and ridge by exact class-balanced "
                        "accuracy on the identification half; group labels "
                        "are unused."
                    ),
                },
                "splits": {
                    split_name: evaluation_metrics(
                        fit.estimator,
                        transformed[split_name],
                        bundle.split(split_name).y,
                        bundle.split(split_name).g,
                    )
                    for split_name in ("train", "val", "test")
                },
            }
            runs.append(run)
            if progress_callback is not None:
                progress_callback(run)
    return runs


def sweep_neurotune_bundle(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    selector_head: SavedLinearHead,
    threshold: float | Iterable[float] = 0.0,
    class_balanced: bool = False,
    split_seed: int = 0,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Cross thresholds with ridge for disjoint-half NeuroTune selection."""
    raw_thresholds = (threshold,) if np.isscalar(threshold) else tuple(threshold)
    thresholds = tuple(float(value) for value in raw_thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("NeuroTune thresholds must be a unique, non-empty grid.")
    if not np.isfinite(thresholds).all():
        raise ValueError("NeuroTune thresholds must be finite.")
    runs: list[dict[str, Any]] = []
    for value in thresholds:
        try:
            runs.extend(_sweep_neurotune_bundle_single_threshold(
                bundle,
                transforms=transforms,
                ridge_lambdas=ridge_lambdas,
                selector_head=selector_head,
                threshold=value,
                class_balanced=class_balanced,
                split_seed=split_seed,
                whitening_relative_tolerance=whitening_relative_tolerance,
                whitening_estimator=whitening_estimator,
                whitening_class_weighted=whitening_class_weighted,
                warm_start=warm_start,
                progress_callback=progress_callback,
            ))
        except ValueError as exc:
            if "removed every representation coordinate" not in str(exc):
                raise
            raise ValueError(
                f"NeuroTune threshold {value:g} removed every representation "
                "coordinate. The threshold grid cannot be pruned silently; "
                "increase or replace this threshold."
            ) from exc
    return runs


def valid_neurotune_thresholds(
    bundle: EmbeddingBundle,
    *,
    selector_head: SavedLinearHead,
    thresholds: Iterable[float],
    split_seed: int,
) -> tuple[float, ...]:
    """Return thresholds that leave at least one V1 representation coordinate."""
    validation = bundle.split("val")
    indices, _, _ = _neurotune_validation_partition(
        validation.y, random_state=split_seed
    )
    valid: list[float] = []
    for raw_value in thresholds:
        value = float(raw_value)
        try:
            identify_neurotune_coordinates(
                selector_head, validation.X[indices], validation.y[indices],
                threshold=value,
            )
        except ValueError as exc:
            if "removed every representation coordinate" not in str(exc):
                raise
            continue
        valid.append(value)
    return tuple(valid)


def sweep_afr_bundle(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    score_head: SavedLinearHead,
    afr_gammas: Iterable[float] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0),
    random_state: int = 0,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fit AFR gamma/ridge candidates using disjoint adaptation and selection halves."""
    requested_transforms = tuple(str(name) for name in transforms)
    if not requested_transforms:
        raise ValueError("AFR requires at least one transform.")
    ridge_values = tuple(float(value) for value in ridge_lambdas)
    if not ridge_values or len(set(ridge_values)) != len(ridge_values):
        raise ValueError("AFR requires unique, non-empty ridge_lambdas.")
    for value in ridge_values:
        LogisticConfig(ridge_lambda=value)
    gamma_values = tuple(float(value) for value in afr_gammas)
    if not gamma_values or len(set(gamma_values)) != len(gamma_values):
        raise ValueError("AFR requires a unique, non-empty gamma grid.")
    if any(not np.isfinite(value) or value < 0 for value in gamma_values):
        raise ValueError("AFR gamma values must be finite and nonnegative.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("AFR random_state must be an integer.")
    random_state = int(random_state)
    if random_state < 0:
        raise ValueError("AFR random_state must be nonnegative.")
    if not isinstance(warm_start, bool):
        raise TypeError("warm_start must be boolean.")

    train = bundle.split("train")
    validation = bundle.split("val")
    if not np.array_equal(np.asarray(score_head.classes_), np.unique(train.y)):
        raise ValueError("AFR saved-head classes differ from bundle training labels.")
    if score_head.n_features_in_ != train.X.shape[1]:
        raise ValueError("AFR saved-head feature dimension differs from the bundle.")
    try:
        adaptation_indices, selection_indices = next(StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=random_state,
        ).split(np.zeros((validation.y.size, 1)), validation.y))
    except ValueError as exc:
        raise ValueError(
            "AFR requires a validation set large enough for a target-stratified "
            "adaptation/selection split containing every class."
        ) from exc
    for indices in (adaptation_indices, selection_indices):
        if not np.array_equal(np.unique(validation.y[indices]), np.unique(train.y)):
            raise ValueError("Each AFR validation half must contain every training class.")
    partition = {
        "random_state": random_state,
        "stratification": "target_label",
        "validation_observations": int(validation.y.size),
        "adaptation_observations": int(adaptation_indices.size),
        "selection_observations": int(selection_indices.size),
        "adaptation_index_sha256": sha256(
            np.asarray(adaptation_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "selection_index_sha256": sha256(
            np.asarray(selection_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    weights_by_gamma = {
        gamma: compute_afr_weights(
            score_head,
            validation.X[adaptation_indices],
            validation.y[adaptation_indices],
            gamma,
        )
        for gamma in gamma_values
    }

    runs: list[dict[str, Any]] = []
    method_names: set[str] = set()
    for requested_name in requested_transforms:
        transform = make_transform(
            requested_name,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            class_weighted=whitening_class_weighted,
        )
        adaptation_X = validation.X[adaptation_indices]
        transform = (
            transform.fit(adaptation_X, validation.y[adaptation_indices])
            if whitening_class_weighted else transform.fit(adaptation_X)
        )
        transform_report = transform.diagnostics()
        method = afr_method_name(transform_report["name"])
        if method in method_names:
            raise ValueError(f"AFR transform {requested_name!r} duplicates method {method!r}.")
        method_names.add(method)
        transformed = {
            split_name: transform.transform(bundle.split(split_name).X)
            for split_name in ("train", "val", "test")
        }
        transformed_adaptation = transformed["val"][adaptation_indices]
        transformed_selection = transformed["val"][selection_indices]
        reference_head = _reference_head_in_transform(score_head, transform)
        for gamma in gamma_values:
            weights, weight_report = weights_by_gamma[gamma]
            for ridge_lambda in ridge_values:
                fit = fit_logistic_from_reference(
                    transformed_adaptation,
                    validation.y[adaptation_indices],
                    LogisticConfig(
                        ridge_lambda=ridge_lambda,
                        class_balanced=False,
                    ),
                    reference_head,
                    proximity_penalty=True,
                    sample_weight=weights,
                    sample_weight_method="afr_inverse_class_frequency_confidence_weight",
                )
                metrics, standard_errors = selection_metric_statistics(
                    fit.estimator,
                    transformed_selection,
                    validation.y[selection_indices],
                )
                selection_group_metrics = evaluation_metrics(
                    fit.estimator,
                    transformed_selection,
                    validation.y[selection_indices],
                    validation.g[selection_indices],
                )
                metrics["worst_group_accuracy"] = selection_group_metrics[
                    "worst_group_accuracy"
                ]
                labels, counts = np.unique(
                    validation.y[selection_indices], return_counts=True
                )
                run: dict[str, Any] = {
                    "status": "complete",
                    "bundle": _bundle_identity(bundle),
                    "method": method,
                    "method_family": "afr",
                    "fitting_method": "AFR",
                    "requested_transform": requested_name,
                    "ridge_lambda": ridge_lambda,
                    "afr_gamma": gamma,
                    "C": 1.0 / ridge_lambda,
                    "sklearn_C": fit.sklearn_C,
                    "class_balanced": True,
                    "class_balance_method": "afr_inverse_class_frequency_confidence_weight",
                    "transform": transform_report,
                    "fit": {
                        "fit_split": "val_afr_adaptation_half",
                        "observations": int(adaptation_indices.size),
                        "uses_group_labels": False,
                        "uses_sample_weight": True,
                        "transform_fit_split": "val_afr_adaptation_half",
                        "score_head": score_head.diagnostics(),
                        **fit.diagnostics(),
                    },
                    "afr": {**weight_report, "partition": partition},
                    "protocol": {
                        "afr_protocol_version": 3,
                        "score_head_source": "saved_trained_last_layer",
                        "score_head_transform": "identity_raw_embeddings",
                        "fit_split": "val_afr_adaptation_half",
                        "selection_split": "val_afr_selection_half",
                        "test_split": "test",
                        "transform_fit_split": "val_afr_adaptation_half",
                        "fixed_sample_weights": True,
                        "proximity_penalty": True,
                        "proximity_center": "original_erm_head_in_transform_coordinates",
                        "optimizer": "converged_offset_lbfgs_logistic_regression",
                        "group_labels_used_for_fitting": False,
                        "group_labels_used_for_model_selection": True,
                        "test_group_labels_used_for_model_selection": False,
                        "oracle": True,
                    },
                    "selection": {
                        "split": "val",
                        "subset": "afr_selection_half",
                        "observations": int(selection_indices.size),
                        "uses_group_labels": True,
                        "oracle": True,
                        "label_counts": {
                            str(label.item() if hasattr(label, "item") else label): int(count)
                            for label, count in zip(labels, counts)
                        },
                        "metrics": metrics,
                        "standard_errors": standard_errors,
                    },
                    "splits": {
                        split_name: selection_group_metrics
                        if split_name == "val" else evaluation_metrics(
                            fit.estimator,
                            transformed[split_name][selection_indices]
                            if split_name == "val" else transformed[split_name],
                            bundle.split(split_name).y[selection_indices]
                            if split_name == "val" else bundle.split(split_name).y,
                            bundle.split(split_name).g[selection_indices]
                            if split_name == "val" else bundle.split(split_name).g,
                        )
                        for split_name in ("train", "val", "test")
                    },
                }
                runs.append(run)
                if progress_callback is not None:
                    progress_callback(run)
    return runs


def sweep_dfr_bundle(
    bundle: EmbeddingBundle,
    *,
    transforms: Iterable[str],
    ridge_lambdas: Iterable[float],
    split_seed: int,
    subsample_count: int = 10,
    whitening_relative_tolerance: float | None = None,
    whitening_estimator: str = "empirical",
    whitening_class_weighted: bool = False,
    erm_head: Any | None = None,
    warm_start: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fit DFR transform/ridge candidates for one embedding bundle."""
    requested_transforms = tuple(str(name) for name in transforms)
    if not requested_transforms:
        raise ValueError("DFR requires at least one transform.")
    ridge_values = tuple(float(value) for value in ridge_lambdas)
    if not ridge_values:
        raise ValueError("DFR requires at least one ridge_lambda.")
    if len(set(ridge_values)) != len(ridge_values):
        raise ValueError("DFR ridge_lambdas contains duplicates.")

    validation = bundle.split("val")
    test = bundle.split("test")
    partition = make_dfr_partition(validation.g, random_state=split_seed)
    partition_report = partition.diagnostics(validation.g)
    balanced = partition.balanced_train_indices
    selection = partition.selection_indices
    runs: list[dict[str, Any]] = []
    method_names: set[str] = set()

    for requested_name in requested_transforms:
        classifiers = fit_dfr_averaged_path(
            validation.X,
            validation.y,
            validation.g,
            ridge_lambdas=ridge_values,
            transform_name=requested_name,
            subsample_count=subsample_count,
            verbose=progress_callback is not None,
            partition=partition,
            whitening_relative_tolerance=whitening_relative_tolerance,
            whitening_estimator=whitening_estimator,
            whitening_class_weighted=whitening_class_weighted,
            erm_head=erm_head,
            warm_start=warm_start,
        )
        transform_report = classifiers[0].transform_diagnostics()
        method = dfr_method_name(transform_report["name"])
        if method in method_names:
            raise ValueError(
                f"DFR transform {requested_name!r} duplicates method {method!r}."
            )
        method_names.add(method)

        for classifier in classifiers:
            split_reports = {
                "train": evaluation_metrics(
                    classifier,
                    validation.X[balanced],
                    validation.y[balanced],
                    validation.g[balanced],
                ),
                "val": evaluation_metrics(
                    classifier,
                    validation.X[selection],
                    validation.y[selection],
                    validation.g[selection],
                ),
                "test": evaluation_metrics(
                    classifier,
                    test.X,
                    test.y,
                    test.g,
                ),
            }
            fit_report = classifier.fit_diagnostics(validation.g)
            worst_validation_group = min(
                split_reports["val"]["per_group"].values(),
                key=lambda entry: float(entry["accuracy"]),
            )
            worst_group_count = int(worst_validation_group["count"])
            worst_group_accuracy = float(worst_validation_group["accuracy"])
            worst_group_standard_error = (
                float(
                    np.sqrt(
                        worst_group_accuracy
                        * (1.0 - worst_group_accuracy)
                        / (worst_group_count - 1)
                    )
                )
                if worst_group_count > 1
                else 0.0
            )
            run = {
                "status": "complete",
                "bundle": _bundle_identity(bundle),
                "method": method,
                "method_family": "dfr",
                "fitting_method": "DFR",
                "requested_transform": requested_name,
                "ridge_lambda": classifier.ridge_lambda,
                "C": 1.0 / classifier.ridge_lambda,
                "sklearn_C": classifier.fit_.sklearn_C,
                # This broad experiment flag keeps the subgroup-balanced DFR
                # methods in the same aggregate panel as the class-balanced
                # final-result comparisons.
                "class_balanced": True,
                "class_balance_method": "subgroup_sampling",
                "transform": transform_report,
                "fit": fit_report,
                "protocol": {
                    "fit_split": "dfr_train_balanced",
                    "dfr_protocol_version": 2,
                    "subsample_count": subsample_count,
                    "model_aggregation": "mean_logits",
                    "transform_fit_split": "dfr_train_balanced",
                    "selection_split": "dfr_val",
                    "test_split": "test",
                    "original_train_split_used": False,
                    "group_labels_used_for_fitting": True,
                    "group_labels_used_for_model_selection": True,
                    "test_group_labels_used_for_model_selection": False,
                    "oracle": True,
                },
                "dfr": partition_report,
                "selection": {
                    "split": "dfr_val",
                    "uses_group_labels": True,
                    "oracle": True,
                    "group_counts": partition_report[
                        "selection_group_counts"
                    ],
                    "metrics": {
                        "worst_group_accuracy": split_reports["val"][
                            "worst_group_accuracy"
                        ],
                    },
                    "standard_errors": {
                        "worst_group_accuracy": worst_group_standard_error,
                    },
                    "worst_group_accuracy": split_reports["val"][
                        "worst_group_accuracy"
                    ],
                },
                "splits": split_reports,
            }
            runs.append(run)
            if progress_callback is not None:
                progress_callback(run)
    return runs
