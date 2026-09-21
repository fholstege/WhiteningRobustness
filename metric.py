"""Classification metrics, with group labels confined to evaluation.

The fitting code never imports this module's group-aware calculations. Group
identifiers arrive only after a classifier has produced predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from numerics import METRIC_DTYPE


def _standard_error_of_mean(values: np.ndarray) -> float:
    """Return the finite-sample standard error of an empirical mean."""
    values = np.asarray(values, dtype=METRIC_DTYPE).reshape(-1)
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


def selection_metric_statistics(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return label-only validation metrics and sampling standard errors."""
    # This helper intentionally has no ``groups`` argument.
    y = np.asarray(y).reshape(-1)
    predictions = estimator.predict(X)
    probabilities = np.asarray(
        estimator.predict_proba(X), dtype=METRIC_DTYPE
    )
    classes = np.asarray(estimator.classes_)

    # Map each true label to its predict_proba column and compute an individual
    # negative log-likelihood. Averaging within y first and then across classes
    # gives every class equal mass, regardless of the validation class mixture.
    class_to_column = {value: index for index, value in enumerate(classes)}
    try:
        columns = np.asarray(
            [class_to_column[value] for value in y],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(
            "Validation labels include a class absent from the head."
        ) from exc
    true_probabilities = np.clip(
        probabilities[np.arange(y.size), columns],
        np.finfo(METRIC_DTYPE).eps,
        1.0,
    )
    observation_losses = -np.log(true_probabilities)
    class_losses = []
    class_accuracy_standard_errors = []
    class_loss_standard_errors = []
    for label in classes:
        mask = y == label
        if not np.any(mask):
            raise ValueError(
                "Balanced log loss requires every fitted class in validation."
            )
        class_losses.append(observation_losses[mask].mean())
        class_accuracy_standard_errors.append(
            _standard_error_of_mean(predictions[mask] == y[mask])
        )
        class_loss_standard_errors.append(
            _standard_error_of_mean(observation_losses[mask])
        )
    balanced_log_loss = float(np.mean(class_losses))
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": float(
            log_loss(y, probabilities, labels=classes)
        ),
        "balanced_log_loss": balanced_log_loss,
    }
    n_classes = len(classes)
    standard_errors = {
        "accuracy": _standard_error_of_mean(predictions == y),
        "balanced_accuracy": float(
            np.sqrt(np.sum(np.square(class_accuracy_standard_errors)))
            / n_classes
        ),
        "log_loss": _standard_error_of_mean(observation_losses),
        "balanced_log_loss": float(
            np.sqrt(np.sum(np.square(class_loss_standard_errors)))
            / n_classes
        ),
    }
    return metrics, standard_errors


def selection_metrics(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Return metrics allowed for label-only validation selection."""
    metrics, _ = selection_metric_statistics(
        estimator,
        X,
        y,
    )
    return metrics


def evaluation_metrics(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    dtype: Any = METRIC_DTYPE,
) -> dict[str, Any]:
    """Compute final metrics after predictions have been produced."""
    y = np.asarray(y).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if len(X) != y.size or y.size != groups.size:
        raise ValueError("X, y, and groups have incompatible lengths.")
    # Obtain predictions before looking at group membership.
    predictions = estimator.predict(X)
    probabilities = np.asarray(
        estimator.predict_proba(X), dtype=dtype
    )
    classes = np.asarray(estimator.classes_)
    # ``predict_proba`` columns follow estimator.classes_, which need not be
    # literal 0, 1, ..., K-1 labels.
    class_to_column = {value: index for index, value in enumerate(classes)}
    try:
        columns = np.asarray([class_to_column[value] for value in y], dtype=int)
    except KeyError as exc:
        raise ValueError("Evaluation labels include a class absent from the head.") from exc
    # Extract p(y_i | x_i) and clip only to avoid log(0).
    clipped = np.clip(
        probabilities[np.arange(y.size), columns],
        np.finfo(dtype).eps,
        1.0,
    )
    losses = -np.log(clipped)

    # First calculate each group separately. Equal-group and worst-group
    # summaries below are reductions over these group-level values.
    per_group: dict[str, dict[str, float | int]] = {}
    for group in np.unique(groups):
        mask = groups == group
        if not np.any(mask):
            raise ValueError(f"Evaluation group {group} is empty.")
        group_log_loss = float(losses[mask].mean())
        per_group[str(int(group))] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(y[mask], predictions[mask])),
            "log_loss": group_log_loss,
            "log_likelihood": -group_log_loss,
        }
    group_accuracies = [entry["accuracy"] for entry in per_group.values()]
    group_losses = [entry["log_loss"] for entry in per_group.values()]
    group_balanced_accuracy = float(np.mean(group_accuracies))
    worst_group_accuracy = float(np.min(group_accuracies))
    group_balanced_log_loss = float(np.mean(group_losses))
    worst_group_log_loss = float(np.max(group_losses))
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": float(
            log_loss(y, probabilities, labels=estimator.classes_)
        ),
        # "Group-balanced" means every observed group receives equal mass.
        "group_balanced_accuracy": group_balanced_accuracy,
        "worst_group_accuracy": worst_group_accuracy,
        "group_balanced_log_likelihood": -group_balanced_log_loss,
        "worst_group_log_likelihood": -worst_group_log_loss,
        # Retain loss-form and historical names for saved-result compatibility.
        "equal_group_accuracy": group_balanced_accuracy,
        "equal_group_log_loss": group_balanced_log_loss,
        "worst_group_log_loss": worst_group_log_loss,
        "per_group": per_group,
    }
