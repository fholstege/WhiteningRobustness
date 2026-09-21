#!/usr/bin/env python3
"""Select ridge penalties on validation and aggregate held-out results.

``hyper_param_sweep.py`` and ``comparison.py`` record every
method/transform-by-ridge result.
This file performs the logically separate model-selection step:

1. group runs by embedding bundle and transformation;
2. select method-specific hyperparameters using the prescribed validation criterion;
3. retrieve that run's test metrics;
4. report means and 90% t confidence intervals across embedding seeds;
5. optionally refit the selected heads and attach paired, within-group test
   bootstrap intervals for equal-group and worst-group accuracy relative to
   identity and standardize.

For every method, the report also stores explicit paired differences relative
to identity and, when available, standardize. Single-representation reports use
the paired test bootstrap; reports with multiple representation seeds use a
paired t interval across those seeds.

Different transformations may use different selection rules through
``SELECTION_RULES`` or repeated ``--selection-rule method=criterion`` options.
``balanced_log_loss`` averages log loss within each validation label and then
across labels, so it balances ``y`` rather than groups.
AFR and oracle DFR use held-out validation worst-group accuracy. NeuroTune uses
exact label-balanced accuracy by default; the paper's label-only SFit criterion
remains available as an optional selection rule. Other methods retain label-only
accuracy or loss selection criteria.
``class_balanced_accuracy`` is the exact label-balanced validation accuracy;
exact ties favor the stronger ridge penalty. ``two_stage`` first maximizes
that exact accuracy and then minimizes class-balanced log loss within exact
ties.

With ``--one-se``, a single-stage criterion instead forms an admissible set
within one standard error of the best validation candidate and selects the
largest ridge penalty in that set. Current sweeps save those validation
standard errors; older sweep JSON must be regenerated before using this rule.

With ``--one-mistake``, accuracy or class-balanced accuracy instead forms an
admissible set whose score is no more than one validation mistake below the
best candidate and selects the largest ridge penalty in that set. For
class-balanced accuracy, the tolerance is the largest score change from one
mistake in any class: ``1 / (n_classes * min_class_count)``.

Both fine-tuned representation seeds and frozen split seeds use the same
ordinary procedure: select once per seed and average the selected held-out
results across seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from data import EmbeddingBundle, SavedLinearHead
from evaluate import ROOT
from metric import evaluation_metrics
from methods import (
    SUBGClassifier, DFRPartition, fit_dfr_averaged_path, dfr_method_name, make_dfr_partition,
    _reference_head_in_transform,
    compute_afr_weights, identify_neurotune_coordinates,
    _ho_erm_validation_partition,
    ho_erm_method_name,
    _neurotune_validation_partition,
    neurotune_method_name,
)
from model import (
    LogisticConfig,
    fit_logistic,
    fit_logistic_from_reference,
)
from whitening import make_transform


STANDARD_AGGREGATION_MODE = "selected_test_per_seed"

PAIRED_DIFFERENCE_METRICS = (
    "group_balanced_accuracy",
    "worst_group_accuracy",
)


# =============================================================================
# Editable aggregation configuration
# =============================================================================

AGGREGATE_CONFIG = {
    # Add one or more sweep JSON files here, or pass --sweep on the command line.
    # comparison.py outputs (e.g. results/comparisons/WB_dino_vitb16_DFR.json)
    # work alone or alongside matching ERM sweeps. DFR variants automatically
    # select ridge by held-out worst-group accuracy, including nonlinear LW.
    "sweep_paths": [],
    # Exact class-balanced validation accuracy; this balances labels, not groups.
    "default_selection_rule": "class_balanced_accuracy",
    # Optional per-method overrides, for example:
    # {"identity": "accuracy", "whiten": "log_loss"}
    "selection_rules": {},
    # Optional subset of saved ridge values eligible for validation selection.
    # None uses the complete saved grid.
    "selection_ridge_lambdas": None,
    # Optional subset of saved AFR gamma values eligible for validation
    # selection. None uses the complete saved gamma grid.
    "selection_afr_gammas": None,
    # Optional subset of saved NeuroTune thresholds eligible for validation
    # selection. None uses the complete saved threshold grid.
    "selection_neurotune_thresholds": None,
    # Optional upper bound on ridge values eligible for validation selection.
    # Values equal to or above this threshold are excluded.
    "max_ridge": None,
    # Optional upper bound on AFR gamma values eligible for validation
    # selection. Values equal to or above this threshold are excluded.
    "max_gamma": None,
    # Exclude fitted heads whose recorded final solver attempt used exactly
    # one iteration. This affects aggregation only; sweep JSON is unchanged.
    "exclude_single_iteration": False,
    # Apply the one-standard-error rule to validation hyperparameter selection.
    "one_se": False,
    # Apply the one-mistake plateau to accuracy-based validation selection.
    "one_mistake": False,
    # After validation selection, refit the selected ordinary head and its
    # transform on train plus validation before the final test evaluation.
    "refit_train_val": None,  # Auto: refit validation-adaptation methods.
    # Refit independent representation seeds concurrently. Keeping each worker
    # to one BLAS thread prevents nested parallelism during eigendecompositions.
    "refit_workers": min(4, os.cpu_count() or 1),
    "refit_blas_threads": 1,
    "confidence": 0.90,
    # Optional paired, group-stratified test bootstrap for single-representation
    # experiments. This refits only the validation-selected heads and compares
    # every method with identity and standardize when standardize is present.
    "bootstrap": False,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 0,
    "output": None,
    # Existing aggregate JSON is protected unless explicitly replaced.
    "overwrite": False,
}


# Selection uses ``run["selection"]``. AFR and DFR explicitly mark their
# group-aware validation selection; other methods keep this object group-free.
SELECTION_CRITERIA = {
    "accuracy": "maximize",
    "class_balanced_accuracy": "maximize",
    "log_loss": "minimize",
    "balanced_log_loss": "minimize",
    "two_stage": "two_stage",
    "worst_group_accuracy": "maximize",
    "sfit": "maximize",
}

# =============================================================================
# Loading and validation
# =============================================================================

def _validated_criterion(criterion: str) -> str:
    """Validate one canonical selection criterion."""
    if criterion not in SELECTION_CRITERIA:
        choices = ", ".join(sorted(SELECTION_CRITERIA))
        raise ValueError(
            f"Unknown selection criterion {criterion!r}; choose from {choices}."
        )
    return criterion


def _is_dfr_method(method: str) -> bool:
    return method == "dfr" or method.startswith("dfr_")


def _is_afr_method(method: str) -> bool:
    return method == "afr" or method.startswith("afr_")


def _is_neurotune_method(method: str) -> bool:
    return method == "neurotune" or method.startswith("neurotune_")


def _is_ho_erm_method(method: str) -> bool:
    return method == "ho_erm" or method.startswith("ho_erm_")


def _comparison_reference_name(method: str, reference: str) -> str:
    """Return the matched preprocessing baseline within a method family."""
    if _is_dfr_method(method):
        return f"dfr_{reference}"
    if _is_neurotune_method(method):
        return f"neurotune_{reference}"
    if _is_ho_erm_method(method):
        return f"ho_erm_{reference}"
    if _is_afr_method(method):
        return f"afr_{reference}"
    return reference


def _is_oracle_dfr_run(run: dict[str, Any]) -> bool:
    selection = run.get("selection", {})
    protocol = run.get("protocol", {})
    return (
        run.get("method_family") == "dfr"
        and _is_dfr_method(str(run.get("method")))
        and selection.get("split") == "dfr_val"
        and selection.get("uses_group_labels") is True
        and selection.get("oracle") is True
        and protocol.get("group_labels_used_for_model_selection") is True
        and protocol.get("test_group_labels_used_for_model_selection") is False
    )


def _is_group_aware_afr_run(run: dict[str, Any]) -> bool:
    selection = run.get("selection", {})
    protocol = run.get("protocol", {})
    return (
        run.get("method_family") == "afr"
        and _is_afr_method(str(run.get("method")))
        and selection.get("split") == "val"
        and selection.get("subset") == "afr_selection_half"
        and selection.get("uses_group_labels") is True
        and protocol.get("group_labels_used_for_model_selection") is True
        and protocol.get("test_group_labels_used_for_model_selection") is False
    )


def _expanded_sweep_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand JSON files and directories into a deterministic file list."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path.is_dir():
            # Aggregate JSON files can live beside sweeps, so directories only
            # contribute files following the sweep writer's naming convention.
            files.extend(sorted(path.rglob("sweep_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Sweep path does not exist: {path}")
    if not files:
        raise ValueError(
            "No sweep JSON files supplied. Set AGGREGATE_CONFIG['sweep_paths'] "
            "or pass --sweep."
        )
    return files


def _run_final_iterations(run: dict[str, Any]) -> list[int]:
    """Return the final fitted head's iteration counts."""
    return run.get("fit", {}).get("n_iter", [])


def _dfr_protocol_key(run: dict[str, Any]) -> tuple[Any, ...] | None:
    if not _is_dfr_method(str(run.get("method", ""))):
        return None
    protocol = run.get("protocol", {})
    return (protocol.get("dfr_protocol_version", 1), protocol.get("subsample_count", 1),
            protocol.get("transform_fit_split", run.get("fit", {}).get("transform_fit_split")))


def load_sweep_runs(paths: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load compatible sweep documents and deduplicate repeated runs."""
    files = _expanded_sweep_files(paths)
    runs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("schema_version") != 1
            or document.get("kind") != "ridge_lambda_sweep"
        ):
            raise ValueError(f"Not a ridge-lambda sweep file: {path}")
        document_runs = document.get("runs")
        if not isinstance(document_runs, list):
            raise TypeError(f"{path}: 'runs' must be a list.")
        for run in document_runs:
            bundle = run.get("bundle", {})
            key = (
                bundle.get("fingerprint"),
                run.get("sample_size", {}).get("fraction"),
                run.get("method"),
                run.get("ridge_lambda"),
                run.get("class_balanced"),
                run.get("afr_gamma"),
                run.get("neurotune", {}).get("threshold"),
                run.get("protocol", {}).get("afr_protocol_version", 1),
                run.get("protocol", {}).get("neurotune_protocol_version", 1),
                _dfr_protocol_key(run),
            )
            if key in seen:
                continue
            seen.add(key)
            runs.append(run)
    return runs, [str(path) for path in files]


# =============================================================================
# Validation selection
# =============================================================================

def select_best_run(
    runs: Iterable[dict[str, Any]],
    *,
    criterion: str,
    one_se: bool = False,
    one_mistake: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select one run, allowing declared AFR/DFR validation-group selection."""
    criterion = _validated_criterion(criterion)
    if one_se and one_mistake:
        raise ValueError("one_se and one_mistake are mutually exclusive.")
    candidate_runs = list(runs)
    if criterion == "worst_group_accuracy":
        if not candidate_runs or not all(
            _is_oracle_dfr_run(run) or _is_group_aware_afr_run(run)
            for run in candidate_runs
        ):
            raise ValueError(
                "worst_group_accuracy selection is allowed only for explicitly "
                "marked group-aware AFR or oracle DFR runs."
            )
    two_stage_secondary = {"two_stage": "balanced_log_loss"}
    required_metrics = (
        ("balanced_accuracy", two_stage_secondary[criterion])
        if criterion in two_stage_secondary
        else ("balanced_accuracy",)
        if criterion == "class_balanced_accuracy"
        else (criterion,)
    )
    usable: list[dict[str, Any]] = []
    for run in candidate_runs:
        if run.get("status") != "complete":
            continue
        selection = run.get("selection", {})
        neurotune_version = int(
            run.get("protocol", {}).get("neurotune_protocol_version", 1)
        )
        expected_split = (
            "dfr_val" if _is_oracle_dfr_run(run)
            else "val_subset2" if _is_ho_erm_method(
                str(run.get("method", ""))
            )
            else "val_subset1" if (
                _is_neurotune_method(str(run.get("method", "")))
                and neurotune_version >= 5
            )
            else "val_subset1_sfit" if (
                _is_neurotune_method(str(run.get("method", "")))
                and neurotune_version >= 4
            )
            else "val_subset2" if (
                _is_neurotune_method(str(run.get("method", "")))
                and neurotune_version >= 2
            )
            else "val"
        )
        if selection.get("split") != expected_split:
            continue
        metrics = selection.get("metrics", {})
        ridge_lambda = run.get("ridge_lambda")
        values = [metrics.get(metric) for metric in required_metrics]
        if any(value is None for value in values) or ridge_lambda is None:
            continue
        if all(np.isfinite(float(value)) for value in values) and np.isfinite(
            float(ridge_lambda)
        ):
            usable.append(run)
    if not usable:
        raise ValueError(
            f"No converged validation results for criterion={criterion!r}."
        )
    def afr_hyperparameters(run: dict[str, Any]) -> tuple[float]:
        return (0.0 if run.get("afr_gamma") is None else float(run["afr_gamma"]),)

    def tuning_hyperparameters(run: dict[str, Any]) -> tuple[float, ...]:
        """Break exact NeuroTune ties toward the less aggressive mask."""
        threshold = run.get("neurotune", {}).get("threshold")
        return (*afr_hyperparameters(run), 0.0 if threshold is None else -float(threshold))

    def one_mistake_tolerance(run: dict[str, Any]) -> float:
        if criterion not in {"accuracy", "class_balanced_accuracy"}:
            raise ValueError(
                "--one-mistake is defined only for accuracy or "
                "class_balanced_accuracy selection."
            )
        raw_counts = run.get("selection", {}).get("label_counts")
        if not isinstance(raw_counts, dict) or not raw_counts:
            raise ValueError(
                "--one-mistake requires nonempty validation label_counts; "
                "rerun the sweep with the current code."
            )
        counts: list[int] = []
        for raw_count in raw_counts.values():
            if isinstance(raw_count, bool):
                raise ValueError(
                    "Validation label_counts must contain positive integers."
                )
            count = int(raw_count)
            if count <= 0 or float(raw_count) != count:
                raise ValueError(
                    "Validation label_counts must contain positive integers."
                )
            counts.append(count)
        if criterion == "accuracy":
            return 1.0 / sum(counts)
        return 1.0 / (len(counts) * min(counts))

    if one_se:
        if criterion == "sfit":
            raise ValueError(
                "--one-se is not defined for NeuroTune SFit selection."
            )
        if criterion in two_stage_secondary:
            raise ValueError(
                "--one-se is not defined for two-stage selection criteria; "
                "choose a single validation metric."
            )
        metric = (
            "balanced_accuracy"
            if criterion == "class_balanced_accuracy"
            else criterion
        )
        direction = SELECTION_CRITERIA[criterion]
        if direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"--one-se requires a single-stage criterion, not {criterion!r}."
            )
        def score(run: dict[str, Any]) -> float:
            return float(run["selection"]["metrics"][metric])

        best_by_score = sorted(
            usable,
            key=(
                lambda run: (-score(run), -float(run["ridge_lambda"]), *tuning_hyperparameters(run))
                if direction == "maximize"
                else (score(run), -float(run["ridge_lambda"]), *tuning_hyperparameters(run))
            ),
        )[0]
        raw_standard_error = best_by_score["selection"].get(
            "standard_errors", {}
        ).get(metric)
        if raw_standard_error is None:
            raise ValueError(
                f"--one-se requires the validation standard error for {metric!r}; "
                "rerun hyper_param_sweep.py with the current code."
            )
        standard_error = float(raw_standard_error)
        if not np.isfinite(standard_error) or standard_error < 0.0:
            raise ValueError(
                f"Validation standard error for {metric!r} must be finite "
                "and nonnegative."
            )
        best_score = score(best_by_score)
        threshold = (
            best_score - standard_error
            if direction == "maximize"
            else best_score + standard_error
        )

        def admissible(run: dict[str, Any]) -> bool:
            return (
                score(run) >= threshold
                if direction == "maximize"
                else score(run) <= threshold
            )

        def one_se_key(run: dict[str, Any]) -> tuple[float, ...]:
            if admissible(run):
                return (
                    0.0,
                    -float(run["ridge_lambda"]),
                    -score(run) if direction == "maximize" else score(run),
                    *tuning_hyperparameters(run),
                )
            return (
                1.0,
                -score(run) if direction == "maximize" else score(run),
                -float(run["ridge_lambda"]),
                *tuning_hyperparameters(run),
            )

        ranked = sorted(usable, key=one_se_key)
        return ranked[0], ranked

    if one_mistake:
        metric = (
            "balanced_accuracy"
            if criterion == "class_balanced_accuracy"
            else criterion
        )

        def score(run: dict[str, Any]) -> float:
            return float(run["selection"]["metrics"][metric])

        best_by_score = sorted(
            usable,
            key=lambda run: (
                -score(run),
                -float(run["ridge_lambda"]),
                *tuning_hyperparameters(run),
            ),
        )[0]
        tolerance = one_mistake_tolerance(best_by_score)
        threshold = score(best_by_score) - tolerance
        rounding_slack = 8.0 * np.finfo(float).eps

        def one_mistake_key(run: dict[str, Any]) -> tuple[float, ...]:
            if score(run) >= threshold - rounding_slack:
                return (
                    0.0,
                    -float(run["ridge_lambda"]),
                    -score(run),
                    *tuning_hyperparameters(run),
                )
            return (
                1.0,
                -score(run),
                -float(run["ridge_lambda"]),
                *tuning_hyperparameters(run),
            )

        ranked = sorted(usable, key=one_mistake_key)
        return ranked[0], ranked

    # Sorting instead of max/min makes the complete ranking available for
    # inspection. Single-stage criteria favor stronger regularization only
    # after exact criterion ties.
    if criterion in two_stage_secondary:
        secondary_metric = two_stage_secondary[criterion]

        def key(run: dict[str, Any]) -> tuple[float, ...]:
            metrics = run["selection"]["metrics"]
            balanced_accuracy = float(metrics["balanced_accuracy"])
            return (
                -balanced_accuracy,
                float(metrics[secondary_metric]),
                -float(run["ridge_lambda"]),
                *tuning_hyperparameters(run),
            )
    elif criterion == "class_balanced_accuracy":

        def key(run: dict[str, Any]) -> tuple[float, ...]:
            return (
                -float(run["selection"]["metrics"]["balanced_accuracy"]),
                -float(run["ridge_lambda"]),
                *tuning_hyperparameters(run),
            )
    elif SELECTION_CRITERIA[criterion] == "maximize":
        key = lambda run: (
            -float(run["selection"]["metrics"][criterion]),
            -float(run["ridge_lambda"]),
            *tuning_hyperparameters(run),
        )
    else:
        key = lambda run: (
            float(run["selection"]["metrics"][criterion]),
            -float(run["ridge_lambda"]),
            *tuning_hyperparameters(run),
        )
    ranked = sorted(usable, key=key)
    return ranked[0], ranked


def _selection_rule_for_method(
    method: str,
    *,
    default_rule: str,
    selection_rules: dict[str, str],
) -> str:
    """Use an exact method override and otherwise fall back to the default."""
    if _is_dfr_method(method):
        configured = selection_rules.get(method, "worst_group_accuracy")
        criterion = _validated_criterion(configured)
        if criterion != "worst_group_accuracy":
            raise ValueError(
                "DFR is an oracle method and must use worst_group_accuracy "
                "selection."
            )
        return criterion
    if _is_afr_method(method):
        return _validated_criterion(
            selection_rules.get(method, "worst_group_accuracy")
        )
    if _is_neurotune_method(method):
        configured = selection_rules.get(
            method,
            selection_rules.get("neurotune", "class_balanced_accuracy"),
        )
        criterion = _validated_criterion(configured)
        allowed = {
            "sfit",
            "accuracy",
            "class_balanced_accuracy",
            "log_loss",
            "balanced_log_loss",
            "two_stage",
        }
        if criterion not in allowed:
            raise ValueError(
                "NeuroTune selection must be group-free; choose sfit, "
                "accuracy, class_balanced_accuracy, log_loss, "
                "balanced_log_loss, or two_stage."
            )
        return criterion
    if _is_ho_erm_method(method):
        configured = selection_rules.get(method, "class_balanced_accuracy")
        criterion = _validated_criterion(configured)
        if criterion != "class_balanced_accuracy":
            raise ValueError(
                "HO_ERM must use class_balanced_accuracy selection."
            )
        return criterion
    return _validated_criterion(selection_rules.get(method, default_rule))


def _selected_validation_metric(
    run: dict[str, Any],
    criterion: str,
) -> float | dict[str, Any]:
    """Serialize the selected validation objective and its admissible set."""
    metrics = run["selection"]["metrics"]
    secondary_metrics = {"two_stage": "balanced_log_loss"}
    if criterion in secondary_metrics:
        secondary = secondary_metrics[criterion]
        return {
            "class_balanced_accuracy": metrics["balanced_accuracy"],
            secondary: metrics[secondary],
        }
    metric = (
        "balanced_accuracy"
        if criterion == "class_balanced_accuracy"
        else criterion
    )
    return float(metrics[metric])


# =============================================================================
# Aggregating selected test metrics
# =============================================================================

def _mean_interval(
    values: Iterable[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    """Return raw values, mean, and a two-sided t confidence half-width."""
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Aggregate metrics must be finite and non-empty.")
    mean = float(array.mean())
    if array.size == 1:
        half_width = None
    else:
        half_width = float(
            stats.sem(array)
            * stats.t.ppf((1.0 + confidence) / 2.0, array.size - 1)
        )
    return {
        "values": array.tolist(),
        "mean": mean,
        "confidence_half_width": half_width,
        "confidence": confidence,
        "n": int(array.size),
    }


def _paired_seed_difference(
    selected_runs: Sequence[dict[str, Any]],
    reference_runs: Sequence[dict[str, Any]],
    *,
    metric: str,
    confidence: float,
) -> dict[str, Any]:
    """Return a paired t interval for method-minus-reference across seeds."""
    if len(selected_runs) < 2:
        raise ValueError(
            "Paired representation-seed intervals require at least two seeds."
        )

    def by_fingerprint(
        runs: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for run in runs:
            fingerprint = str(run["bundle"].get("fingerprint"))
            if fingerprint in result:
                raise ValueError(
                    "Paired representation-seed intervals require one selected "
                    f"run per bundle; duplicate fingerprint={fingerprint}."
                )
            result[fingerprint] = run
        return result

    method_by_fingerprint = by_fingerprint(selected_runs)
    reference_by_fingerprint = by_fingerprint(reference_runs)
    if method_by_fingerprint.keys() != reference_by_fingerprint.keys():
        raise ValueError(
            "Paired representation-seed intervals require identical embedding "
            "bundles for the method and reference."
        )

    differences = np.asarray(
        [
            _selected_test_metric(method_by_fingerprint[fingerprint], metric)
            - _selected_test_metric(reference_by_fingerprint[fingerprint], metric)
            for fingerprint in sorted(method_by_fingerprint)
        ],
        dtype=np.float64,
    )
    observed = float(differences.mean())
    standard_error = float(stats.sem(differences))
    half_width = float(
        standard_error
        * stats.t.ppf((1.0 + confidence) / 2.0, differences.size - 1)
    )
    reference_method = str(reference_runs[0]["method"])
    return {
        "metric": metric,
        "reference_method": reference_method,
        "difference": "method_minus_reference",
        "resampling_unit": (
            "train_validation_split" if all("frozen_split" in run["bundle"] for run in selected_runs)
            else "representation_seed"
        ),
        "interval_method": "paired_t",
        "confidence": confidence,
        **({"split_replicate_count": int(differences.size)}
           if all("frozen_split" in run["bundle"] for run in selected_runs)
           else {"representation_replicate_count": int(differences.size)}),
        "observed_difference": observed,
        "standard_error": standard_error,
        "confidence_interval": {
            "lower": observed - half_width,
            "upper": observed + half_width,
        },
        "values": differences.tolist(),
    }


def _experiment_key(run: dict[str, Any]) -> tuple[Any, ...]:
    """Separate result panels that should never be averaged together."""
    bundle = run["bundle"]
    return (
        bundle.get("dataset"),
        bundle.get("representation"),
        bool(run.get("class_balanced")),
        run.get("sample_size", {}).get("fraction"),
    )


def _validated_selection_ridge_lambdas(
    values: Iterable[float] | None,
) -> tuple[float, ...] | None:
    """Validate an optional unique, positive aggregation-time ridge subset."""
    if values is None:
        return None
    ridges = tuple(float(value) for value in values)
    if not ridges:
        raise ValueError("selection_ridge_lambdas cannot be empty.")
    if any(not np.isfinite(value) or value <= 0.0 for value in ridges):
        raise ValueError(
            "Every selection ridge_lambda must be finite and strictly positive."
        )
    if len(set(ridges)) != len(ridges):
        raise ValueError("selection_ridge_lambdas contains duplicate values.")
    return ridges


def _validated_selection_neurotune_thresholds(
    values: Iterable[float] | None,
) -> tuple[float, ...] | None:
    """Validate an optional unique, finite NeuroTune threshold subset."""
    if values is None:
        return None
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("selection_neurotune_thresholds cannot be empty.")
    if any(not np.isfinite(value) for value in thresholds):
        raise ValueError(
            "Every selection NeuroTune threshold must be finite."
        )
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(
            "selection_neurotune_thresholds contains duplicate values."
        )
    return thresholds


def _validated_selection_afr_gammas(
    values: Iterable[float] | None,
) -> tuple[float, ...] | None:
    """Validate an optional unique, nonnegative AFR gamma subset."""
    if values is None:
        return None
    gammas = tuple(float(value) for value in values)
    if not gammas:
        raise ValueError("selection_afr_gammas cannot be empty.")
    if any(not np.isfinite(value) or value < 0.0 for value in gammas):
        raise ValueError(
            "Every selection AFR gamma must be finite and nonnegative."
        )
    if len(set(gammas)) != len(gammas):
        raise ValueError("selection_afr_gammas contains duplicate values.")
    return gammas


def _ridge_is_selected(value: object, selected: tuple[float, ...]) -> bool:
    ridge = float(value)
    return any(
        np.isclose(ridge, candidate, rtol=1e-12, atol=1e-12)
        for candidate in selected
    )


def _worst_group_accuracy(
    correct: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Calculate worst-group accuracy from per-observation correctness."""
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if correct.size == 0 or correct.size != groups.size:
        raise ValueError(
            "Correctness and group arrays must be non-empty and aligned."
        )
    return float(
        min(correct[groups == group].mean() for group in np.unique(groups))
    )


def _group_accuracy_from_correctness(
    correct: np.ndarray,
    groups: np.ndarray,
    metric: str,
) -> float:
    """Calculate a reporting-only group accuracy from fixed predictions."""
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if correct.size == 0 or correct.size != groups.size:
        raise ValueError("Correctness and group arrays must be non-empty and aligned.")
    group_ids = np.unique(groups)
    group_accuracies = {
        int(group): float(correct[groups == group].mean()) for group in group_ids
    }
    if metric == "group_balanced_accuracy":
        return float(np.mean(list(group_accuracies.values())))
    if metric == "worst_group_accuracy":
        return float(min(group_accuracies.values()))
    raise ValueError(f"Unsupported paired group metric: {metric!r}.")


def _selected_test_metric(run: dict[str, Any], metric: str) -> float:
    """Read a selected run's test metric."""
    try:
        return float(run["splits"]["test"][metric])
    except KeyError as exc:
        raise ValueError(f"Selected run has no test metric {metric!r}.") from exc


def _refit_selected_test_predictions_serial(
    selected_by_method: dict[str, list[dict[str, Any]]],
    *,
    refit_train_val: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reproduce fixed selected heads and retain test correctness vectors.

    Hyperparameters have already been selected. Ordinary methods refit on
    train, or train plus validation when requested. AFR rebuilds
    their validation adaptation sets; DFR reconstructs its balanced validation
    sample. Only predictions on the unchanged test split enter the bootstrap.
    """
    records: dict[str, dict[str, Any]] = {}
    saved_head_cache: dict[tuple[str, int], SavedLinearHead] = {}
    afr_partition_cache: dict[tuple[str, int, bool], np.ndarray] = {}
    afr_weight_cache: dict[
        tuple[str, str, int, float, bool],
        tuple[np.ndarray, dict[str, Any]],
    ] = {}
    neurotune_cache: dict[
        tuple[str, str, int, int, float],
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    ] = {}
    total = sum(len(selected_runs) for selected_runs in selected_by_method.values())
    completed = 0

    def report_progress(method: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, total, method)

    for method, selected_runs in sorted(selected_by_method.items()):
        for run in sorted(
            selected_runs,
            key=lambda item: str(item["bundle"].get("fingerprint")),
        ):
            bundle_identity = run["bundle"]
            fingerprint = str(bundle_identity.get("fingerprint"))
            bundle_path = bundle_identity.get("path")
            if not bundle_path:
                raise ValueError(
                    "Paired bootstrap requires bundle.path in every selected run."
                )
            record = records.get(fingerprint)
            if record is None:
                bundle = EmbeddingBundle.load(bundle_path, verify_hashes=False)
                if "frozen_split" in bundle_identity:
                    from frozen_results import resplit_bundle
                    bundle = resplit_bundle(bundle, int(bundle_identity["frozen_split"]["split_seed"]))
                if bundle.fingerprint != fingerprint:
                    raise ValueError(
                        "Embedding manifest fingerprint changed since the sweep: "
                        f"{bundle.root}"
                    )
                test = bundle.split("test")
                record = {
                    "bundle": bundle,
                    "bundle_identity": bundle_identity,
                    "y": np.asarray(test.y),
                    "groups": np.asarray(test.g),
                    "correct": {},
                    "metrics": {},
                }
                records[fingerprint] = record
            else:
                bundle = record["bundle"]

            transform_report = run.get("transform", {})
            if run.get("method_family") == "afr":
                validation = bundle.split("val")
                random_state = int(run["afr"]["partition"]["random_state"])
                score_report = run.get("fit", {}).get("score_head", {})
                if not isinstance(score_report, dict) or not score_report.get("path"):
                    raise ValueError("AFR refit requires the saved score-head path.")
                train = bundle.split("train")
                head_key = (str(score_report["path"]), int(train.X.shape[1]))
                score_head = saved_head_cache.get(head_key)
                if score_head is None:
                    score_head = SavedLinearHead.load(
                        score_report["path"],
                        classes=np.unique(train.y),
                        input_features=train.X.shape[1],
                        verify_hashes=False,
                    )
                    saved_head_cache[head_key] = score_head
                partition_key = (fingerprint, random_state, bool(refit_train_val))
                indices = afr_partition_cache.get(partition_key)
                if indices is None:
                    if refit_train_val:
                        indices = np.arange(validation.y.size)
                    else:
                        from sklearn.model_selection import StratifiedShuffleSplit
                        indices, _ = next(StratifiedShuffleSplit(
                            n_splits=1, test_size=0.5, random_state=random_state,
                        ).split(np.zeros((validation.y.size, 1)), validation.y))
                    indices = np.asarray(indices, dtype=np.int64)
                    afr_partition_cache[partition_key] = indices
                transform = make_transform(
                    run["requested_transform"],
                    whitening_relative_tolerance=transform_report.get(
                        "configured_relative_tolerance"
                    ),
                    whitening_estimator=transform_report.get("estimator", "empirical"),
                    class_weighted=bool(transform_report.get("class_weighted", False)),
                )
                transform = (
                    transform.fit(validation.X[indices], validation.y[indices])
                    if transform_report.get("class_weighted", False)
                    else transform.fit(validation.X[indices])
                )
                weight_key = (
                    fingerprint,
                    str(score_report["path"]),
                    random_state,
                    float(run["afr_gamma"]),
                    bool(refit_train_val),
                )
                cached_weights = afr_weight_cache.get(weight_key)
                if cached_weights is None:
                    cached_weights = compute_afr_weights(
                        score_head,
                        validation.X[indices],
                        validation.y[indices],
                        float(run["afr_gamma"]),
                    )
                    afr_weight_cache[weight_key] = cached_weights
                weights, weight_diagnostics = cached_weights
                transformed_fit_X = transform.transform(validation.X[indices])
                reference_head = _reference_head_in_transform(
                    score_head, transform
                )
                fit = fit_logistic_from_reference(
                    transformed_fit_X,
                    validation.y[indices],
                    LogisticConfig(
                        ridge_lambda=float(run["ridge_lambda"]),
                        class_balanced=False,
                    ),
                    reference_head,
                    proximity_penalty=True,
                    sample_weight=weights,
                    sample_weight_method=(
                        "afr_inverse_class_frequency_confidence_weight"
                    ),
                )
                test_X = transform.transform(bundle.split("test").X)
                correct = np.asarray(
                    fit.estimator.predict(test_X) == record["y"], dtype=bool
                )
                if not refit_train_val and not np.isclose(
                    _worst_group_accuracy(correct, record["groups"]),
                    run["splits"]["test"]["worst_group_accuracy"],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError("AFR reconstruction did not reproduce saved test accuracy.")
                record["correct"][method] = correct
                record["metrics"][method] = evaluation_metrics(
                    fit.estimator, test_X, record["y"], record["groups"]
                )
                coefficient, _ = transform.inverse_head(
                    fit.estimator.coef_, fit.estimator.intercept_
                )
                record.setdefault("coefficients", {})[method] = np.asarray(
                    coefficient, dtype=np.float64
                )
                split = "val_full_afr_reweighted" if refit_train_val else "val_afr_adaptation_half"
                record.setdefault("refit_diagnostics", {})[method] = {
                    "transform_fit_split": split,
                    "head_fit_split": split,
                    "observations": int(indices.size),
                    "score_head": score_head.diagnostics(),
                    "afr": weight_diagnostics,
                    "transform": transform.diagnostics(),
                    "fit": fit.diagnostics(),
                }
                report_progress(method)
                continue

            if _is_oracle_dfr_run(run):
                validation = bundle.split("val")
                split_seed = int(run.get("dfr", {}).get("random_state"))
                if refit_train_val:
                    # After choosing the ridge on the held-out DFR half, fit
                    # the final DFR head from the whole validation split. DFR
                    # remains group-balanced: every group contributes the
                    # full smallest-group count, sampled reproducibly from
                    # larger groups.
                    groups = np.asarray(validation.g, dtype=np.int64)
                    group_ids, group_counts = np.unique(
                        groups,
                        return_counts=True,
                    )
                    samples_per_group = int(group_counts.min())
                    rng = np.random.default_rng(split_seed)
                    balanced = np.concatenate(
                        [
                            rng.choice(
                                np.flatnonzero(groups == group),
                                size=samples_per_group,
                                replace=False,
                            )
                            for group in group_ids
                        ]
                    )
                    balanced = rng.permutation(balanced)
                    partition = DFRPartition(
                        random_state=split_seed, observations=len(validation.y),
                        fit_pool_indices=np.arange(len(validation.y)),
                        balanced_train_indices=balanced,
                        selection_indices=np.array([], dtype=np.int64))
                    classifier = fit_dfr_averaged_path(
                        validation.X, validation.y, validation.g,
                        partition=partition, ridge_lambdas=[float(run["ridge_lambda"])],
                        transform_name=str(run["requested_transform"]),
                        subsample_count=int(run.get("protocol", {}).get("subsample_count", 1)),
                        whitening_relative_tolerance=transform_report.get("configured_relative_tolerance"),
                        whitening_estimator=transform_report.get("estimator", "empirical"),
                        whitening_class_weighted=bool(transform_report.get("class_weighted", False)),
                    )[0]
                    record["correct"][method] = np.asarray(
                        classifier.predict(bundle.split("test").X) == record["y"], dtype=bool)
                    record["metrics"][method] = evaluation_metrics(
                        classifier, bundle.split("test").X, record["y"], record["groups"])
                    record.setdefault("coefficients", {})[method] = np.asarray(
                        classifier.fit_.estimator.coef_, dtype=np.float64
                    )
                    record.setdefault("refit_diagnostics", {})[method] = {
                        "transform_fit_split": "val_full_group_balanced_sample",
                        "transform_fit_observations": int(len(balanced)),
                        "head_fit_split": "val_full_group_balanced_sample",
                        "head_fit_observations": int(len(balanced)),
                        "samples_per_group": samples_per_group,
                        "subsample_count": len(classifier.members_),
                        "transform": classifier.transform_diagnostics(),
                        "fit": classifier.fit_diagnostics(validation.g),
                    }
                    report_progress(method)
                    continue
                partition = make_dfr_partition(
                    validation.g,
                    random_state=split_seed,
                )
                classifier = fit_dfr_averaged_path(
                    validation.X, validation.y, validation.g,
                    partition=partition, ridge_lambdas=[float(run["ridge_lambda"])],
                    transform_name=str(run["requested_transform"]),
                    subsample_count=int(run.get("protocol", {}).get("subsample_count", 1)),
                    whitening_relative_tolerance=transform_report.get("configured_relative_tolerance"),
                    whitening_estimator=transform_report.get("estimator", "empirical"),
                    whitening_class_weighted=bool(transform_report.get("class_weighted", False)),
                )[0]
                reproduced_method = dfr_method_name(
                    classifier.transform_diagnostics()["name"]
                )
                if reproduced_method != method:
                    raise ValueError(
                        "Selected DFR method could not be reproduced: "
                        f"saved={method!r}, reproduced={reproduced_method!r}."
                    )
                predictions = classifier.predict(bundle.split("test").X)
                correct = np.asarray(predictions == record["y"], dtype=bool)
                reproduced = _worst_group_accuracy(correct, record["groups"])
                saved = float(run["splits"]["test"]["worst_group_accuracy"])
                if not refit_train_val and not np.isclose(
                    reproduced, saved, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        "Bootstrap DFR refit did not reproduce the saved test "
                        f"result for {method} ({reproduced} != {saved})."
                    )
                record["correct"][method] = correct
                record.setdefault("coefficients", {})[method] = np.asarray(
                    classifier.fit_.estimator.coef_, dtype=np.float64
                )
                continue

            if run.get("method_family") == "neurotune":
                train = bundle.split("train")
                validation = bundle.split("val")
                selector_report = run.get("fit", {}).get("selector_head", {})
                if not isinstance(selector_report, dict) or not selector_report.get("path"):
                    raise ValueError(
                        "NeuroTune reconstruction requires the saved selector-head path."
                    )
                head_key = (str(selector_report["path"]), int(train.X.shape[1]))
                selector_head = saved_head_cache.get(head_key)
                if selector_head is None:
                    selector_head = SavedLinearHead.load(
                        selector_report["path"],
                        classes=np.unique(train.y),
                        input_features=train.X.shape[1],
                        verify_hashes=False,
                    )
                    saved_head_cache[head_key] = selector_head
                version = int(run.get("protocol", {}).get("neurotune_protocol_version", 1))
                partition = run.get("protocol", {}).get("validation_partition", {})
                partition_seed = int(partition.get("random_state", 0))
                threshold = float(run["neurotune"]["threshold"])
                selector_key = (
                    fingerprint,
                    str(selector_report["path"]),
                    version,
                    partition_seed,
                    threshold,
                )
                cached_selector = neurotune_cache.get(selector_key)
                if cached_selector is None:
                    if version >= 2:
                        identification_indices, tuning_indices, _ = (
                            _neurotune_validation_partition(
                                validation.y,
                                random_state=partition_seed,
                            )
                        )
                    else:
                        identification_indices = np.arange(
                            validation.y.size, dtype=np.int64
                        )
                        tuning_indices = identification_indices
                    retained, identification = identify_neurotune_coordinates(
                        selector_head,
                        validation.X[identification_indices],
                        validation.y[identification_indices],
                        threshold=threshold,
                    )
                    cached_selector = (
                        np.asarray(identification_indices, dtype=np.int64),
                        np.asarray(tuning_indices, dtype=np.int64),
                        retained,
                        identification,
                    )
                    neurotune_cache[selector_key] = cached_selector
                (
                    identification_indices,
                    tuning_indices,
                    retained,
                    identification,
                ) = cached_selector
                transform = make_transform(
                    str(run["requested_transform"]),
                    whitening_relative_tolerance=transform_report.get(
                        "configured_relative_tolerance"
                    ),
                    whitening_estimator=str(
                        transform_report.get("estimator", "empirical")
                    ),
                    class_weighted=bool(
                        transform_report.get("class_weighted", False)
                    ),
                )
                if version >= 2:
                    if refit_train_val:
                        fit_X = validation.X[:, retained]
                        fit_y = validation.y
                    elif version >= 4:
                        fit_X = validation.X[tuning_indices][:, retained]
                        fit_y = validation.y[tuning_indices]
                    else:
                        fit_X = validation.X[identification_indices][:, retained]
                        fit_y = validation.y[identification_indices]
                else:
                    fit_X, fit_y = train.X[:, retained], train.y
                transform = (
                    transform.fit(fit_X, fit_y)
                    if transform_report.get("class_weighted", False)
                    else transform.fit(fit_X)
                )
                reproduced_method = neurotune_method_name(
                    transform.diagnostics()["name"]
                )
                if reproduced_method != method:
                    raise ValueError(
                        "Selected NeuroTune transform could not be reproduced: "
                        f"saved={method!r}, reproduced={reproduced_method!r}."
                    )
                fit = fit_logistic(
                    transform.transform(fit_X),
                    fit_y,
                    LogisticConfig(
                        ridge_lambda=float(run["ridge_lambda"]),
                        class_balanced=bool(run.get("class_balanced")),
                    ),
                )
                if not fit.converged:
                    raise RuntimeError(
                        f"NeuroTune refit did not converge for {method}: "
                        + "; ".join(fit.convergence_messages)
                    )
                test_X = transform.transform(bundle.split("test").X[:, retained])
                correct = np.asarray(
                    fit.estimator.predict(test_X) == record["y"], dtype=bool
                )
                reproduced = _worst_group_accuracy(correct, record["groups"])
                saved = float(run["splits"]["test"]["worst_group_accuracy"])
                if not refit_train_val and not np.isclose(
                    reproduced, saved, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        "NeuroTune reconstruction did not reproduce the saved "
                        f"test result for {method} ({reproduced} != {saved})."
                    )
                record["correct"][method] = correct
                record["metrics"][method] = evaluation_metrics(
                    fit.estimator, test_X, record["y"], record["groups"]
                )
                retained_coefficient, _ = transform.inverse_head(
                    fit.estimator.coef_, fit.estimator.intercept_
                )
                coefficient = np.zeros(
                    (retained_coefficient.shape[0], train.X.shape[1]),
                    dtype=np.float64,
                )
                coefficient[:, retained] = retained_coefficient
                record.setdefault("coefficients", {})[method] = coefficient
                record.setdefault("refit_diagnostics", {})[method] = {
                    "identification_split": "val_subset1" if version >= 2 else "val",
                    "transform_fit_split": (
                        ("val_subset1_plus_val_subset2" if refit_train_val else "val_subset1")
                        if 2 <= version < 4
                        else ("val_full" if refit_train_val else "val_subset2")
                        if version >= 4
                        else "train"
                    ),
                    "head_fit_split": (
                        ("val_subset1_plus_val_subset2" if refit_train_val else "val_subset1")
                        if 2 <= version < 4
                        else ("val_full" if refit_train_val else "val_subset2")
                        if version >= 4
                        else "train"
                    ),
                    "observations": int(fit_y.size),
                    "neurotune": identification,
                    "transform": transform.diagnostics(),
                    "fit": fit.diagnostics(),
                }
                report_progress(method)
                continue

            if run.get("method_family") == "ho_erm":
                validation = bundle.split("val")
                partition_report = run.get("protocol", {}).get(
                    "validation_partition", {}
                )
                if refit_train_val:
                    fit_indices = np.arange(validation.y.size, dtype=np.int64)
                    fit_split = "val_full"
                else:
                    fit_indices = _ho_erm_validation_partition(
                        validation.y,
                        random_state=int(partition_report.get("random_state", 0)),
                    )[0]
                    fit_split = "val_subset1"
                fit_X = validation.X[fit_indices]
                fit_y = validation.y[fit_indices]
                transform = make_transform(
                    str(run["requested_transform"]),
                    whitening_relative_tolerance=transform_report.get(
                        "configured_relative_tolerance"
                    ),
                    whitening_estimator=str(
                        transform_report.get("estimator", "empirical")
                    ),
                    class_weighted=bool(
                        transform_report.get("class_weighted", False)
                    ),
                )
                transform = (
                    transform.fit(fit_X, fit_y)
                    if transform_report.get("class_weighted", False)
                    else transform.fit(fit_X)
                )
                reproduced_method = ho_erm_method_name(
                    transform.diagnostics()["name"]
                )
                if reproduced_method != method:
                    raise ValueError(
                        "Selected HO_ERM transform could not be reproduced: "
                        f"saved={method!r}, reproduced={reproduced_method!r}."
                    )
                fit = fit_logistic(
                    transform.transform(fit_X),
                    fit_y,
                    LogisticConfig(
                        ridge_lambda=float(run["ridge_lambda"]),
                        class_balanced=True,
                    ),
                )
                if not fit.converged:
                    raise RuntimeError(
                        f"HO_ERM refit did not converge for {method}: "
                        + "; ".join(fit.convergence_messages)
                    )
                test_X = transform.transform(bundle.split("test").X)
                correct = np.asarray(
                    fit.estimator.predict(test_X) == record["y"], dtype=bool
                )
                if not refit_train_val and not np.isclose(
                    _worst_group_accuracy(correct, record["groups"]),
                    run["splits"]["test"]["worst_group_accuracy"],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "HO_ERM reconstruction did not reproduce saved test "
                        f"accuracy for {method}."
                    )
                record["correct"][method] = correct
                record["metrics"][method] = evaluation_metrics(
                    fit.estimator, test_X, record["y"], record["groups"]
                )
                coefficient, _ = transform.inverse_head(
                    fit.estimator.coef_, fit.estimator.intercept_
                )
                record.setdefault("coefficients", {})[method] = np.asarray(
                    coefficient, dtype=np.float64
                )
                record.setdefault("refit_diagnostics", {})[method] = {
                    "transform_fit_split": fit_split,
                    "head_fit_split": fit_split,
                    "observations": int(fit_indices.size),
                    "class_balanced": True,
                    "transform": transform.diagnostics(),
                    "fit": fit.diagnostics(),
                }
                report_progress(method)
                continue

            train = bundle.split("train")
            validation = bundle.split("val")
            fit_X = (
                np.concatenate((train.X, validation.X), axis=0)
                if refit_train_val
                else train.X
            )
            fit_y = (
                np.concatenate((train.y, validation.y), axis=0)
                if refit_train_val
                else train.y
            )
            transform = make_transform(
                str(run.get("requested_transform") or method),
                whitening_relative_tolerance=transform_report.get(
                    "configured_relative_tolerance"
                ),
                whitening_estimator=str(
                    transform_report.get("estimator", "empirical")
                ),
                class_weighted=bool(transform_report.get("class_weighted", False)),
            )
            transform = (
                transform.fit(fit_X, fit_y)
                if transform_report.get("class_weighted", False)
                else transform.fit(fit_X)
            )
            reproduced_method = str(transform.diagnostics()["name"])
            if reproduced_method != method:
                raise ValueError(
                    "Selected transform could not be reproduced: "
                    f"saved method={method!r}, reproduced={reproduced_method!r}."
                )

            fit = fit_logistic(
                transform.transform(fit_X),
                fit_y,
                LogisticConfig(
                    ridge_lambda=float(run["ridge_lambda"]),
                    class_balanced=bool(run.get("class_balanced")),
                ),
            )
            if not fit.converged:
                raise RuntimeError(
                    f"Bootstrap refit did not converge for {method}: "
                    + "; ".join(fit.convergence_messages)
                )
            predictions = fit.estimator.predict(
                transform.transform(bundle.split("test").X)
            )
            correct = np.asarray(predictions == record["y"], dtype=bool)
            reproduced = _worst_group_accuracy(correct, record["groups"])
            saved = float(run["splits"]["test"]["worst_group_accuracy"])
            if not refit_train_val and not np.isclose(
                reproduced, saved, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "Bootstrap refit did not reproduce the saved test result "
                    f"for {method} ({reproduced} != {saved})."
                )
            record["correct"][method] = correct
            coefficient, _ = transform.inverse_head(
                fit.estimator.coef_, fit.estimator.intercept_
            )
            record.setdefault("coefficients", {})[method] = np.asarray(
                coefficient, dtype=np.float64
            )
            if refit_train_val:
                record["metrics"][method] = evaluation_metrics(
                    fit.estimator,
                    transform.transform(bundle.split("test").X),
                    record["y"],
                    record["groups"],
                )
            report_progress(method)
    # Bundle objects contain memory maps and are only needed while fitting.
    # Dropping them keeps process-worker results compact and cheap to transfer.
    for record in records.values():
        record.pop("bundle", None)
    return records


def _refit_seed_job(
    selected_by_method: dict[str, list[dict[str, Any]]],
    *,
    refit_train_val: bool,
    blas_threads: int,
) -> dict[str, dict[str, Any]]:
    """Refit one representation seed with bounded numerical parallelism."""
    with threadpool_limits(limits=blas_threads):
        return _refit_selected_test_predictions_serial(
            selected_by_method,
            refit_train_val=refit_train_val,
        )


def _refit_selected_test_predictions(
    selected_by_method: dict[str, list[dict[str, Any]]],
    *,
    refit_train_val: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    workers: int = 1,
    blas_threads: int = 1,
) -> dict[str, dict[str, Any]]:
    """Refit selected heads, concurrently across independent bundle seeds."""
    workers = int(workers)
    blas_threads = int(blas_threads)
    if workers < 1 or blas_threads < 1:
        raise ValueError("Refit workers and BLAS threads must be positive integers.")

    by_fingerprint: dict[str, dict[str, list[dict[str, Any]]]] = {}
    total = 0
    for method, selected_runs in selected_by_method.items():
        for run in selected_runs:
            fingerprint = str(run["bundle"].get("fingerprint"))
            by_fingerprint.setdefault(fingerprint, {}).setdefault(method, []).append(run)
            total += 1

    if workers == 1 or len(by_fingerprint) <= 1:
        with threadpool_limits(limits=blas_threads):
            return _refit_selected_test_predictions_serial(
                selected_by_method,
                refit_train_val=refit_train_val,
                progress=progress,
            )

    records: dict[str, dict[str, Any]] = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=min(workers, len(by_fingerprint))) as executor:
        futures = {
            executor.submit(
                _refit_seed_job,
                seed_runs,
                refit_train_val=refit_train_val,
                blas_threads=blas_threads,
            ): seed_runs
            for seed_runs in by_fingerprint.values()
        }
        for future in as_completed(futures):
            seed_runs = futures[future]
            seed_records = future.result()
            overlap = records.keys() & seed_records.keys()
            if overlap:
                raise RuntimeError(
                    "Parallel refit returned duplicate bundle fingerprints: "
                    + ", ".join(sorted(overlap))
                )
            records.update(seed_records)
            for method, runs in seed_runs.items():
                completed += len(runs)
                if progress is not None:
                    progress(completed, total, method)
    return records


def _paired_group_metric_bootstraps(
    records: dict[str, dict[str, Any]],
    *,
    method: str,
    reference_methods: Sequence[str],
    confidence: float,
    n_resamples: int,
    rng: np.random.Generator,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Bootstrap all group-accuracy contrasts in one shared resampling pass.

    The refitted correctness vectors are fixed, so the only stochastic work is
    the within-group resampling. For each method and group, draw the paired
    correctness outcomes for all requested references together. This avoids
    rerunning the bootstrap setup and keeps all requested contrasts on the same
    resampling pass.
    """
    if n_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer.")
    reference_methods = tuple(dict.fromkeys(reference_methods))
    if not reference_methods:
        raise ValueError("At least one bootstrap reference is required.")
    paired_records = [
        record
        for record in records.values()
        if method in record["correct"]
    ]
    if not paired_records:
        raise ValueError(f"No selected test predictions found for {method!r}.")
    for reference_method in reference_methods:
        missing_reference = [
            str(record["bundle_identity"].get("fingerprint"))
            for record in paired_records
            if reference_method not in record["correct"]
        ]
        if missing_reference:
            raise ValueError(
                f"Paired bootstrap requires {reference_method!r} for every "
                "bundle; missing fingerprints: "
                f"{', '.join(missing_reference)}"
            )

    bootstrap_differences = {
        metric: {
            reference_method: np.zeros(n_resamples, dtype=np.float64)
            for reference_method in reference_methods
        }
        for metric in PAIRED_DIFFERENCE_METRICS
    }
    per_bundle = {
        metric: {reference_method: [] for reference_method in reference_methods}
        for metric in PAIRED_DIFFERENCE_METRICS
    }
    prepared_records: list[dict[str, Any]] = []
    for record in paired_records:
        groups = record["groups"]
        method_correct = record["correct"][method]
        group_counts: dict[str, int] = {}
        prepared_groups: list[dict[str, Any]] = []

        # Drawing multinomial counts over the four paired correctness outcomes
        # is exactly equivalent to resampling observation indices within this
        # group, but avoids materializing B x n_test index arrays. Store the
        # probabilities once, then draw them in progress-reporting chunks.
        for group in np.unique(groups):
            mask = groups == group
            n_group = int(mask.sum())
            group_counts[str(int(group))] = n_group
            # Shape is (n_resamples, n_references, 4), with categories ordered
            # as (method_correct, reference_correct). One multinomial call
            # performs the bootstrap draw for every requested contrast.
            probabilities = []
            for reference_method in reference_methods:
                reference_correct = record["correct"][reference_method]
                joint_category = (
                    2 * method_correct[mask].astype(np.int8)
                    + reference_correct[mask].astype(np.int8)
                )
                counts = np.bincount(joint_category, minlength=4)
                probabilities.append(counts / n_group)
            prepared_groups.append(
                {
                    "group": int(group),
                    "n_group": n_group,
                    "probabilities": np.asarray(probabilities),
                }
            )
        for reference_method in reference_methods:
            for metric in PAIRED_DIFFERENCE_METRICS:
                observed_difference = (
                    _group_accuracy_from_correctness(
                        method_correct, groups, metric
                    )
                    - _group_accuracy_from_correctness(
                        record["correct"][reference_method], groups, metric
                    )
                )
                per_bundle[metric][reference_method].append(
                    {
                        "bundle": record["bundle_identity"],
                        "group_counts": group_counts,
                        "observed_difference": float(observed_difference),
                    }
                )
        prepared_records.append({"groups": prepared_groups})

    # Complete every group and bundle for a block of resamples before
    # reporting it as finished. At most 20 updates keeps interactive output
    # useful without sacrificing the vectorized multinomial implementation.
    chunk_size = max(1, (n_resamples + 19) // 20)
    if progress is not None:
        progress(0, n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        chunk_resamples = stop - start
        chunk_differences = {
            metric: {
                reference_method: np.zeros(chunk_resamples, dtype=np.float64)
                for reference_method in reference_methods
            }
            for metric in PAIRED_DIFFERENCE_METRICS
        }
        for prepared_record in prepared_records:
            method_group_bootstrap = {
                reference_method: [] for reference_method in reference_methods
            }
            reference_group_bootstrap = {
                reference_method: [] for reference_method in reference_methods
            }
            for prepared_group in prepared_record["groups"]:
                n_group = prepared_group["n_group"]
                draws = rng.multinomial(
                    n_group,
                    prepared_group["probabilities"],
                    size=(chunk_resamples, len(reference_methods)),
                )
                for reference_index, reference_method in enumerate(
                    reference_methods
                ):
                    reference_draws = draws[:, reference_index, :]
                    method_group_bootstrap[reference_method].append(
                        (reference_draws[:, 2] + reference_draws[:, 3]) / n_group
                    )
                    reference_group_bootstrap[reference_method].append(
                        (reference_draws[:, 1] + reference_draws[:, 3]) / n_group
                    )
            for reference_method in reference_methods:
                method_matrix = np.column_stack(
                    method_group_bootstrap[reference_method]
                )
                reference_matrix = np.column_stack(
                    reference_group_bootstrap[reference_method]
                )
                metric_values = {
                    "group_balanced_accuracy": (
                        method_matrix.mean(axis=1), reference_matrix.mean(axis=1)
                    ),
                    "worst_group_accuracy": (
                        method_matrix.min(axis=1), reference_matrix.min(axis=1)
                    ),
                }
                for metric, (
                    method_values,
                    reference_values,
                ) in metric_values.items():
                    chunk_differences[metric][reference_method] += (
                        method_values - reference_values
                    )
        for metric in PAIRED_DIFFERENCE_METRICS:
            for reference_method in reference_methods:
                bootstrap_differences[metric][reference_method][start:stop] = (
                    chunk_differences[metric][reference_method]
                )
        if progress is not None:
            progress(stop, n_resamples)

    summaries: dict[str, dict[str, dict[str, Any]]] = {
        metric: {} for metric in PAIRED_DIFFERENCE_METRICS
    }
    alpha = 1.0 - confidence
    for metric in PAIRED_DIFFERENCE_METRICS:
        for reference_method in reference_methods:
            differences = bootstrap_differences[metric][reference_method] / len(
                paired_records
            )
            observed = float(
                np.mean(
                    [
                        entry["observed_difference"]
                        for entry in per_bundle[metric][reference_method]
                    ]
                )
            )
            lower, upper = np.quantile(
                differences,
                [alpha / 2.0, 1.0 - alpha / 2.0],
            )
            summaries[metric][reference_method] = {
                "metric": metric,
                "reference_method": reference_method,
                "difference": "method_minus_reference",
                "resampling_unit": "test_observation",
                "stratification": "test_group",
                "interval_method": "percentile",
                "confidence": confidence,
                "n_resamples": n_resamples,
                "representation_replicate_count": len(paired_records),
                "observed_difference": observed,
                "bootstrap_mean": float(differences.mean()),
                "bootstrap_standard_error": float(
                    differences.std(ddof=1) if n_resamples > 1 else 0.0
                ),
                "confidence_interval": {
                    "lower": float(lower),
                    "upper": float(upper),
                },
                "per_bundle": per_bundle[metric][reference_method],
            }
    return summaries


def aggregate_runs(
    runs: Iterable[dict[str, Any]],
    *,
    default_selection_rule: str = "class_balanced_accuracy",
    selection_rules: dict[str, str] | None = None,
    selection_ridge_lambdas: Iterable[float] | None = None,
    selection_afr_gammas: Iterable[float] | None = None,
    selection_neurotune_thresholds: Iterable[float] | None = None,
    max_ridge: float | None = None,
    max_gamma: float | None = None,
    exclude_single_iteration: bool = False,
    one_se: bool = False,
    one_mistake: bool = False,
    refit_train_val: bool | None = None,
    confidence: float = 0.90,
    bootstrap: bool = False,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_progress: bool = False,
    refit_progress: bool = False,
    refit_workers: int | None = None,
    refit_blas_threads: int = 1,
) -> dict[str, Any]:
    """Select per bundle/method, then average selected test metrics."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one.")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer.")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be nonnegative.")
    if refit_workers is None:
        refit_workers = int(AGGREGATE_CONFIG["refit_workers"])
    if (
        isinstance(refit_workers, bool)
        or int(refit_workers) != refit_workers
        or int(refit_workers) < 1
    ):
        raise ValueError("refit_workers must be a positive integer.")
    if (
        isinstance(refit_blas_threads, bool)
        or int(refit_blas_threads) != refit_blas_threads
        or int(refit_blas_threads) < 1
    ):
        raise ValueError("refit_blas_threads must be a positive integer.")
    refit_workers = int(refit_workers)
    refit_blas_threads = int(refit_blas_threads)
    selected_ridges = _validated_selection_ridge_lambdas(
        selection_ridge_lambdas
    )
    selected_neurotune_thresholds = (
        _validated_selection_neurotune_thresholds(
            selection_neurotune_thresholds
        )
    )
    selected_afr_gammas = _validated_selection_afr_gammas(
        selection_afr_gammas
    )
    if max_ridge is not None:
        max_ridge = float(max_ridge)
        if not np.isfinite(max_ridge) or max_ridge <= 0.0:
            raise ValueError("max_ridge must be finite and strictly positive.")
    if max_gamma is not None:
        max_gamma = float(max_gamma)
        if not np.isfinite(max_gamma) or max_gamma <= 0.0:
            raise ValueError("max_gamma must be finite and strictly positive.")
    if one_se and one_mistake:
        raise ValueError("one_se and one_mistake are mutually exclusive.")
    default_rule = _validated_criterion(default_selection_rule)
    rules = {
        str(method): _validated_criterion(criterion)
        for method, criterion in (selection_rules or {}).items()
    }

    # First isolate experimental settings such as dataset and representation.
    experiment_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        if not isinstance(run.get("bundle"), dict):
            raise ValueError("Every sweep run must contain bundle identity.")
        experiment_groups.setdefault(_experiment_key(run), []).append(run)
    if not experiment_groups:
        raise ValueError("No sweep runs are available to aggregate.")

    experiments: list[dict[str, Any]] = []
    excluded_single_iteration_count = 0
    for experiment_key, experiment_runs in sorted(
        experiment_groups.items(), key=lambda item: str(item[0])
    ):
        if len({_dfr_protocol_key(r) for r in experiment_runs
                if _is_dfr_method(str(r.get("method", "")))}) > 1:
            raise ValueError("Cannot combine different DFR protocols or subsample counts; rerun or aggregate separately.")
        if len({r.get("protocol", {}).get("afr_protocol_version", 1)
                for r in experiment_runs if _is_afr_method(str(r.get("method", "")))}) > 1:
            raise ValueError("Cannot combine different AFR protocols; rerun or aggregate separately.")
        if len({r.get("protocol", {}).get("neurotune_protocol_version", 1)
                for r in experiment_runs if _is_neurotune_method(str(r.get("method", "")))}) > 1:
            raise ValueError(
                "Cannot combine different NeuroTune protocols; rerun or "
                "aggregate separately."
            )
        if selected_ridges is not None:
            available_ridges: dict[tuple[str, str], set[float]] = {}
            for run in experiment_runs:
                pair = (
                    str(run["bundle"].get("fingerprint")),
                    str(run.get("method")),
                )
                available_ridges.setdefault(pair, set()).add(
                    float(run["ridge_lambda"])
                )
            missing_ridges: list[str] = []
            for pair, available in sorted(available_ridges.items()):
                missing = [
                    ridge
                    for ridge in selected_ridges
                    if not _ridge_is_selected(ridge, tuple(available))
                ]
                if missing:
                    missing_ridges.append(
                        f"fingerprint={pair[0]}, method={pair[1]}: "
                        + ", ".join(f"{ridge:g}" for ridge in missing)
                    )
            if missing_ridges:
                raise ValueError(
                    "Requested --ridge-lambda values are not saved for every "
                    "bundle/method pair; missing " + "; ".join(missing_ridges)
                )
        if selected_neurotune_thresholds is not None:
            available_thresholds: dict[tuple[str, str], set[float]] = {}
            for run in experiment_runs:
                method = str(run.get("method"))
                if not _is_neurotune_method(method):
                    continue
                pair = (
                    str(run["bundle"].get("fingerprint")),
                    method,
                )
                threshold = run.get("neurotune", {}).get("threshold")
                if threshold is not None:
                    available_thresholds.setdefault(pair, set()).add(
                        float(threshold)
                    )
            missing_thresholds: list[str] = []
            for pair, available in sorted(available_thresholds.items()):
                missing = [
                    threshold
                    for threshold in selected_neurotune_thresholds
                    if not _ridge_is_selected(threshold, tuple(available))
                ]
                if missing:
                    missing_thresholds.append(
                        f"fingerprint={pair[0]}, method={pair[1]}: "
                        + ", ".join(f"{threshold:g}" for threshold in missing)
                    )
            if missing_thresholds:
                raise ValueError(
                    "Requested --neurotune-threshold values are not saved for "
                    "every NeuroTune bundle/method pair; missing "
                    + "; ".join(missing_thresholds)
                )
        if selected_afr_gammas is not None:
            available_gammas: dict[tuple[str, str], set[float]] = {}
            for run in experiment_runs:
                method = str(run.get("method"))
                if not _is_afr_method(method):
                    continue
                pair = (str(run["bundle"].get("fingerprint")), method)
                available_gammas.setdefault(pair, set()).add(
                    float(run["afr_gamma"])
                )
            missing_gammas = []
            for pair, available in sorted(available_gammas.items()):
                missing = [
                    gamma for gamma in selected_afr_gammas
                    if not _ridge_is_selected(gamma, tuple(available))
                ]
                if missing:
                    missing_gammas.append(
                        f"fingerprint={pair[0]}, method={pair[1]}: "
                        + ", ".join(f"{gamma:g}" for gamma in missing)
                    )
            if missing_gammas:
                raise ValueError(
                    "Requested --afr-gamma values are not saved for every "
                    "AFR bundle/method pair; missing "
                    + "; ".join(missing_gammas)
                )

        # Ridge selection is independent for every representation seed and
        # transformation.
        candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        available_pairs: set[tuple[str, str]] = set()
        for run in experiment_runs:
            fingerprint = str(run["bundle"].get("fingerprint"))
            method = str(run.get("method"))
            available_pairs.add((fingerprint, method))
            if selected_ridges is not None and not _ridge_is_selected(
                run.get("ridge_lambda"), selected_ridges
            ):
                continue
            if max_ridge is not None and float(run["ridge_lambda"]) >= max_ridge:
                continue
            if (
                selected_neurotune_thresholds is not None
                and _is_neurotune_method(method)
                and not _ridge_is_selected(
                    run.get("neurotune", {}).get("threshold"),
                    selected_neurotune_thresholds,
                )
            ):
                continue
            if (
                selected_afr_gammas is not None
                and _is_afr_method(method)
                and not _ridge_is_selected(run["afr_gamma"], selected_afr_gammas)
            ):
                continue
            if (
                max_gamma is not None
                and _is_afr_method(method)
                and float(run["afr_gamma"]) >= max_gamma
            ):
                continue
            iterations = _run_final_iterations(run)
            if (
                exclude_single_iteration
                and iterations
                and max(int(value) for value in iterations) == 1
            ):
                excluded_single_iteration_count += 1
                continue
            candidates.setdefault((fingerprint, method), []).append(run)
        missing_pairs = available_pairs.difference(candidates)
        if missing_pairs:
            missing = ", ".join(
                f"fingerprint={fingerprint}, method={method}"
                for fingerprint, method in sorted(missing_pairs)
            )
            restrictions = []
            if selected_ridges is not None:
                restrictions.append(
                    "--ridge-lambda "
                    + " --ridge-lambda ".join(
                        f"{ridge:g}" for ridge in selected_ridges
                    )
                )
            if selected_neurotune_thresholds is not None:
                restrictions.append(
                    "--neurotune-threshold "
                    + " --neurotune-threshold ".join(
                        f"{threshold:g}"
                        for threshold in selected_neurotune_thresholds
                    )
                )
            if max_ridge is not None:
                restrictions.append(f"--max-ridge {max_ridge:g}")
            if max_gamma is not None:
                restrictions.append(f"--max-gamma {max_gamma:g}")
            if selected_afr_gammas is not None:
                restrictions.append(
                    "--afr-gamma "
                    + " --afr-gamma ".join(
                        f"{gamma:g}" for gamma in selected_afr_gammas
                    )
                )
            if exclude_single_iteration:
                restrictions.append("--exclude-single-iteration")
            restriction = " and ".join(restrictions)
            raise ValueError(
                f"{restriction} requires matching candidates for every "
                f"bundle/method pair; none remain "
                f"for {missing}."
            )

        selected_by_method: dict[str, list[dict[str, Any]]] = {}
        for (fingerprint, method), method_runs in candidates.items():
            criterion = _selection_rule_for_method(
                method,
                default_rule=default_rule,
                selection_rules=rules,
            )
            selected, _ = select_best_run(
                method_runs,
                criterion=criterion,
                one_se=one_se,
                one_mistake=one_mistake,
            )
            selected_by_method.setdefault(method, []).append(selected)

        refit_methods = {
            method for method, selected_runs in selected_by_method.items()
            if not _is_neurotune_method(method)
            and (refit_train_val if refit_train_val is not None else
                 selected_runs[0].get("method_family") in {
                     "dfr", "afr", "ho_erm"
                 })
        }
        refit_records = (
            _refit_selected_test_predictions(
                {method: selected_by_method[method] for method in refit_methods},
                refit_train_val=True,
                progress=(
                    lambda completed, total, method: print(
                        f"\rrefit selected heads: {completed:,}/{total:,} ({method})",
                        end="\n" if completed == total else "",
                        flush=True,
                    )
                    if refit_progress
                    else None
                ),
                workers=refit_workers,
                blas_threads=refit_blas_threads,
            )
            if refit_methods
            else None
        )
        if refit_records is not None:
            selected_by_method = {
                method: [
                    {
                        **run,
                        "test_refit": refit_records[
                            str(run["bundle"].get("fingerprint"))
                        ].get("refit_diagnostics", {}).get(method),
                        "splits": {
                            **run["splits"],
                            "test": refit_records[
                                str(run["bundle"].get("fingerprint"))
                            ]["metrics"][method],
                        },
                    }
                    for run in selected_runs
                ] if method in refit_methods else selected_runs
                for method, selected_runs in selected_by_method.items()
            }

        use_test_bootstrap = bootstrap and all(
            len(method_runs) == 1
            for method_runs in selected_by_method.values()
        )
        bootstrap_records = None
        if use_test_bootstrap:
            bootstrap_records = refit_records or {}
            remaining = {m: rs for m, rs in selected_by_method.items() if m not in refit_methods}
            if remaining:
                for fingerprint, record in _refit_selected_test_predictions(
                    remaining,
                    workers=refit_workers,
                    blas_threads=refit_blas_threads,
                ).items():
                    if fingerprint in bootstrap_records:
                        bootstrap_records[fingerprint]["correct"].update(record["correct"])
                    else:
                        bootstrap_records[fingerprint] = record
        bootstrap_reference_methods = tuple(
            method
            for method in (
                "identity",
                "standardize",
                "whiten_ledoit_wolf",
            )
            if use_test_bootstrap and method in selected_by_method
        )
        bootstrap_rng = np.random.default_rng(bootstrap_seed)

        methods: list[dict[str, Any]] = []
        for method, selected_runs in sorted(selected_by_method.items()):
            # Sort replicas by displayed seed for stable JSON and tables.
            selected_runs.sort(
                key=lambda run: str(
                    run["bundle"].get("split_seed", run["bundle"].get("representation_seed"))
                )
            )
            criterion = _selection_rule_for_method(
                method,
                default_rule=default_rule,
                selection_rules=rules,
            )
            group_ids = sorted(
                {
                    int(group)
                    for run in selected_runs
                    for group in run["splits"]["test"]["per_group"]
                }
            )
            metrics = {
                "accuracy": _mean_interval(
                    (
                        run["splits"]["test"]["accuracy"]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                ),
                "group_balanced_accuracy": _mean_interval(
                    (
                        run["splits"]["test"]["group_balanced_accuracy"]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                ),
                "worst_group_accuracy": _mean_interval(
                    (
                        run["splits"]["test"]["worst_group_accuracy"]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                ),
                "group_balanced_log_loss": _mean_interval(
                    (
                        run["splits"]["test"]["equal_group_log_loss"]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                ),
                "worst_group_log_loss": _mean_interval(
                    (
                        run["splits"]["test"]["worst_group_log_loss"]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                ),
            }
            per_group = {
                str(group): _mean_interval(
                    (
                        run["splits"]["test"]["per_group"][str(group)][
                            "accuracy"
                        ]
                        for run in selected_runs
                    ),
                    confidence=confidence,
                )
                for group in group_ids
            }
            difference_fields: dict[str, Any] = {}
            bootstrap_fields: dict[str, Any] = {}
            if bootstrap_records is not None:
                bootstrap_summaries = _paired_group_metric_bootstraps(
                    bootstrap_records,
                    method=method,
                    reference_methods=bootstrap_reference_methods,
                    confidence=confidence,
                    n_resamples=bootstrap_resamples,
                    rng=bootstrap_rng,
                    progress=(
                        lambda completed, total, method=method: print(
                            (
                                f"\rbootstrap {method}: "
                                f"{completed:,}/{total:,}"
                            ),
                            end="\n" if completed == total else "",
                            flush=True,
                        )
                        if bootstrap_progress
                        else None
                    ),
                )
                # Keep the original identity field stable for existing
                # consumers, and add the explicit standardize contrast when it
                # is available.
                worst_group_bootstraps = bootstrap_summaries[
                    "worst_group_accuracy"
                ]
                bootstrap_fields["paired_test_bootstrap"] = (
                    worst_group_bootstraps["identity"]
                )
                if "standardize" in worst_group_bootstraps:
                    bootstrap_fields["paired_test_bootstrap_standardize"] = (
                        worst_group_bootstraps["standardize"]
                    )
                if len(selected_runs) == 1:
                    difference_fields["paired_differences"] = {
                        reference_method: {
                            metric: metric_summaries[reference_method]
                            for metric, metric_summaries in (
                                bootstrap_summaries.items()
                            )
                        }
                        for reference_method in bootstrap_reference_methods
                    }
                    difference_fields["paired_difference_identity"] = (
                        worst_group_bootstraps["identity"]
                    )
                    if "standardize" in worst_group_bootstraps:
                        difference_fields["paired_difference_standardize"] = (
                            worst_group_bootstraps["standardize"]
                        )
            if len(selected_runs) > 1:
                paired_differences: dict[str, dict[str, Any]] = {}
                for reference_method in (
                    "identity",
                    "standardize",
                    "whiten_ledoit_wolf",
                ):
                    # Compare preprocessing variants within their own method
                    # family, rather than against a differently fitted head.
                    reference_name = _comparison_reference_name(
                        method, reference_method
                    )
                    reference_runs = selected_by_method.get(reference_name)
                    if reference_runs is None:
                        continue
                    paired_differences[reference_name] = {
                        metric: _paired_seed_difference(
                            selected_runs,
                            reference_runs,
                            metric=metric,
                            confidence=confidence,
                        )
                        for metric in PAIRED_DIFFERENCE_METRICS
                    }
                    difference_fields[f"paired_difference_{reference_method}"] = (
                        paired_differences[reference_name][
                            "worst_group_accuracy"
                        ]
                    )
                difference_fields["paired_differences"] = paired_differences
            methods.append(
                {
                    "method": method,
                    "aggregation_mode": STANDARD_AGGREGATION_MODE,
                    "selection_rule": criterion,
                    "selection_split": selected_runs[0]["selection"]["split"],
                    "test_fit_split": (
                        "dfr_val_full_balanced"
                        if method in refit_methods and _is_dfr_method(method)
                        else "val_full_afr_reweighted"
                        if method in refit_methods and _is_afr_method(method)
                        else "val_full"
                        if method in refit_methods and _is_ho_erm_method(method)
                        else "train_val" if method in refit_methods
                        else "val_subset1" if _is_ho_erm_method(method)
                        else "val_subset2" if _is_neurotune_method(method)
                        else "train"
                    ),
                    "replicate_count": len(selected_runs),
                    "selected_ridge_lambdas": [
                        float(run["ridge_lambda"]) for run in selected_runs
                    ],
                    "selected_afr_gammas": [
                        (
                            None
                            if run.get("afr_gamma") is None
                            else float(run["afr_gamma"])
                        )
                        for run in selected_runs
                    ],
                    "selected_neurotune_thresholds": [
                        (
                            None
                            if run.get("neurotune", {}).get("threshold") is None
                            else float(run["neurotune"]["threshold"])
                        )
                        for run in selected_runs
                    ],
                    "selected": [
                        {
                            "bundle": run["bundle"],
                            "ridge_lambda": run["ridge_lambda"],
                            "test_refit": run.get("test_refit"),
                            "afr_gamma": run.get("afr_gamma"),
                            "neurotune_threshold": run.get(
                                "neurotune", {}
                            ).get("threshold"),
                            "validation_metric": _selected_validation_metric(
                                run,
                                criterion,
                            ),
                            **(
                                {
                                    "validation_standard_error": float(
                                        run["selection"]["standard_errors"][
                                            (
                                                "balanced_accuracy"
                                                if criterion
                                                == "class_balanced_accuracy"
                                                else criterion
                                            )
                                        ]
                                    )
                                }
                                if one_se
                                else {}
                            ),
                            **(
                                {
                                    "validation_one_mistake_tolerance": (
                                        1.0
                                        / (
                                            len(run["selection"]["label_counts"])
                                            * min(
                                                int(count)
                                                for count in run["selection"][
                                                    "label_counts"
                                                ].values()
                                            )
                                        )
                                        if criterion
                                        == "class_balanced_accuracy"
                                        else 1.0
                                        / sum(
                                            int(count)
                                            for count in run["selection"][
                                                "label_counts"
                                            ].values()
                                        )
                                    )
                                }
                                if one_mistake
                                else {}
                            ),
                        }
                        for run in selected_runs
                    ],
                    "test": {
                        **metrics,
                        "per_group_accuracy": per_group,
                    },
                    **difference_fields,
                    **bootstrap_fields,
                }
            )

        dataset, representation, class_balanced, dataset_size_fraction = (
            experiment_key
        )
        experiments.append(
            {
                "dataset": dataset,
                "representation": representation,
                "class_balanced": class_balanced,
                "dataset_size_fraction": dataset_size_fraction,
                **(
                    {"sample_size": experiment_runs[0]["sample_size"]}
                    if dataset_size_fraction is not None
                    else {}
                ),
                "aggregation_mode": STANDARD_AGGREGATION_MODE,
                "candidate_ridge_lambdas": sorted(
                    {
                        float(run["ridge_lambda"])
                        for run in experiment_runs
                        if selected_ridges is None
                        or _ridge_is_selected(
                            run["ridge_lambda"], selected_ridges
                        )
                        if max_ridge is None
                        or float(run["ridge_lambda"]) < max_ridge
                    }
                ),
                "candidate_afr_gammas": sorted(
                    {
                        float(run["afr_gamma"])
                        for run in experiment_runs
                        if run.get("afr_gamma") is not None
                        and (
                            selected_afr_gammas is None
                            or _ridge_is_selected(
                                run["afr_gamma"], selected_afr_gammas
                            )
                        )
                        and (
                            max_gamma is None
                            or float(run["afr_gamma"]) < max_gamma
                        )
                    }
                ),
                "candidate_neurotune_thresholds": sorted(
                    {
                        float(run["neurotune"]["threshold"])
                        for run in experiment_runs
                        if run.get("neurotune", {}).get("threshold") is not None
                        and (
                            selected_neurotune_thresholds is None
                            or _ridge_is_selected(
                                run["neurotune"]["threshold"],
                                selected_neurotune_thresholds,
                            )
                        )
                    }
                ),
                "methods": methods,
            }
        )

    has_test_bootstrap = any(
        "paired_test_bootstrap" in method
        for experiment in experiments
        for method in experiment["methods"]
    )
    has_standardize_bootstrap = any(
        "paired_test_bootstrap_standardize" in method
        for experiment in experiments
        for method in experiment["methods"]
    )
    has_ledoit_wolf_bootstrap = any(
        "whiten_ledoit_wolf"
        in method.get("paired_differences", {})
        for experiment in experiments
        for method in experiment["methods"]
    )
    bootstrap_metadata = (
        {
            "enabled": True,
            "metric": "worst_group_accuracy",
            "reference_method": "identity",
            **(
                {
                    "reference_methods": [
                        "identity",
                        *(
                            ["standardize"]
                            if has_standardize_bootstrap
                            else []
                        ),
                        *(
                            ["whiten_ledoit_wolf"]
                            if has_ledoit_wolf_bootstrap
                            else []
                        ),
                    ]
                }
                if has_standardize_bootstrap or has_ledoit_wolf_bootstrap
                else {}
            ),
            "resampling": "paired_within_test_group",
            "n_resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "conditional_on": [
                "embedding_bundle",
                "train_validation_split",
                "validation_selected_ridge_lambda",
                "fitted_representation",
            ],
            "reselects_hyperparameters": False,
            **(
                {"shared_resampling_pass": True}
                if has_standardize_bootstrap
                else {}
            ),
        }
        if has_test_bootstrap
        else {"enabled": False}
    )
    oracle_group_methods = sorted(
        {
            method["method"]
            for experiment in experiments
            for method in experiment["methods"]
            if _is_dfr_method(method["method"])
        }
    )
    selection_splits = {
        method["selection_split"]
        for experiment in experiments
        for method in experiment["methods"]
    }

    return {
        "schema_version": 1,
        "kind": "ridge_lambda_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "split": "val" if selection_splits == {"val"} else "method_specific",
            "oracle_split": "dfr_val" if oracle_group_methods else None,
            "default_rule": default_rule,
            "method_rules": rules,
            "ridge_lambda_subset": (
                list(selected_ridges) if selected_ridges is not None else None
            ),
            "afr_gamma_subset": (
                list(selected_afr_gammas)
                if selected_afr_gammas is not None
                else None
            ),
            "neurotune_threshold_subset": (
                list(selected_neurotune_thresholds)
                if selected_neurotune_thresholds is not None
                else None
            ),
            "max_ridge": max_ridge,
            "max_gamma": max_gamma,
            "single_iteration_filter": {
                "enabled": bool(exclude_single_iteration),
                "definition": "max(final_fit.n_iter) == 1",
                "excluded_run_count": excluded_single_iteration_count,
            },
            "test_refit_train_val": any(
                selected.get("test_refit") is not None
                for experiment in experiments
                for method in experiment["methods"]
                for selected in method["selected"]
            ),
            "refit_train_val_mode": "auto" if refit_train_val is None else refit_train_val,
            "refit_workers": refit_workers,
            "refit_blas_threads": refit_blas_threads,
            "one_standard_error_rule": {
                "enabled": bool(one_se),
                "standard_error_source": "best_validation_candidate",
                "admissible_set": (
                    "score_at_least_best_minus_one_standard_error_for_"
                    "maximization_or_score_at_most_best_plus_one_standard_"
                    "error_for_minimization"
                ),
                "selection": "largest_ridge_lambda_in_admissible_set",
            },
            "one_mistake_rule": {
                "enabled": bool(one_mistake),
                "admissible_set": (
                    "score_at_least_best_minus_one_validation_mistake"
                ),
                "class_balanced_accuracy_tolerance": (
                    "1/(n_classes*minimum_validation_class_count)"
                ),
                "accuracy_tolerance": "1/n_validation",
                "selection": "largest_ridge_lambda_in_admissible_set",
            },
            "tie_break": (
                "largest_ridge_lambda_within_one_standard_error_of_best"
                if one_se
                else "largest_ridge_lambda_within_one_mistake_of_best"
                if one_mistake
                else "larger_ridge_lambda_after_criterion_ties"
            ),
            "class_balanced_accuracy_rule": {
                "selection": (
                    "largest_ridge_lambda_within_one_mistake_of_best"
                    if one_mistake
                    else (
                        "largest_ridge_lambda_after_exact_class_balanced_"
                        "accuracy_ties"
                    )
                ),
            },
            "two_stage_rule": {
                "primary": "class_balanced_accuracy",
                "secondary": "balanced_log_loss",
            },
            "uses_group_labels": bool(oracle_group_methods),
            "oracle_group_selection_methods": oracle_group_methods,
            "test_group_labels_used_for_selection": False,
        },
        "confidence": confidence,
        "aggregation": {
            "mode": STANDARD_AGGREGATION_MODE,
            "selection": "independent_per_seed",
            "test_summary": "mean_of_selected_test_results_across_seeds",
        },
        "paired_differences": {
            "metric": "worst_group_accuracy",
            "metrics": list(PAIRED_DIFFERENCE_METRICS),
            "reference_methods": ["identity", "standardize"],
            "difference": "method_minus_reference",
            "single_representation_interval": "paired_test_bootstrap",
            "multiple_representation_interval": "paired_t_across_seeds",
        },
        "bootstrap": bootstrap_metadata,
        "experiments": experiments,
    }


# =============================================================================
# Terminal table and optional JSON
# =============================================================================

def _format_interval(summary: dict[str, Any]) -> str:
    """Render means consistently, adding a CI only when replicas permit it."""
    mean = float(summary["mean"])
    half_width = summary["confidence_half_width"]
    if half_width is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{float(half_width):.4f}"


def _format_ridge_values(values: Iterable[float]) -> str:
    unique = sorted(set(float(value) for value in values))
    return ",".join(f"{value:.3g}" for value in unique)


def _format_optional_hyperparameter_values(
    values: Iterable[float | None],
) -> str:
    return _format_ridge_values(value for value in values if value is not None)


def _format_paired_difference(summary: dict[str, Any] | None) -> str:
    """Render an observed paired difference and its stored interval."""
    if summary is None:
        return ""
    interval = summary["confidence_interval"]
    return (
        f"{float(summary['observed_difference']):+.4f} "
        f"[{float(interval['lower']):+.4f},{float(interval['upper']):+.4f}]"
    )


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> list[str]:
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


def format_aggregate(report: dict[str, Any]) -> str:
    """Render one selected-test table for each compatible experiment panel."""
    lines: list[str] = []
    confidence_percent = 100 * float(report["confidence"])
    ridge_subset = report.get("selection", {}).get("ridge_lambda_subset")
    neurotune_threshold_subset = report.get("selection", {}).get(
        "neurotune_threshold_subset"
    )
    max_ridge = report.get("selection", {}).get("max_ridge")
    max_gamma = report.get("selection", {}).get("max_gamma")
    for experiment in report["experiments"]:
        methods = experiment["methods"]
        has_afr_gamma = any(
            any(value is not None for value in method.get("selected_afr_gammas", []))
            for method in methods
        )
        has_neurotune_threshold = any(
            any(
                value is not None
                for value in method.get("selected_neurotune_thresholds", [])
            )
            for method in methods
        )
        has_identity_difference = any(
            "paired_difference_identity" in method for method in methods
        )
        has_standardize_difference = any(
            "paired_difference_standardize" in method
            for method in methods
        )
        group_ids = sorted(
            {
                int(group)
                for method in methods
                for group in method["test"]["per_group_accuracy"]
            }
        )
        headers = (
            "method",
            "selection",
            "ridge_lambda",
            *(("afr_gamma",) if has_afr_gamma else ()),
            *(
                ("neurotune_threshold",)
                if has_neurotune_threshold
                else ()
            ),
            "n",
            "equal-group",
            "worst-group",
            *(f"g={group}" for group in group_ids),
            *(() if not has_identity_difference else ("Δworst vs identity",)),
            *(
                ()
                if not has_standardize_difference
                else ("Δworst vs standardize",)
            ),
        )
        rows = []
        for method in methods:
            test = method["test"]
            rows.append(
                (
                    method["method"],
                    method["selection_rule"],
                    _format_ridge_values(
                        method["selected_ridge_lambdas"]
                    ),
                    *((_format_optional_hyperparameter_values(
                        method.get("selected_afr_gammas", [])
                    ),)
                      if has_afr_gamma else ()),
                    *((_format_optional_hyperparameter_values(
                        method.get("selected_neurotune_thresholds", [])
                    ),)
                      if has_neurotune_threshold else ()),
                    str(method["replicate_count"]),
                    _format_interval(test["group_balanced_accuracy"]),
                    _format_interval(test["worst_group_accuracy"]),
                    *(
                        _format_interval(
                            test["per_group_accuracy"][str(group)]
                        )
                        for group in group_ids
                    ),
                    *(
                        ()
                        if not has_identity_difference
                        else (
                            _format_paired_difference(
                                method.get("paired_difference_identity")
                            ),
                        )
                    ),
                    *(
                        ()
                        if not has_standardize_difference
                        else (
                            _format_paired_difference(
                                method.get("paired_difference_standardize")
                            ),
                        )
                    ),
                )
            )
        lines.extend(
            (
                (
                    f"dataset={experiment['dataset']}  "
                    f"representation={experiment['representation']}  "
                    f"class_balanced={experiment['class_balanced']}"
                    + (
                        "  dataset_size="
                        f"{float(experiment['dataset_size_fraction']):g}x "
                        "of original train+val"
                        if experiment.get("dataset_size_fraction") is not None
                        else ""
                    )
                ),
                (
                    "ridge selected on validation per seed; test entries are "
                    "means across selected seeds"
                    + (
                        "; ridge candidates="
                        + _format_ridge_values(ridge_subset)
                        if ridge_subset is not None
                        else ""
                    )
                    + (
                        "; NeuroTune threshold candidates="
                        + _format_ridge_values(neurotune_threshold_subset)
                        if neurotune_threshold_subset is not None
                        else ""
                    )
                    + (
                        f"; ridge values at or above {float(max_ridge):g} excluded"
                        if max_ridge is not None
                        else ""
                    )
                    + (
                        f"; AFR gamma values at or above {float(max_gamma):g} excluded"
                        if max_gamma is not None
                        else ""
                    )
                    + (
                        "; selected heads refit after validation selection"
                        if any(
                            selected.get("test_refit") is not None
                            for method in methods
                            for selected in method.get("selected", [])
                        )
                        else ""
                    )
                    + (
                        f" ± {confidence_percent:.0f}% CI"
                        if any(method["replicate_count"] > 1 for method in methods)
                        else ""
                    )
                ),
                *_render_table(headers, rows),
                "",
            )
        )
    lines.extend(
        (
            "g=* columns are per-group accuracy.",
        )
    )
    if not report.get("selection", {}).get("one_mistake_rule", {}).get(
        "enabled"
    ):
        lines.extend(
            (
                "class_balanced_accuracy uses exact label-balanced accuracy",
                "and only then favors stronger ridge; ordinary methods do not use",
                "group labels for selection.",
            )
        )
    else:
        lines.extend(
            (
                "class_balanced_accuracy uses label-balanced accuracy; ordinary",
                "methods do not use group labels for selection.",
            )
        )
    if report.get("selection", {}).get("one_standard_error_rule", {}).get(
        "enabled"
    ):
        lines.append(
            "one-SE selection keeps candidates within one validation standard"
        )
        lines.append(
            "error of the best score, then selects the largest ridge_lambda."
        )
    if report.get("selection", {}).get("one_mistake_rule", {}).get("enabled"):
        lines.append(
            "one-mistake selection keeps accuracy candidates within one"
        )
        lines.append(
            "validation mistake of the best, then selects the largest ridge_lambda."
        )
    single_iteration_filter = report.get("selection", {}).get(
        "single_iteration_filter", {}
    )
    if single_iteration_filter.get("enabled"):
        lines.append(
            "Single-iteration fits were excluded from validation selection "
            f"({int(single_iteration_filter.get('excluded_run_count', 0))} "
            "saved runs excluded)."
        )
    if report.get("selection", {}).get("oracle_group_selection_methods"):
        lines.append(
            "DFR is an explicit oracle exception: it selects exact best held-out"
        )
        lines.append(
            "worst-group accuracy on dfr_val; test groups remain selection-free."
        )
        if report.get("selection", {}).get("test_refit_train_val"):
            lines.append(
                "Final DFR preprocessing and head both use the same "
                "group-balanced sample from the full validation pool."
            )
    if report.get("bootstrap", {}).get("enabled"):
        lines.append(
            "Δworst is method minus its named reference. Percentile intervals use"
        )
        lines.append(
            "paired test observations resampled within group, conditional on the"
        )
        lines.append(
            "fixed representation, split, and validation-selected head."
        )
    if any(
        "paired_difference_identity" in method
        and method.get("replicate_count", 0) > 1
        for experiment in report["experiments"]
        for method in experiment["methods"]
    ):
        lines.append(
            "For repeated representations, Δworst uses paired t intervals across"
        )
        lines.append("matched representation seeds.")
        if any(
            _is_dfr_method(method["method"])
            for experiment in report["experiments"]
            for method in experiment["methods"]
        ):
            lines.append("For DFR rows, identity and standardize references are DFR identity and DFR standardize.")
    return "\n".join(lines)


def save_aggregate(
    report: dict[str, Any],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save an aggregate JSON, optionally replacing it atomically."""
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite aggregate result: {path}")
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


# =============================================================================
# Command-line overrides
# =============================================================================

def _parse_selection_rule(value: str) -> tuple[str, str]:
    """Parse ``method=criterion`` while keeping error messages actionable."""
    method, separator, criterion = value.partition("=")
    if not separator or not method.strip() or not criterion.strip():
        raise argparse.ArgumentTypeError(
            "selection rules must have the form method=criterion"
        )
    try:
        return method, _validated_criterion(criterion)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_ridge_lambda_list(values: Sequence[str]) -> list[float]:
    """Parse space- or hyphen-separated ridge values from one CLI option."""
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
            continue
        except ValueError:
            pass
        # A hyphen separates compact-list entries unless it is the exponent
        # sign in scientific notation, as in ``1e-05-0.001-0.1``.
        parts = re.split(r"(?<![eE])-", value)
        if len(parts) < 2 or any(not part for part in parts):
            raise argparse.ArgumentTypeError(
                "--ridge-lambda values must be numbers separated by spaces "
                "or hyphens."
            )
        try:
            parsed.extend(float(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --ridge-lambda list: {value!r}."
            ) from exc
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Select ridge penalties on validation and aggregate test results."
    )
    parser.add_argument(
        "--sweep",
        dest="sweep_paths",
        nargs="+",
        help=(
            "Sweep JSON files or directories containing sweep JSON files, "
            "including outputs from finetune_results.py and comparison.py."
        ),
    )
    parser.add_argument(
        "--criterion",
        dest="default_selection_rule",
        help=(
            "Selection rule for ordinary methods and NeuroTune: accuracy, "
            "class_balanced_accuracy, log_loss, balanced_log_loss, "
            "two_stage, or sfit. When omitted, ordinary methods and "
            "NeuroTune use class_balanced_accuracy. DFR and AFR "
            "retain their method-specific group-aware defaults."
        ),
    )
    parser.add_argument(
        "--selection-rule",
        action="append",
        type=_parse_selection_rule,
        help=(
            "Per-method override such as whiten=log_loss or "
            "neurotune=sfit. The neurotune family defaults to "
            "class_balanced_accuracy; an exact method rule such as "
            "neurotune_identity=accuracy takes precedence. May be supplied "
            "more than once."
        ),
    )
    stability_rule = parser.add_mutually_exclusive_group()
    stability_rule.add_argument(
        "--one-se",
        action="store_true",
        default=None,
        help=(
            "Apply the one-standard-error rule: retain validation candidates "
            "within one SE of the best score, then select the largest ridge."
        ),
    )
    stability_rule.add_argument(
        "--one-mistake",
        action="store_true",
        default=None,
        help=(
            "Retain accuracy candidates within one validation mistake of "
            "the best score, then select the largest ridge. For class-balanced "
            "accuracy, one mistake is measured in the smallest class."
        ),
    )
    parser.add_argument(
        "--refit-train-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Default: refit DFR, AFR, and HO-ERM from the full validation "
            "pool; paper-protocol NeuroTune is not refitted. "
            "Use --no-refit-train-val to report saved fits. Explicitly enabling "
            "this flag also refits each selected ordinary "
            "transform and head on train plus validation before test "
            "evaluation. For DFR, fit preprocessing and the head on the same "
            "group-balanced sample from the full validation pool."
        ),
    )
    parser.add_argument(
        "--refit-workers",
        type=int,
        help=(
            "Number of representation seeds refitted concurrently "
            f"(default: {AGGREGATE_CONFIG['refit_workers']})."
        ),
    )
    parser.add_argument(
        "--refit-blas-threads",
        type=int,
        help=(
            "Numerical-library threads per refit worker "
            f"(default: {AGGREGATE_CONFIG['refit_blas_threads']})."
        ),
    )
    parser.add_argument(
        "--ridge-lambda",
        dest="selection_ridge_lambdas",
        nargs="+",
        help=(
            "Saved ridge values eligible for validation selection, supplied "
            "as a space-separated or compact hyphen-separated list."
        ),
    )
    parser.add_argument(
        "--afr-gamma",
        "--afr-gammas",
        dest="selection_afr_gammas",
        nargs="+",
        type=float,
        help=(
            "Saved AFR gamma values eligible for validation selection. "
            "A single value fixes gamma while ridge is still selected normally."
        ),
    )
    parser.add_argument(
        "--neurotune-threshold",
        "--neurotune-thresholds",
        dest="selection_neurotune_thresholds",
        nargs="+",
        type=float,
        help=(
            "Saved NeuroTune threshold values eligible for validation "
            "selection. A single value fixes the threshold while ridge is "
            "still selected normally."
        ),
    )
    parser.add_argument(
        "--max-ridge",
        dest="max_ridge",
        type=float,
        help=(
            "Exclude saved ridge values equal to or above this threshold from "
            "validation selection."
        ),
    )
    parser.add_argument(
        "--max-gamma",
        dest="max_gamma",
        type=float,
        help=(
            "Exclude saved AFR gamma values equal to or above this threshold "
            "from validation selection."
        ),
    )
    parser.add_argument(
        "--exclude-single-iteration",
        action="store_true",
        default=None,
        help=(
            "Exclude saved candidates whose final fit recorded exactly one "
            "solver iteration; fail if a bundle/method has none left."
        ),
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=None,
        help=(
            "Refit selected heads and run a paired, within-group test "
            "bootstrap for equal-group and worst-group "
            "accuracy versus identity and, when available, standardize."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        dest="bootstrap_resamples",
        type=int,
        help="Number of paired test bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help="Random seed for paired test bootstrap resampling (default: 0).",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Atomically replace an existing aggregate JSON output.",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    """Combine editable defaults with explicit command-line values."""
    config = {
        **AGGREGATE_CONFIG,
        "sweep_paths": list(AGGREGATE_CONFIG["sweep_paths"]),
        "selection_rules": dict(AGGREGATE_CONFIG["selection_rules"]),
    }
    for key in (
        "sweep_paths",
        "default_selection_rule",
        "selection_ridge_lambdas",
        "selection_afr_gammas",
        "selection_neurotune_thresholds",
        "max_ridge",
        "max_gamma",
        "exclude_single_iteration",
        "one_se",
        "one_mistake",
        "refit_train_val",
        "refit_workers",
        "refit_blas_threads",
        "confidence",
        "bootstrap",
        "bootstrap_resamples",
        "bootstrap_seed",
        "output",
        "overwrite",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    # A CLI --criterion is an explicit request for one group-free criterion
    # across ordinary methods and NeuroTune.
    if args.default_selection_rule is not None:
        config["selection_rules"]["neurotune"] = (
            args.default_selection_rule
        )
    if args.selection_rule:
        config["selection_rules"].update(dict(args.selection_rule))
    if args.selection_ridge_lambdas is not None:
        config["selection_ridge_lambdas"] = _parse_ridge_lambda_list(
            args.selection_ridge_lambdas
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = resolved_config(parse_args(argv))
    runs, source_files = load_sweep_runs(config["sweep_paths"])
    report = aggregate_runs(
        runs,
        default_selection_rule=config["default_selection_rule"],
        selection_rules=config["selection_rules"],
        selection_ridge_lambdas=config["selection_ridge_lambdas"],
        selection_afr_gammas=config["selection_afr_gammas"],
        selection_neurotune_thresholds=config[
            "selection_neurotune_thresholds"
        ],
        max_ridge=config["max_ridge"],
        max_gamma=config["max_gamma"],
        exclude_single_iteration=bool(config["exclude_single_iteration"]),
        one_se=bool(config["one_se"]),
        one_mistake=bool(config["one_mistake"]),
        refit_train_val=config["refit_train_val"],
        confidence=float(config["confidence"]),
        bootstrap=bool(config["bootstrap"]),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_progress=bool(config["bootstrap"]),
        refit_progress=config["refit_train_val"] is not False,
        refit_workers=int(config["refit_workers"]),
        refit_blas_threads=int(config["refit_blas_threads"]),
    )
    report["source_files"] = source_files
    print(format_aggregate(report))
    if config["output"] is not None:
        saved = save_aggregate(
            report,
            config["output"],
            overwrite=bool(config["overwrite"]),
        )
        print(
            f"\nSaved: {saved}"
        )


if __name__ == "__main__":
    main()
