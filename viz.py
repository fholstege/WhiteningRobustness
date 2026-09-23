#!/usr/bin/env python3
"""Create the main-text and appendix empirical comparison figures.

Each figure has two panels. The left panel shows mean held-out performance for
no preprocessing, standardization, and whitening. The right panel shows the two
paired whitening effects with confidence intervals and two-sided paired-test
significance markers. Nonlinear Ledoit--Wolf whitening is the default.

Run both manuscript presets::

    python viz.py

Or render one preset, explicit sweep files, or existing aggregate reports::

    python viz.py --figure main
    python viz.py --group-accuracy-panels --output plots/group_accuracy
    python viz.py --sweep results_v2/WB_resnet50_finetune.json \
        --criterion class_balanced_accuracy --exclude-single-iteration
    python viz.py report_1.json report_2.json report_3.json --output plots/custom
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import stats

from aggregate import aggregate_runs, load_sweep_runs
from style import (
    PAPER_BLUE,
    PAPER_GREEN,
    PAPER_GREY,
    PAPER_ORANGE,
    PAPER_RED,
    PLOT_DATASET_TICK_FONT_SIZE,
    PLOT_LABEL_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_ANNOTATION_FONT_SIZE,
    PLOT_BAR_ANNOTATION_FONT_SIZE,
    TEXT_GREY,
    configure_plot_style,
    method_label,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "plots"
DEFAULT_METRIC = "worst_group_accuracy"
DEFAULT_CONFIDENCE = 0.95
VALUE_ANNOTATION_FONT_SIZE = PLOT_ANNOTATION_FONT_SIZE
BAR_VALUE_ANNOTATION_FONT_SIZE = PLOT_BAR_ANNOTATION_FONT_SIZE
EMPIRICAL_LEGEND_FONT_SIZE = PLOT_LEGEND_FONT_SIZE
HSPACE_DIFF = 0.15
HSPACE_LEGEND = -0.05
METHODS = ("identity", "standardize", "whiten")
BASELINES = ("identity", "standardize")
WHITENING_COMPARISON_METHODS = ("whiten", "whiten_ledoit_wolf_nonlinear")
WHITENING_METHODS = {
    "ledoit-wolf-nonlinear": "whiten_ledoit_wolf_nonlinear",
    "empirical": "whiten",
    "ledoit-wolf": "whiten_ledoit_wolf",
}
WHITENING_LABELS = {
    "whiten_ledoit_wolf_nonlinear": "Nonlinear Ledoit-Wolf",
    "whiten": "Empirical whitening",
    "whiten_ledoit_wolf": "Ledoit-Wolf whitening",
}
WHITENING_COMPARISON_LABELS = {
    "whiten": "Empirical whitening",
    "whiten_ledoit_wolf_nonlinear": "Whitening with the nonlinear shrinkage estimator",
}
WHITENING_COMPARISON_COLORS = {
    "whiten": PAPER_ORANGE,
    "whiten_ledoit_wolf_nonlinear": PAPER_RED,
}
WHITENING_COMPARISON_EFFECT_COLOR = PAPER_ORANGE
DEFAULT_WHITENING_ESTIMATOR = "ledoit-wolf-nonlinear"
DEFAULT_WHITENING_METHOD = WHITENING_METHODS[DEFAULT_WHITENING_ESTIMATOR]
FIGURE_CHOICES = ("main", "appendix", "both")
METRIC_CHOICES = (
    "accuracy",
    "group_balanced_accuracy",
    "worst_group_accuracy",
)
SELECTION_CHOICES = (
    "accuracy",
    "class_balanced_accuracy",
    "log_loss",
    "balanced_log_loss",
    "two_stage",
)

# These are deliberately editable manuscript presets rather than an experiment
# framework. They point to sweeps, not previously aggregated reports, so every
# figure applies the criterion and filters requested at plotting time.
FIGURE_SWEEPS: Mapping[str, tuple[str, ...]] = {
    "main": (
        "results_paper/WB_resnet50_finetune.json",
        "results_paper/CelebA_resnet50_finetune.json",
        "results_paper/multiNLI_BERT_finetune.json",
    ),
    "appendix": (
        "results_paper/WB_dino_vitb16_finetune.json",
        "results_paper/CelebA_dino_vitb16_finetune.json",
        "results_paper/multiNLI_debertav3_finetune.json",
    ),
}

METHOD_LABELS = {method: method_label(method) for method in METHODS}
METHOD_COLORS = {
    "identity": PAPER_BLUE,
    "standardize": PAPER_GREEN,
    "whiten": PAPER_RED,
}
CONTRAST_MARKERS = {
    "identity": "o",
    "standardize": "s",
}
CONTRAST_COLORS = {
    "identity": PAPER_BLUE,
    "standardize": PAPER_GREEN,
}
DATASET_LABELS = {
    "WB": "Waterbirds",
    "CelebA": "CelebA",
    "multiNLI": "MultiNLI",
}
REPRESENTATION_LABELS = {
    "resnet50": "ResNet-50",
    "bert": "BERT",
    "dino_vitb16": "DINO ViT-B/16",
    "debertav3": "DeBERTa-v3",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "clip_openai_vitb16": "CLIP ViT-B/16",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "group_balanced_accuracy": "Equal-group accuracy",
    "worst_group_accuracy": "Worst-group accuracy",
}


@dataclass(frozen=True)
class Estimate:
    """A mean and confidence interval across matched replicates."""

    mean: float
    lower: float
    upper: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class Effect:
    """A paired whitening-minus-baseline effect."""

    observed: float
    lower: float
    upper: float
    p_value: float
    replicate_count: int

    @property
    def significant(self) -> bool:
        """Whether the two-sided paired test rejects at five percent."""
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


@dataclass(frozen=True)
class EmpiricalRow:
    """Plot-ready results for one dataset and representation."""

    dataset: str
    representation: str
    estimates: Mapping[str, Estimate]
    effects: Mapping[str, Effect]


@dataclass(frozen=True)
class WhiteningComparisonRow:
    """Matched empirical and nonlinear-whitening results for one dataset."""

    dataset: str
    representation: str
    estimates: Mapping[str, Estimate]
    effect: Effect


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_report(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and minimally validate one ``aggregate.py`` report."""
    source = _resolved_path(path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("kind") != "ridge_lambda_aggregate":
        raise ValueError(f"Not an aggregate.py report: {source}")
    if report.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported aggregate schema {report.get('schema_version')!r}: "
            f"{source}"
        )
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(
            "Each empirical figure input must contain aggregate experiments: "
            f"{source}"
        )
    return report, source


def aggregate_sweeps(
    paths: Sequence[str | Path],
    *,
    criterion: str,
    exclude_single_iteration: bool,
    confidence: float,
) -> dict[str, Any]:
    """Aggregate selected sweep files in memory for immediate plotting."""
    runs, source_files = load_sweep_runs(paths)
    report = aggregate_runs(
        runs,
        default_selection_rule=criterion,
        exclude_single_iteration=exclude_single_iteration,
        confidence=confidence,
    )
    report["source_files"] = source_files
    return report


def _method_map(experiment: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    methods = experiment.get("methods")
    if not isinstance(methods, list):
        raise TypeError("Each aggregate experiment must contain a method list.")
    result = {str(method.get("method")): method for method in methods}
    if len(result) != len(methods):
        raise ValueError("An aggregate experiment contains duplicate methods.")
    return result


def _metric_values(method: Mapping[str, Any], metric: str) -> np.ndarray:
    try:
        summary = method["test"][metric]
    except KeyError as exc:
        available = ", ".join(sorted(method.get("test", {})))
        raise ValueError(
            f"Metric {metric!r} is unavailable for method "
            f"{method.get('method')!r}. Available metrics: {available}."
        ) from exc
    values = np.asarray(summary.get("values"), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"Malformed {metric!r} values for method {method.get('method')!r}."
        )
    stored_mean = float(summary["mean"])
    if not np.isfinite(stored_mean) or not np.isclose(
        stored_mean, values.mean(), atol=1e-12
    ):
        raise ValueError(
            f"Inconsistent {metric!r} mean for method "
            f"{method.get('method')!r}."
        )
    return values


def _replicate_keys(method: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    selected = method.get("selected")
    if not isinstance(selected, list):
        raise ValueError(
            f"Method {method.get('method')!r} has no selected replicate records."
        )
    keys = []
    for entry in selected:
        bundle = entry.get("bundle", {})
        fingerprint = bundle.get("fingerprint")
        if fingerprint is None:
            raise ValueError("A selected bundle is missing its fingerprint.")
        keys.append((str(bundle.get("representation_seed")), str(fingerprint)))
    return tuple(keys)


def _mean_estimate(values: np.ndarray, confidence: float) -> Estimate:
    mean = float(values.mean())
    if values.size == 1:
        lower = upper = mean
    else:
        half_width = float(
            stats.sem(values)
            * stats.t.ppf((1.0 + confidence) / 2.0, values.size - 1)
        )
        lower = mean - half_width
        upper = mean + half_width
    return Estimate(
        mean=mean,
        lower=lower,
        upper=upper,
        values=tuple(float(value) for value in values),
    )


def _paired_effect(
    whitening: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    metric: str,
    confidence: float,
) -> Effect:
    whitening_values = _metric_values(whitening, metric)
    reference_values = _metric_values(reference, metric)
    if whitening_values.shape != reference_values.shape:
        raise ValueError("Paired effects require matching replicate counts.")
    if _replicate_keys(whitening) != _replicate_keys(reference):
        raise ValueError(
            "Paired effects require identical representation seeds and bundle "
            "fingerprints in the same order."
        )
    differences = whitening_values - reference_values
    observed = float(differences.mean())
    if differences.size == 1:
        lower = upper = observed
        p_value = float("nan")
    elif np.allclose(differences, differences[0], rtol=0.0, atol=1e-15):
        lower = upper = observed
        p_value = 1.0 if np.isclose(observed, 0.0, atol=1e-15) else 0.0
    else:
        half_width = float(
            stats.sem(differences)
            * stats.t.ppf((1.0 + confidence) / 2.0, differences.size - 1)
        )
        lower = observed - half_width
        upper = observed + half_width
        p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)
    return Effect(
        observed=observed,
        lower=lower,
        upper=upper,
        p_value=p_value,
        replicate_count=int(differences.size),
    )


def prepare_row(
    report: Mapping[str, Any],
    *,
    metric: str = DEFAULT_METRIC,
    whitening_method: str = DEFAULT_WHITENING_METHOD,
    confidence: float = DEFAULT_CONFIDENCE,
) -> EmpiricalRow:
    """Convert one aggregate experiment into means and paired effects."""
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Unsupported empirical metric {metric!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise ValueError("Each report must contain exactly one experiment.")
    experiment = experiments[0]
    methods = _method_map(experiment)
    requested = {
        "identity": "identity",
        "standardize": "standardize",
        "whiten": whitening_method,
    }
    missing = [stored for stored in requested.values() if stored not in methods]
    if missing:
        raise ValueError(
            f"representation={experiment.get('representation')!r} is missing "
            f"method(s): {', '.join(missing)}."
        )
    selected = {name: methods[stored] for name, stored in requested.items()}
    keys = [_replicate_keys(selected[method]) for method in METHODS]
    if any(key != keys[0] for key in keys[1:]):
        raise ValueError(
            "Identity, standardization, and whitening must use the same matched "
            "replicates."
        )
    estimates = {
        method: _mean_estimate(_metric_values(selected[method], metric), confidence)
        for method in METHODS
    }
    effects = {
        baseline: _paired_effect(
            selected["whiten"],
            selected[baseline],
            metric=metric,
            confidence=confidence,
        )
        for baseline in BASELINES
    }
    return EmpiricalRow(
        dataset=str(experiment.get("dataset")),
        representation=str(experiment.get("representation")),
        estimates=estimates,
        effects=effects,
    )


def prepare_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    metric: str = DEFAULT_METRIC,
    whitening_method: str = DEFAULT_WHITENING_METHOD,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[EmpiricalRow]:
    """Prepare distinct datasets in the supplied manuscript order."""
    if not reports:
        raise ValueError("At least one aggregate report is required.")
    singleton_reports = [
        {**report, "experiments": [experiment]}
        for report in reports
        for experiment in report.get("experiments", [])
    ]
    rows = [
        prepare_row(
            report,
            metric=metric,
            whitening_method=whitening_method,
            confidence=confidence,
        )
        for report in singleton_reports
    ]
    datasets = [row.dataset for row in rows]
    if len(set(datasets)) != len(datasets):
        raise ValueError("Each figure must contain distinct datasets.")
    dataset_order = {
        dataset: index for index, dataset in enumerate(DATASET_LABELS)
    }
    original_order = {id(row): index for index, row in enumerate(rows)}
    rows.sort(
        key=lambda row: (
            dataset_order.get(row.dataset, len(dataset_order)),
            original_order[id(row)],
        )
    )
    return rows


def prepare_whitening_comparison_row(
    report: Mapping[str, Any],
    *,
    metric: str = DEFAULT_METRIC,
    confidence: float = DEFAULT_CONFIDENCE,
) -> WhiteningComparisonRow:
    """Compare empirical and nonlinear Ledoit--Wolf whitening on matched seeds."""
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Unsupported empirical metric {metric!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise ValueError("Each report must contain exactly one experiment.")
    experiment = experiments[0]
    methods = _method_map(experiment)
    missing = [
        method for method in WHITENING_COMPARISON_METHODS if method not in methods
    ]
    if missing:
        raise ValueError(
            f"representation={experiment.get('representation')!r} is missing "
            f"method(s): {', '.join(missing)}."
        )
    selected = {
        method: methods[method] for method in WHITENING_COMPARISON_METHODS
    }
    if _replicate_keys(selected["whiten"]) != _replicate_keys(
        selected["whiten_ledoit_wolf_nonlinear"]
    ):
        raise ValueError(
            "Empirical and nonlinear whitening must use the same matched replicates."
        )
    estimates = {
        method: _mean_estimate(_metric_values(selected[method], metric), confidence)
        for method in WHITENING_COMPARISON_METHODS
    }
    effect = _paired_effect(
        selected["whiten_ledoit_wolf_nonlinear"],
        selected["whiten"],
        metric=metric,
        confidence=confidence,
    )
    return WhiteningComparisonRow(
        dataset=str(experiment.get("dataset")),
        representation=str(experiment.get("representation")),
        estimates=estimates,
        effect=effect,
    )


def prepare_whitening_comparison_rows(
    reports: Sequence[Mapping[str, Any]],
    *,
    metric: str = DEFAULT_METRIC,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[WhiteningComparisonRow]:
    """Prepare the empirical-versus-nonlinear comparison in dataset order."""
    if not reports:
        raise ValueError("At least one aggregate report is required.")
    singleton_reports = [
        {**report, "experiments": [experiment]}
        for report in reports
        for experiment in report.get("experiments", [])
    ]
    rows = [
        prepare_whitening_comparison_row(
            report,
            metric=metric,
            confidence=confidence,
        )
        for report in singleton_reports
    ]
    datasets = [row.dataset for row in rows]
    if len(set(datasets)) != len(datasets):
        raise ValueError("Each figure must contain distinct datasets.")
    dataset_order = {
        dataset: index for index, dataset in enumerate(DATASET_LABELS)
    }
    original_order = {id(row): index for index, row in enumerate(rows)}
    rows.sort(
        key=lambda row: (
            dataset_order.get(row.dataset, len(dataset_order)),
            original_order[id(row)],
        )
    )
    return rows


def _row_label(row: EmpiricalRow) -> str:
    dataset = DATASET_LABELS.get(row.dataset, row.dataset)
    representation = REPRESENTATION_LABELS.get(
        row.representation, row.representation
    )
    return f"{dataset}\n{representation}"


def _dataset_label(row: EmpiricalRow) -> str:
    return DATASET_LABELS.get(row.dataset, row.dataset)


def _whitening_label(whitening_method: str) -> str:
    try:
        return WHITENING_LABELS[whitening_method]
    except KeyError as exc:
        raise ValueError(f"Unknown whitening method {whitening_method!r}.") from exc


def _contrast_label(baseline: str) -> str:
    if baseline == "identity":
        return f"Whitening − {method_label(baseline)}"
    if baseline == "standardize":
        return "Whitening − Standardization"
    raise ValueError(f"Unknown contrast baseline {baseline!r}.")


def _draw_mean_panel(
    axis: Any,
    rows: Sequence[EmpiricalRow],
    *,
    metric: str,
) -> None:
    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.23
    offsets = (-width, 0.0, width)
    for method, offset in zip(METHODS, offsets):
        estimates = [row.estimates[method] for row in rows]
        means = 100.0 * np.asarray([estimate.mean for estimate in estimates])
        lower = 100.0 * np.asarray(
            [estimate.mean - estimate.lower for estimate in estimates]
        )
        upper = 100.0 * np.asarray(
            [estimate.upper - estimate.mean for estimate in estimates]
        )
        confidence_tops = means + upper
        bars = axis.bar(
            positions + offset,
            means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            yerr=np.vstack((lower, upper)),
            error_kw={
                "ecolor": TEXT_GREY,
                "elinewidth": 0.9,
                "capsize": 2.5,
                "capthick": 0.9,
            },
            zorder=3,
        )
        for bar, mean, confidence_top in zip(bars, means, confidence_tops):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                confidence_top + 1.2,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_ANNOTATION_FONT_SIZE,
                color=TEXT_GREY,
                zorder=4,
            )
    all_upper = [
        100.0 * estimate.upper
        for row in rows
        for estimate in row.estimates.values()
    ]
    axis.set_ylim(0.0, min(100.0, max(all_upper) + 10.0))
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_dataset_label(row) for row in rows], fontsize=PLOT_DATASET_TICK_FONT_SIZE
    )
    axis.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
    axis.xaxis.grid(False)


def _draw_effect_panel(
    axis: Any,
    rows: Sequence[EmpiricalRow],
    *,
    metric: str,
    confidence: float,
) -> None:
    positions = np.arange(len(rows) - 1, -1, -1, dtype=np.float64)
    offsets = {"identity": 0.13, "standardize": -0.13}
    bounds = [0.0]
    for baseline in BASELINES:
        for row, y in zip(rows, positions):
            effect = row.effects[baseline]
            observed = 100.0 * effect.observed
            lower = 100.0 * effect.lower
            upper = 100.0 * effect.upper
            bounds.extend((lower, upper))
            axis.errorbar(
                observed,
                y + offsets[baseline],
                xerr=np.asarray([[observed - lower], [upper - observed]]),
                fmt=CONTRAST_MARKERS[baseline],
                markersize=6.0,
                markerfacecolor=CONTRAST_COLORS[baseline],
                markeredgecolor=CONTRAST_COLORS[baseline],
                markeredgewidth=1.2,
                ecolor=CONTRAST_COLORS[baseline],
                elinewidth=1.2,
                capsize=3.0,
                capthick=1.2,
                zorder=3,
            )
            axis.annotate(
                f"{observed:+.1f}",
                xy=(observed, y + offsets[baseline]),
                xytext=(0, -9) if baseline == "standardize" else (0, 7),
                textcoords="offset points",
                ha="center",
                va="top" if baseline == "standardize" else "bottom",
                fontsize=VALUE_ANNOTATION_FONT_SIZE,
                color=CONTRAST_COLORS[baseline],
                zorder=4,
            )
    span = max(bounds) - min(bounds)
    padding = max(0.8, 0.12 * span)
    axis.set_xlim(min(bounds) - padding, max(bounds) + 1.5 * padding)
    axis.axvline(0.0, color=PAPER_GREY, linestyle="--", linewidth=0.9, zorder=1)
    axis.set_ylim(-0.5, len(rows) - 0.5)
    axis.set_yticks(positions)
    # The encoder is already printed below the matching bar group in panel (a).
    # Repeating only the dataset here keeps long appendix encoder names from
    # crowding the space between panels.
    axis.set_yticklabels(
        [_dataset_label(row) for row in rows], fontsize=PLOT_DATASET_TICK_FONT_SIZE
    )
    axis.set_xlabel(f"Difference in {METRIC_LABELS[metric].lower()} (%)")
    axis.yaxis.grid(False)
    axis.xaxis.grid(True)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.spines["left"].set_visible(False)


def make_figure(
    rows: Sequence[EmpiricalRow],
    *,
    metric: str = DEFAULT_METRIC,
    confidence: float = DEFAULT_CONFIDENCE,
    whitening_method: str = DEFAULT_WHITENING_METHOD,
    equal_group_rows: Sequence[EmpiricalRow] | None = None,
) -> Any:
    """Draw means and paired effects, optionally in two metric columns."""
    if not rows:
        raise ValueError("Cannot plot an empty empirical row list.")
    configure_plot_style()
    figure, grid = plt.subplots(
        2 if equal_group_rows is not None else 1,
        2,
        figsize=(11.2, 8.0 if equal_group_rows is not None else 5.5),
        squeeze=False,
    )
    axes = (grid[0, 0], grid[1, 0]) if equal_group_rows is not None else grid[0]
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.94 if equal_group_rows is not None else 0.86,
        bottom=0.25 if equal_group_rows is not None else 0.31,
        wspace=0.36,
        hspace=HSPACE_DIFF,
    )
    _draw_mean_panel(axes[0], rows, metric=metric)
    _draw_effect_panel(
        axes[1],
        rows,
        metric=metric,
        confidence=confidence,
    )
    if equal_group_rows is not None:
        _draw_mean_panel(grid[0, 1], equal_group_rows, metric="group_balanced_accuracy")
        _draw_effect_panel(
            grid[1, 1], equal_group_rows,
            metric="group_balanced_accuracy", confidence=confidence,
        )
        for axis in grid[0]:
            axis.set_ylim(0.0, 105.0)
            axis.set_yticks(np.arange(0, 101, 20))
            axis.set_ylabel(axis.get_ylabel(), fontsize=PLOT_LABEL_FONT_SIZE)
            axis.set_xticklabels([
                _dataset_label(row)
                for row in rows
            ], fontsize=PLOT_DATASET_TICK_FONT_SIZE)
        grid[1, 1].tick_params(axis="y", labelleft=False)
        for axis, panel_metric in zip(grid[1], (metric, "group_balanced_accuracy")):
            axis.set_xlabel(
                f"Difference in {METRIC_LABELS[panel_metric].lower()} (%)",
                fontsize=PLOT_LABEL_FONT_SIZE,
            )
        for axis in grid.flat:
            axis.tick_params(axis="y", labelsize=PLOT_TICK_FONT_SIZE)
        for axis in grid[1]:
            axis.tick_params(axis="x", labelsize=PLOT_TICK_FONT_SIZE)
    method_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    contrast_handles = [
        Line2D(
            [0],
            [0],
            marker=CONTRAST_MARKERS[baseline],
            color=CONTRAST_COLORS[baseline],
            markerfacecolor=CONTRAST_COLORS[baseline],
            markeredgewidth=1.2,
            linewidth=1.2,
            label=_contrast_label(baseline),
        )
        for baseline in BASELINES
    ]
    # Matplotlib fills multi-column legends down each column.  Interleave the
    # handles so the visual first row is the three method boxes and the second
    # row is the two paired-effect lines.
    legend_handles = (
        method_handles[0],
        contrast_handles[0],
        method_handles[1],
        contrast_handles[1],
        method_handles[2],
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025 - HSPACE_LEGEND),
        ncol=3,
        fontsize=EMPIRICAL_LEGEND_FONT_SIZE,
        handletextpad=0.55,
        columnspacing=1.25,
    )
    return figure


def make_whitening_comparison_figure(
    rows: Sequence[WhiteningComparisonRow],
    *,
    metric: str = DEFAULT_METRIC,
    confidence: float = DEFAULT_CONFIDENCE,
    equal_group_rows: Sequence[WhiteningComparisonRow] | None = None,
    axes: Any = None,
    include_legend: bool = True,
) -> Any:
    """Draw empirical versus nonlinear whitening and their paired difference."""
    if not rows:
        raise ValueError("Cannot plot an empty whitening comparison.")
    configure_plot_style()
    if equal_group_rows is not None:
        figure, grid = plt.subplots(2, 2, figsize=(11.2, 8.0))
        figure.subplots_adjust(left=0.075, right=0.985, top=0.94,
                               bottom=0.25, wspace=0.36, hspace=HSPACE_DIFF)
        make_whitening_comparison_figure(
            rows, metric=metric, confidence=confidence,
            axes=(grid[0, 0], grid[1, 0]),
        )
        make_whitening_comparison_figure(
            equal_group_rows, metric="group_balanced_accuracy", confidence=confidence,
            axes=(grid[0, 1], grid[1, 1]), include_legend=False,
        )
        for axis in grid[0]:
            axis.set_ylim(0, 105)
            axis.set_yticks(np.arange(0, 101, 20))
            axis.set_ylabel(axis.get_ylabel(), fontsize=PLOT_LABEL_FONT_SIZE)
            axis.set_xticklabels([
                _dataset_label(row)
                for row in rows
            ], fontsize=PLOT_DATASET_TICK_FONT_SIZE)
        grid[1, 1].tick_params(axis="y", labelleft=False)
        for axis, panel_metric in zip(grid[1], (metric, "group_balanced_accuracy")):
            axis.set_xlabel(
                f"Difference in {METRIC_LABELS[panel_metric].lower()}",
                fontsize=PLOT_LABEL_FONT_SIZE,
            )
            axis.tick_params(axis="x", labelsize=PLOT_TICK_FONT_SIZE)
        for axis in grid.flat:
            axis.tick_params(axis="y", labelsize=PLOT_TICK_FONT_SIZE)
        return figure
    if axes is None:
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.5),
                                   gridspec_kw={"width_ratios": (1.16, 1.0)})
        figure.subplots_adjust(left=0.075, right=0.985, top=0.86,
                               bottom=0.31, wspace=0.30)
    else:
        figure = axes[0].figure

    positions = np.arange(len(rows), dtype=np.float64)
    width = 0.30
    offsets = (-width / 2.0, width / 2.0)
    for method, offset in zip(WHITENING_COMPARISON_METHODS, offsets):
        estimates = [row.estimates[method] for row in rows]
        means = 100.0 * np.asarray([estimate.mean for estimate in estimates])
        lower = 100.0 * np.asarray(
            [estimate.mean - estimate.lower for estimate in estimates]
        )
        upper = 100.0 * np.asarray(
            [estimate.upper - estimate.mean for estimate in estimates]
        )
        bars = axes[0].bar(
            positions + offset,
            means,
            width=width,
            color=WHITENING_COMPARISON_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            yerr=np.vstack((lower, upper)),
            error_kw={
                "ecolor": TEXT_GREY,
                "elinewidth": 0.9,
                "capsize": 2.5,
                "capthick": 0.9,
            },
            zorder=3,
        )
        for bar, mean, confidence_top in zip(bars, means, means + upper):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                confidence_top + 1.2,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_ANNOTATION_FONT_SIZE,
                color=TEXT_GREY,
                zorder=4,
            )
    all_upper = [
        100.0 * estimate.upper
        for row in rows
        for estimate in row.estimates.values()
    ]
    axes[0].set_ylim(0.0, min(100.0, max(all_upper) + 10.0))
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [_dataset_label(row) for row in rows], fontsize=PLOT_DATASET_TICK_FONT_SIZE
    )
    axes[0].set_ylabel(f"{METRIC_LABELS[metric]} (%)")
    axes[0].xaxis.grid(False)

    effect_positions = np.arange(len(rows) - 1, -1, -1, dtype=np.float64)
    bounds = [0.0]
    for row, y in zip(rows, effect_positions):
        observed = 100.0 * row.effect.observed
        lower = 100.0 * row.effect.lower
        upper = 100.0 * row.effect.upper
        bounds.extend((lower, upper))
        axes[1].errorbar(
            observed,
            y,
            xerr=np.asarray([[observed - lower], [upper - observed]]),
            fmt="o",
            markersize=6.0,
            markerfacecolor=WHITENING_COMPARISON_EFFECT_COLOR,
            markeredgecolor=WHITENING_COMPARISON_EFFECT_COLOR,
            markeredgewidth=1.2,
            ecolor=WHITENING_COMPARISON_EFFECT_COLOR,
            elinewidth=1.2,
            capsize=3.0,
            capthick=1.2,
            zorder=3,
        )
        axes[1].annotate(
            f"{observed:+.1f}",
            xy=(observed, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_ANNOTATION_FONT_SIZE,
            color=WHITENING_COMPARISON_EFFECT_COLOR,
            zorder=4,
        )
    span = max(bounds) - min(bounds)
    padding = max(0.8, 0.12 * span)
    axes[1].set_xlim(min(bounds) - padding, max(bounds) + 1.5 * padding)
    axes[1].axvline(
        0.0, color=PAPER_GREY, linestyle="--", linewidth=0.9, zorder=1
    )
    axes[1].set_ylim(-0.5, len(rows) - 0.5)
    axes[1].set_yticks(effect_positions)
    axes[1].set_yticklabels(
        [_dataset_label(row) for row in rows], fontsize=PLOT_DATASET_TICK_FONT_SIZE
    )
    axes[1].set_xlabel(f"Difference in {METRIC_LABELS[metric].lower()}")
    axes[1].yaxis.grid(False)
    axes[1].xaxis.grid(True)
    axes[1].tick_params(axis="y", length=0, pad=8)
    axes[1].spines["left"].set_visible(False)

    method_handles = [
        Patch(
            facecolor=WHITENING_COMPARISON_COLORS[method],
            edgecolor="white",
            label=WHITENING_COMPARISON_LABELS[method],
        )
        for method in WHITENING_COMPARISON_METHODS
    ]
    effect_handle = Line2D(
        [0],
        [0],
        marker="o",
        color=WHITENING_COMPARISON_EFFECT_COLOR,
        markerfacecolor=WHITENING_COMPARISON_EFFECT_COLOR,
        markeredgewidth=1.2,
        linewidth=1.2,
        label="Nonlinear shrinkage estimator − empirical",
    )
    if not include_legend:
        return figure
    figure.legend(
        handles=method_handles + [effect_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025 - HSPACE_LEGEND),
        ncol=2,
        fontsize=EMPIRICAL_LEGEND_FONT_SIZE,
        handletextpad=0.55,
        columnspacing=1.25,
    )
    return figure


def _format_whitening_comparison_summary(
    rows: Sequence[WhiteningComparisonRow],
    *,
    metric: str,
) -> str:
    lines = [f"{METRIC_LABELS[metric]}; empirical versus nonlinear whitening"]
    for row in rows:
        means = ", ".join(
            f"{WHITENING_COMPARISON_LABELS[method]}="
            f"{100 * row.estimates[method].mean:.2f}"
            for method in WHITENING_COMPARISON_METHODS
        )
        effect = row.effect
        p_value = (
            "n/a" if not np.isfinite(effect.p_value) else f"{effect.p_value:.4g}"
        )
        lines.append(f"  {_row_label(row).replace(chr(10), ' / ')}: {means}")
        lines.append(
            "    Nonlinear Ledoit–Wolf - empirical: "
            f"{100 * effect.observed:+.2f} pp "
            f"[{100 * effect.lower:+.2f}, {100 * effect.upper:+.2f}], "
            f"p={p_value}, n={effect.replicate_count}"
        )
    return "\n".join(lines)


def _format_summary(
    rows: Sequence[EmpiricalRow],
    *,
    metric: str,
    whitening_method: str,
) -> str:
    lines = [
        f"{METRIC_LABELS[metric]}; "
        f"whitening estimator={_whitening_label(whitening_method)}"
    ]
    for row in rows:
        means = ", ".join(
            f"{METHOD_LABELS[method]}={100 * row.estimates[method].mean:.2f}"
            for method in METHODS
        )
        lines.append(f"  {_row_label(row).replace(chr(10), ' / ')}: {means}")
        for baseline in BASELINES:
            effect = row.effects[baseline]
            p_value = (
                "n/a"
                if not np.isfinite(effect.p_value)
                else f"{effect.p_value:.4g}"
            )
            lines.append(
                f"    {_contrast_label(baseline)}: "
                f"{100 * effect.observed:+.2f} pp "
                f"[{100 * effect.lower:+.2f}, {100 * effect.upper:+.2f}], "
                f"p={p_value}, n={effect.replicate_count}"
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two-panel empirical figures with method means and paired "
            "whitening effects."
        )
    )
    parser.add_argument(
        "reports",
        nargs="*",
        help=(
            "Optional aggregate JSON files for one custom figure. When omitted, "
            "the selected manuscript preset is used."
        ),
    )
    parser.add_argument(
        "--sweep",
        dest="sweep_paths",
        nargs="+",
        help=(
            "Ridge-sweep JSON files to aggregate in memory before plotting. "
            "May include multiple datasets."
        ),
    )
    parser.add_argument(
        "--criterion",
        choices=SELECTION_CHOICES,
        default="class_balanced_accuracy",
        help=(
            "Validation criterion applied to sweep inputs "
            "(default: class_balanced_accuracy)."
        ),
    )
    parser.add_argument(
        "--exclude-single-iteration",
        action="store_true",
        help=(
            "Before selection, exclude candidates whose final solver fit used "
            "exactly one iteration."
        ),
    )
    parser.add_argument(
        "--figure",
        choices=FIGURE_CHOICES,
        default="both",
        help="Manuscript preset to render when reports are omitted (default: both).",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default=DEFAULT_METRIC,
        help=f"Held-out metric to plot (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--group-accuracy-panels", action="store_true",
        help="Plot worst-group and equal-group accuracy in a 2x2 layout.",
    )
    parser.add_argument(
        "--compare-whitening-estimators",
        action="store_true",
        help=(
            "Compare empirical whitening with nonlinear Ledoit--Wolf whitening "
            "and show their paired difference."
        ),
    )
    whitening_choice = parser.add_mutually_exclusive_group()
    whitening_choice.add_argument(
        "--whitening-estimator",
        choices=tuple(WHITENING_METHODS),
        default=None,
        help=(
            "Whitening estimator used as c: empirical, ledoit-wolf, or "
            "ledoit-wolf-nonlinear (default: ledoit-wolf-nonlinear)."
        ),
    )
    whitening_choice.add_argument(
        "--whitening-method",
        choices=tuple(WHITENING_LABELS),
        default=None,
        help=(
            "Stored method-name compatibility alias. Prefer "
            "--whitening-estimator."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Two-sided confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output stem for a custom or single-preset figure. With --figure "
            "both, '_main' and '_appendix' are appended."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png",),
        help="Output formats (default: png).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figure files.",
    )
    args = parser.parse_args(argv)
    if args.reports and args.sweep_paths:
        parser.error("Aggregate reports and --sweep inputs are mutually exclusive.")
    if args.reports and args.figure != "both":
        parser.error("--figure cannot be combined with explicit reports.")
    if args.sweep_paths and args.figure != "both":
        parser.error("--figure cannot be combined with explicit --sweep inputs.")
    if not 0.0 < float(args.confidence) < 1.0:
        parser.error("--confidence must lie strictly between zero and one.")
    if args.compare_whitening_estimators and (
        args.whitening_estimator is not None or args.whitening_method is not None
    ):
        parser.error(
            "--compare-whitening-estimators cannot be combined with a single "
            "whitening estimator."
        )
    if args.whitening_estimator is None and args.whitening_method is None:
        args.whitening_estimator = DEFAULT_WHITENING_ESTIMATOR
        args.whitening_method = WHITENING_METHODS[args.whitening_estimator]
    elif args.whitening_estimator is not None:
        args.whitening_method = WHITENING_METHODS[args.whitening_estimator]
    else:
        args.whitening_estimator = next(
            estimator
            for estimator, method in WHITENING_METHODS.items()
            if method == args.whitening_method
        )
    return args


def _preset_names(figure: str) -> tuple[str, ...]:
    if figure == "both":
        return ("main", "appendix")
    return (figure,)


def _output_stem(
    args: argparse.Namespace,
    name: str,
    *,
    custom: bool,
    input_paths: Sequence[str | Path],
) -> Path:
    if args.output:
        stem = _resolved_path(args.output)
        if not custom and args.figure == "both":
            stem = stem.with_name(f"{stem.name}_{name}")
        return stem
    if args.group_accuracy_panels:
        estimator = ("whitening_estimators" if args.compare_whitening_estimators
                     else args.whitening_estimator.replace("-", "_"))
        return DEFAULT_OUTPUT_DIR / f"{estimator}_{name}_group_accuracy"
    if custom:
        input_label = (
            Path(input_paths[0]).stem
            if len(input_paths) == 1
            else "empirical_custom"
        )
        if args.compare_whitening_estimators:
            return DEFAULT_OUTPUT_DIR / (
                f"{input_label}_whitening_estimators_"
                f"{args.criterion}_{args.metric}"
            )
        estimator = args.whitening_estimator.replace("-", "_")
        return DEFAULT_OUTPUT_DIR / (
            f"{input_label}_{estimator}_"
            f"{args.criterion}_{args.metric}"
        )
    if args.compare_whitening_estimators:
        return DEFAULT_OUTPUT_DIR / f"whitening_estimators_{name}_{args.metric}"
    estimator = args.whitening_estimator.replace("-", "_")
    return DEFAULT_OUTPUT_DIR / f"{estimator}_{name}_{args.metric}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.group_accuracy_panels and args.metric != DEFAULT_METRIC:
        raise ValueError("--group-accuracy-panels requires the default metric.")
    custom = bool(args.reports or args.sweep_paths)
    if args.reports:
        jobs = (("custom", "reports", tuple(args.reports)),)
    elif args.sweep_paths:
        jobs = (("custom", "sweeps", tuple(args.sweep_paths)),)
    else:
        jobs = tuple(
            (name, "sweeps", FIGURE_SWEEPS[name])
            for name in _preset_names(args.figure)
        )
    saved: list[Path] = []
    summaries: list[str] = []
    for name, input_kind, input_paths in jobs:
        if input_kind == "sweeps":
            reports = [
                aggregate_sweeps(
                    input_paths,
                    criterion=args.criterion,
                    exclude_single_iteration=bool(
                        args.exclude_single_iteration
                    ),
                    confidence=float(args.confidence),
                )
            ]
        else:
            reports = [load_report(path)[0] for path in input_paths]
        if args.compare_whitening_estimators:
            rows = prepare_whitening_comparison_rows(
                reports,
                metric=args.metric,
                confidence=float(args.confidence),
            )
            figure = make_whitening_comparison_figure(
                rows,
                metric=args.metric,
                confidence=float(args.confidence),
                equal_group_rows=(prepare_whitening_comparison_rows(
                    reports, metric="group_balanced_accuracy",
                    confidence=float(args.confidence),
                ) if args.group_accuracy_panels else None),
            )
        else:
            rows = prepare_rows(
                reports,
                metric=args.metric,
                whitening_method=args.whitening_method,
                confidence=float(args.confidence),
            )
            figure = make_figure(
                rows,
                metric=args.metric,
                confidence=float(args.confidence),
                whitening_method=args.whitening_method,
                equal_group_rows=(prepare_rows(
                    reports, metric="group_balanced_accuracy",
                    whitening_method=args.whitening_method,
                    confidence=float(args.confidence),
                ) if args.group_accuracy_panels else None),
            )
        try:
            saved.extend(
                save_figure(
                    figure,
                    _output_stem(
                        args,
                        name,
                        custom=custom,
                        input_paths=input_paths,
                    ),
                    formats=args.formats,
                    overwrite=bool(args.overwrite),
                )
            )
        finally:
            plt.close(figure)
        comparison = (
            "empirical versus nonlinear Ledoit--Wolf"
            if args.compare_whitening_estimators
            else f"whitening_estimator={args.whitening_estimator}"
        )
        provenance = (
            f"criterion={args.criterion}; "
            "exclude_single_iteration="
            f"{bool(args.exclude_single_iteration)}; {comparison}"
            if input_kind == "sweeps"
            else "pre-aggregated input"
        )
        if args.compare_whitening_estimators:
            formatted_summary = _format_whitening_comparison_summary(
                rows,
                metric=args.metric,
            )
        else:
            formatted_summary = _format_summary(
                rows,
                metric=args.metric,
                whitening_method=args.whitening_method,
            )
        summaries.append(
            f"{name.title()} ({provenance})\n"
            + formatted_summary
        )
    print("\n\n".join(summaries))
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
