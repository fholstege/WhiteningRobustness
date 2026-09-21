#!/usr/bin/env python3
"""Plot whitening effects for DFR, AFR, and NeuroTune comparisons.

The three side-by-side panels show paired changes from nonlinear Ledoit--Wolf
whitening relative to the identity and standardization variants of each
method. Hyperparameters are selected independently for every method,
transformation, and representation seed using ``aggregate.py``.

Run the manuscript comparison preset::

    python viz_comparison.py

Use ``--no-refit-train-val`` for a quick plot of the test results already
stored in the sweep files.
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
    PAPER_RED,
    TEXT_GREY,
    configure_plot_style,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results_paper" / "comparisons"
DEFAULT_OUTPUT = ROOT / "plots" / "comparison_worst_group_accuracy"
DEFAULT_METRIC = "worst_group_accuracy"
DEFAULT_CONFIDENCE = 0.95
VALUE_ANNOTATION_FONT_SIZE = 12
COMPARISON_ANNOTATTION_FONT_SIZE = 10
BAR_VALUE_ANNOTATION_FONT_SIZE = 9
COMPARISON_LEGEND_FONT_SIZE = 14
HSPACE_DIFF = 0.15
HSPACE_LEGEND = 0.0
BAR_METHODS = ("identity", "standardize", "whiten")
BAR_LABELS = {
    "identity": "None",
    "standardize": "Standardization",
    "whiten": "Whitening",
}
BAR_COLORS = {
    "identity": PAPER_BLUE,
    "standardize": PAPER_GREEN,
    "whiten": PAPER_RED
}
METRIC_CHOICES = (
    "accuracy",
    "group_balanced_accuracy",
    "worst_group_accuracy",
)
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "group_balanced_accuracy": "Equal-group accuracy",
    "worst_group_accuracy": "Worst-group accuracy",
}
BASELINES = ("identity", "standardize")
BASELINE_LABELS = {
    "identity": "Whitening − None",
    "standardize": "Whitening − Standardization",
}
BASELINE_COLORS = {
    "identity": PAPER_BLUE,
    "standardize": PAPER_GREEN,
}
BASELINE_MARKERS = {
    "identity": "o",
    "standardize": "s",
}


@dataclass(frozen=True)
class MethodSpec:
    """File and stored-method naming for one comparison family."""

    key: str
    label: str
    file_token: str
    stored_prefix: str


@dataclass(frozen=True)
class DatasetSpec:
    """File and display naming for one dataset."""

    key: str
    label: str
    file_stem: str


@dataclass(frozen=True)
class Effect:
    """A paired whitening-minus-baseline effect across representation seeds."""

    observed: float
    lower: float
    upper: float
    p_value: float
    replicate_count: int


@dataclass(frozen=True)
class Estimate:
    """A mean and confidence interval across representation seeds."""

    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ComparisonPanel:
    """Method means and whitening contrasts for one comparison family."""

    method: MethodSpec
    estimates: Mapping[str, Mapping[str, Estimate]]
    effects: Mapping[str, Mapping[str, Effect]]


METHOD_SPECS = (
    MethodSpec("dfr", "DFR", "DFR", "dfr"),
    MethodSpec("afr", "AFR", "AFR", "afr"),
    MethodSpec("nt", "NT", "NEUROTUNE", "neurotune"),
)
DATASET_SPECS = (
    DatasetSpec("WB", "Waterbirds", "WB_resnet50"),
    DatasetSpec("CelebA", "CelebA", "CelebA_resnet50"),
    DatasetSpec("multiNLI", "MultiNLI", "multiNLI_BERT"),
)


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def comparison_paths(
    results_dir: str | Path,
    *,
    use_aggregates: bool = False,
    neurotune_selection: str = "class-balanced",
) -> dict[tuple[str, str], Path]:
    """Return the required input paths in method-then-dataset order."""
    directory = _resolved_path(results_dir)
    if neurotune_selection not in {"sfit", "class-balanced"}:
        raise ValueError(
            "neurotune_selection must be 'sfit' or 'class-balanced'."
        )
    paths = {}
    for method in METHOD_SPECS:
        for dataset in DATASET_SPECS:
            suffix = "_aggregate" if use_aggregates else ""
            if (
                use_aggregates
                and method.key == "nt"
                and neurotune_selection == "class-balanced"
            ):
                suffix += "_cb"
            paths[(method.key, dataset.key)] = directory / (
                f"{dataset.file_stem}_{method.file_token}{suffix}.json"
            )
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Missing comparison input file(s):\n" + formatted
        )
    return paths


def aggregate_comparisons(
    paths: Mapping[tuple[str, str], Path],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    refit_train_val: bool | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Select each comparison sweep and aggregate its held-out results."""
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for key, path in paths.items():
        if path.stem.endswith(("_aggregate", "_aggregate_cb")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("kind") != "ridge_lambda_aggregate":
                raise ValueError(f"Not an aggregate.py report: {path}")
            if report.get("schema_version") != 1:
                raise ValueError(
                    f"Unsupported aggregate schema "
                    f"{report.get('schema_version')!r}: {path}"
                )
            stored_confidence = float(report.get("confidence", float("nan")))
            if not np.isclose(stored_confidence, confidence):
                raise ValueError(
                    f"Aggregate confidence is {stored_confidence:g} in {path}; "
                    f"rerun aggregate.py with --confidence {confidence:g}."
                )
            # ``aggregate.py`` records the protocol-specific held-out test
            # result in every aggregate.  Ordinary comparison methods may
            # refit their selected head using the full validation pool, while
            # the NeuroTune protocol deliberately evaluates the fit from its
            # second validation subset.  Consequently, a missing
            # ``test_refit`` is expected for a valid NeuroTune aggregate and
            # must not prevent plotting its stored held-out results.
            reports[key] = report
            continue
        runs, source_files = load_sweep_runs([path])
        report = aggregate_runs(
            runs,
            confidence=confidence,
            refit_train_val=refit_train_val,
        )
        report["source_files"] = source_files
        reports[key] = report
    return reports


def _method_map(experiment: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    methods = experiment.get("methods")
    if not isinstance(methods, list):
        raise ValueError("Each aggregate experiment must contain methods.")
    result = {str(method.get("method")): method for method in methods}
    if len(result) != len(methods):
        raise ValueError("An aggregate experiment contains duplicate methods.")
    return result


def _metric_values(method: Mapping[str, Any], metric: str) -> np.ndarray:
    try:
        summary = method["test"][metric]
    except KeyError as exc:
        raise ValueError(
            f"Metric {metric!r} is unavailable for method "
            f"{method.get('method')!r}."
        ) from exc
    values = np.asarray(summary.get("values"), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"Malformed {metric!r} values for method {method.get('method')!r}."
        )
    return values


def _replicate_keys(method: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    selected = method.get("selected")
    if not isinstance(selected, list):
        raise ValueError(
            f"Method {method.get('method')!r} has no selected records."
        )
    keys = []
    for entry in selected:
        bundle = entry.get("bundle", {})
        fingerprint = bundle.get("fingerprint")
        if fingerprint is None:
            raise ValueError("A selected bundle is missing its fingerprint.")
        keys.append((str(bundle.get("representation_seed")), str(fingerprint)))
    return tuple(keys)


def _paired_effect(
    whitening: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    metric: str,
    confidence: float,
) -> Effect:
    whitening_values = _metric_values(whitening, metric)
    baseline_values = _metric_values(baseline, metric)
    if whitening_values.shape != baseline_values.shape:
        raise ValueError("Paired effects require matching replicate counts.")
    if _replicate_keys(whitening) != _replicate_keys(baseline):
        raise ValueError(
            "Paired effects require identical representation seeds and bundle "
            "fingerprints in the same order."
        )
    differences = whitening_values - baseline_values
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


def _mean_estimate(
    method: Mapping[str, Any], *, metric: str, confidence: float
) -> Estimate:
    """Calculate the displayed mean interval from the stored replicates."""
    values = _metric_values(method, metric)
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
    return Estimate(mean=mean, lower=lower, upper=upper)


def prepare_panels(
    reports: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    metric: str = DEFAULT_METRIC,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[ComparisonPanel]:
    """Extract matched nonlinear-whitening effects in manuscript order."""
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Unsupported comparison metric {metric!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")
    panels = []
    for method in METHOD_SPECS:
        dataset_estimates: dict[str, dict[str, Estimate]] = {}
        dataset_effects: dict[str, dict[str, Effect]] = {}
        for dataset in DATASET_SPECS:
            try:
                report = reports[(method.key, dataset.key)]
            except KeyError as exc:
                raise ValueError(
                    f"Missing aggregate report for {method.label} on "
                    f"{dataset.label}."
                ) from exc
            experiments = report.get("experiments")
            if not isinstance(experiments, list) or len(experiments) != 1:
                raise ValueError(
                    "Each comparison report must contain exactly one experiment."
                )
            experiment = experiments[0]
            if str(experiment.get("dataset")) != dataset.key:
                raise ValueError(
                    f"Expected dataset {dataset.key!r}, found "
                    f"{experiment.get('dataset')!r}."
                )
            methods = _method_map(experiment)
            whitening_name = (
                f"{method.stored_prefix}_whiten_ledoit_wolf_nonlinear"
            )
            required = [
                whitening_name,
                *(f"{method.stored_prefix}_{name}" for name in BASELINES),
            ]
            missing = [name for name in required if name not in methods]
            if missing:
                raise ValueError(
                    f"{method.label} on {dataset.label} is missing method(s): "
                    + ", ".join(missing)
                )
            whitening = methods[whitening_name]
            selected = {
                "identity": methods[f"{method.stored_prefix}_identity"],
                "standardize": methods[f"{method.stored_prefix}_standardize"],
                "whiten": whitening,
            }
            replicate_keys = [_replicate_keys(selected[name]) for name in BAR_METHODS]
            if any(keys != replicate_keys[0] for keys in replicate_keys[1:]):
                raise ValueError(
                    "Identity, standardization, and whitening must use identical "
                    "representation seeds and bundle fingerprints."
                )
            dataset_estimates[dataset.key] = {
                name: _mean_estimate(
                    selected[name], metric=metric, confidence=confidence
                )
                for name in BAR_METHODS
            }
            dataset_effects[dataset.key] = {
                baseline: _paired_effect(
                    whitening,
                    methods[f"{method.stored_prefix}_{baseline}"],
                    metric=metric,
                    confidence=confidence,
                )
                for baseline in BASELINES
            }
        panels.append(
            ComparisonPanel(
                method=method,
                estimates=dataset_estimates,
                effects=dataset_effects,
            )
        )
    return panels


def make_figure(
    panels: Sequence[ComparisonPanel],
    *,
    metric: str = DEFAULT_METRIC,
) -> Any:
    """Draw method accuracies above paired whitening differences for each family."""
    if [panel.method.key for panel in panels] != [
        method.key for method in METHOD_SPECS
    ]:
        raise ValueError(
            "Comparison panels must follow the configured method order."
        )
    configure_plot_style()
    figure, axes = plt.subplots(
        2, len(METHOD_SPECS), figsize=(12.2, 8.5), squeeze=False
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.90,
        bottom=0.20,
        wspace=0.13,
        hspace=HSPACE_DIFF,
    )

    positions = np.arange(len(DATASET_SPECS) - 1, -1, -1, dtype=np.float64)
    offsets = {"identity": 0.13, "standardize": -0.13}
    all_accuracy_upper = [
        100.0 * estimate.upper
        for panel in panels
        for dataset in DATASET_SPECS
        for estimate in panel.estimates[dataset.key].values()
    ]
    accuracy_top = min(100.0, max(all_accuracy_upper) + 8.0)

    for column, panel in enumerate(panels):
        mean_axis = axes[0, column]
        positions_for_bars = np.arange(len(DATASET_SPECS), dtype=np.float64)
        width = 0.23
        for offset, name in zip((-width, 0.0, width), BAR_METHODS):
            estimates = [panel.estimates[dataset.key][name] for dataset in DATASET_SPECS]
            means = 100.0 * np.asarray([estimate.mean for estimate in estimates])
            lower = 100.0 * np.asarray([estimate.mean - estimate.lower for estimate in estimates])
            upper = 100.0 * np.asarray([estimate.upper - estimate.mean for estimate in estimates])
            bars = mean_axis.bar(
                positions_for_bars + offset, means, width=width,
                color=BAR_COLORS[name], edgecolor="white", linewidth=0.7,
                yerr=np.vstack((lower, upper)),
                error_kw={"ecolor": TEXT_GREY, "elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9},
                zorder=3,
            )
            for bar, mean, estimate in zip(bars, means, estimates):
                mean_axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    100.0 * estimate.upper + 1.0, f"{mean:.1f}",
                    ha="center", va="bottom",
                    fontsize=BAR_VALUE_ANNOTATION_FONT_SIZE, color=TEXT_GREY,
                    zorder=4,
                )
        mean_axis.set_title(panel.method.label)
        mean_axis.set_ylim(0.0, accuracy_top)
        mean_axis.set_xticks(positions_for_bars)
        mean_axis.set_xticklabels(
            [dataset.label for dataset in DATASET_SPECS]
        )
        mean_axis.xaxis.grid(False)
        if column == 0:
            mean_axis.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        else:
            mean_axis.tick_params(axis="y", labelleft=False)

        axis = axes[1, column]
        bounds = [0.0]
        for baseline in BASELINES:
            for dataset, y in zip(DATASET_SPECS, positions):
                effect = panel.effects[dataset.key][baseline]
                observed = 100.0 * effect.observed
                lower = 100.0 * effect.lower
                upper = 100.0 * effect.upper
                bounds.extend((lower, upper))
                axis.errorbar(
                    observed,
                    y + offsets[baseline],
                    xerr=np.asarray([[observed - lower], [upper - observed]]),
                    fmt=BASELINE_MARKERS[baseline],
                    markersize=6.0,
                    markerfacecolor=BASELINE_COLORS[baseline],
                    markeredgecolor=BASELINE_COLORS[baseline],
                    markeredgewidth=1.2,
                    ecolor=BASELINE_COLORS[baseline],
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
                    color=BASELINE_COLORS[baseline],
                    zorder=4,
                )
        axis.axvline(
            0.0,
            color=PAPER_GREY,
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        span = max(bounds) - min(bounds)
        padding = max(0.8, 0.12 * span)
        axis.set_xlim(min(bounds) - padding, max(bounds) + 1.5 * padding)
        axis.set_ylim(-0.5, len(DATASET_SPECS) - 0.5)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [dataset.label for dataset in DATASET_SPECS] if column == 0 else []
        )
        axis.yaxis.grid(False)
        axis.xaxis.grid(True)
        axis.tick_params(axis="y", length=0, pad=8)
        axis.spines["left"].set_visible(False)

    figure.supxlabel(
        f"Difference in {METRIC_LABELS[metric].lower()} (%)",
        y=0.125,
        fontsize=16,
    )
    
    bar_handles = [
        Patch(facecolor=BAR_COLORS[name], edgecolor="white", label=BAR_LABELS[name])
        for name in BAR_METHODS
    ]
    contrast_handles = [
        Line2D(
            [0],
            [0],
            marker=BASELINE_MARKERS[baseline],
            color=BASELINE_COLORS[baseline],
            markerfacecolor=BASELINE_COLORS[baseline],
            markeredgewidth=1.2,
            linewidth=1.2,
            label=BASELINE_LABELS[baseline],
        )
        for baseline in BASELINES
    ]
    figure.legend(
        handles=(
            bar_handles[0], contrast_handles[0], bar_handles[1],
            contrast_handles[1], bar_handles[2],
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025 - HSPACE_LEGEND),
        ncol=3,
        fontsize=COMPARISON_LEGEND_FONT_SIZE,
        handletextpad=0.55,
        columnspacing=1.25,
    )
    return figure


def format_summary(
    panels: Sequence[ComparisonPanel],
    *,
    metric: str,
    confidence: float,
) -> str:
    """Format the plotted estimates for terminal provenance."""
    lines = [
        f"{METRIC_LABELS[metric]}; nonlinear Ledoit--Wolf whitening; "
        f"{100.0 * confidence:.0f}% paired t intervals"
    ]
    for panel in panels:
        lines.append(panel.method.label)
        for dataset in DATASET_SPECS:
            contrasts = []
            for baseline in BASELINES:
                effect = panel.effects[dataset.key][baseline]
                contrasts.append(
                    f"vs {baseline}: {100 * effect.observed:+.2f} pp "
                    f"[{100 * effect.lower:+.2f}, {100 * effect.upper:+.2f}]"
                )
            lines.append(f"  {dataset.label}: " + "; ".join(contrasts))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired nonlinear-whitening effects for DFR, AFR, and NT."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing the comparison sweep JSON files.",
    )
    parser.add_argument(
        "--use-aggregates",
        action="store_true",
        help=(
            "Read precomputed '*_aggregate.json' reports from --results-dir "
            "instead of selecting and refitting the sweep files again."
        ),
    )
    parser.add_argument(
        "--neurotune-selection",
        choices=("class-balanced", "sfit"),
        default="class-balanced",
        help=(
            "NeuroTune aggregate to plot: 'class-balanced' uses the "
            "*_NEUROTUNE_aggregate_cb.json reports used by the manuscript; "
            "'sfit' uses the paper-metric *_NEUROTUNE_aggregate.json reports "
            "(default: class-balanced)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default=DEFAULT_METRIC,
        help=f"Held-out metric to plot (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Two-sided paired confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--refit-train-val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Refit selected heads before plotting. The default uses aggregate.py "
            "method-specific behavior; --no-refit-train-val uses stored tests."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output figure stem.",
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
    if not 0.0 < float(args.confidence) < 1.0:
        parser.error("--confidence must lie strictly between zero and one.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = comparison_paths(
        args.results_dir,
        use_aggregates=bool(args.use_aggregates),
        neurotune_selection=str(args.neurotune_selection),
    )
    reports = aggregate_comparisons(
        paths,
        confidence=float(args.confidence),
        refit_train_val=args.refit_train_val,
    )
    panels = prepare_panels(
        reports,
        metric=args.metric,
        confidence=float(args.confidence),
    )
    figure = make_figure(panels, metric=args.metric)
    try:
        saved = save_figure(
            figure,
            _resolved_path(args.output),
            formats=args.formats,
            overwrite=bool(args.overwrite),
        )
    finally:
        plt.close(figure)
    print(format_summary(panels, metric=args.metric, confidence=args.confidence))
    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
