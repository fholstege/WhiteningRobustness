#!/usr/bin/env python3
"""Create the manuscript figures from a synthetic sweep JSON report.

Run::

    python viz_synthetic.py results/synthetic_sweep.json

This writes decision boundaries and sample-size-specific performance curves to
``plots/synthetic``. Coefficient confirmations for Theorem 2 and Proposition 2
are written only when the report contains the corresponding validation rows.
Performance figures include ERM, standardization, and whitening by default.

Run ``synthetic_sweep.py`` to write all editable gamma values to one report,
without theorem/proposition checks. Then generate all figures with::

    python viz_synthetic.py results/synthetic_gamma_sweep.json

This writes a 3-by-2 figure: gamma varies by row, with overall accuracy on the
left and worst-group accuracy on the right, each plotted against q/n.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from style import (
    PAPER_BLUE,
    PAPER_GREEN,
    PAPER_ORANGE,
    PAPER_PURPLE,
    PAPER_RED,
    PAPER_GREY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    TEXT_GREY,
    configure_plot_style,
    method_label,
    method_style,
    save_figure,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "synthetic"
# Fraction of each panel's simulation-average/Theorem-limit range added to
# both y-axis ends. Increase this single value for more vertical whitespace.
THEOREM2_Y_PADDING_FRACTION = 0.5
# Proposition 2 is plotted in a tightly concentrated regime, so the broad
# convergence-plot padding used for Theorem 2 makes its y-axis unnecessarily
# tall.
PROPOSITION2_Y_PADDING_FRACTION = 0.02


def load_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a complete synthetic sweep report."""
    source = Path(path).expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("kind") != "synthetic_whitening_estimator_sweep":
        raise ValueError(f"Not a synthetic sweep report: {source}")
    if report.get("status") != "complete":
        raise ValueError(f"Synthetic sweep is not complete: {source}")
    if int(report.get("schema_version", 0)) < 10:
        raise ValueError(
            "The report predates the separate Theorem 2 and Proposition 2 "
            "validation sweeps. "
            "Rerun synthetic_sweep.py before plotting."
        )
    if not report.get("results"):
        raise ValueError("Synthetic sweep contains no result rows.")
    samples = report.get("plot_data", {}).get("decision_boundary_samples")
    if not samples:
        raise ValueError(
            "Synthetic sweep contains no decision-boundary samples. "
            "Rerun synthetic_sweep.py with plot_sample_size > 0."
        )
    return report


def _mean_and_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite plot values.")
    if array.size == 1:
        return float(array[0]), 0.0
    half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
    return float(array.mean()), float(half_width)


def _available_methods(report: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(str(row["method"]) for row in report["results"]))


def _method_rows(
    report: Mapping[str, Any],
    *,
    condition_id: int,
    method: str,
) -> list[Mapping[str, Any]]:
    """Return one method's rows at a condition, ordered by simulation."""
    rows = [
        row
        for row in report["results"]
        if int(row["condition_id"]) == condition_id
        and row["method"] == method
    ]
    rows.sort(key=lambda row: int(row["simulation"]))
    if not rows:
        raise ValueError(
            f"No results exist for condition={condition_id}, method={method!r}."
        )
    simulations = [int(row["simulation"]) for row in rows]
    if len(simulations) != len(set(simulations)):
        raise ValueError(
            f"Duplicate simulations exist for condition={condition_id}, "
            f"method={method!r}."
        )
    return rows


def _boundary_condition_id(
    report: Mapping[str, Any],
    *,
    boundary_ratio: float | None,
    boundary_condition_id: int | None,
) -> int:
    """Resolve an exact swept d/n ratio or an explicit condition ID."""
    if boundary_ratio is not None and boundary_condition_id is not None:
        raise ValueError(
            "Pass either boundary_ratio or boundary_condition_id, not both."
        )
    if boundary_ratio is None:
        if boundary_condition_id is not None:
            return int(boundary_condition_id)
        return int(report["plot_data"]["boundary_condition_id"])

    requested = float(boundary_ratio)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("boundary_ratio must be a positive finite number.")
    conditions = report.get("conditions", [])
    matches = [
        condition
        for condition in conditions
        if np.isclose(
            float(condition["realized_d_n_ratio"]),
            requested,
            rtol=1e-12,
            atol=1e-12,
        )
    ]
    if len(matches) != 1:
        available = ", ".join(
            f"{float(condition['realized_d_n_ratio']):g}"
            for condition in conditions
        )
        raise ValueError(
            f"No unique swept condition has d/n={requested:g}. "
            f"Available ratios: {available}."
        )
    return int(matches[0]["condition_id"])


def _boundary_sample(
    report: Mapping[str, Any],
    *,
    condition_id: int,
    simulation: int,
) -> Mapping[str, Any]:
    matches = [
        sample
        for sample in report["plot_data"]["decision_boundary_samples"]
        if int(sample["condition_id"]) == condition_id
        and int(sample["simulation"]) == simulation
    ]
    if len(matches) != 1:
        raise ValueError(
            "No unique stored boundary sample for "
            f"condition={condition_id}, simulation={simulation}."
        )
    return matches[0]


def _coefficient_summary(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    coefficients = row["original_coordinate_coefficients"]
    intercept = np.asarray(coefficients["intercept"], dtype=np.float64).reshape(-1)
    if intercept.size != 1:
        raise ValueError("Decision-boundary plotting requires a binary head.")
    return (
        float(coefficients["core"]),
        float(coefficients["spurious"]),
        float(intercept[0]),
        float(coefficients["noise_norm"]),
    )


def _coefficient_summaries(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [_coefficient_summary(row) for row in rows], dtype=np.float64
    )
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Decision-boundary coefficients are malformed.")
    if np.any(values[:, 3] <= 0.0):
        raise ValueError("Decision-boundary noise norms must be positive.")
    return values


def _boundary_title(method: str) -> str:
    if method == "identity":
        return "ERM"
    if method in {
        "whiten",
        "whiten_ledoit_wolf",
        "whiten_ledoit_wolf_nonlinear",
    }:
        return "ERM + Whitening"
    return f"ERM + {method_label(method)}"


def _draw_boundary_panel(
    axis: Any,
    *,
    coordinates: np.ndarray,
    labels: np.ndarray,
    coefficient_summaries: np.ndarray,
    point_predictions: np.ndarray,
    noise_coordinate_scale: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    if not np.isfinite(noise_coordinate_scale) or noise_coordinate_scale <= 0.0:
        raise ValueError("Decision-boundary noise coordinate scale must be positive.")
    if point_predictions.shape != labels.shape:
        raise ValueError("Full-model boundary predictions do not match sampled labels.")
    if np.all(np.abs(coefficient_summaries[:, :2]) <= 1e-12):
        raise ValueError("The first two decision-boundary coefficients are zero.")

    core_grid = np.linspace(*x_limits, 240)
    spurious_grid = np.linspace(*y_limits, 240)
    grid_core, grid_spurious = np.meshgrid(core_grid, spurious_grid)
    grid_linear_score = (
        coefficient_summaries[:, 0, None, None] * grid_core
        + coefficient_summaries[:, 1, None, None] * grid_spurious
        + coefficient_summaries[:, 2, None, None]
    )
    # Marginalize each fitted full-dimensional head over the independent
    # Gaussian noise coordinates rather than plotting its two-coordinate
    # projection. This is the prediction probability at (x_c, x_s).
    grid_probability = ndtr(
        grid_linear_score
        / (noise_coordinate_scale * coefficient_summaries[:, 3, None, None])
    )
    grid_probability = grid_probability.mean(axis=0)
    prediction_cmap = LinearSegmentedColormap.from_list(
        "class_predictions", [PAPER_BLUE, "#ffffff", PAPER_RED]
    )
    axis.imshow(
        grid_probability,
        origin="lower",
        extent=(*x_limits, *y_limits),
        aspect="auto",
        interpolation="bilinear",
        cmap=prediction_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.75,
        zorder=0,
    )

    point_colors = np.where(point_predictions, PAPER_RED, PAPER_BLUE)
    for label, marker in ((0, "o"), (1, "s")):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=24,
            marker=marker,
            c=point_colors[mask],
            edgecolors="white",
            linewidths=0.65,
            alpha=0.9,
            zorder=4,
        )
    axis.grid(color="white", linestyle=":", alpha=0.7)
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)


def plot_decision_boundaries(
    report: Mapping[str, Any],
    *,
    condition_id: int,
    simulation: int,
    unwhitened_method: str,
    whitened_method: str,
) -> Any:
    """Plot the two fitted test prediction surfaces in a 1-by-2 layout."""
    sample = _boundary_sample(
        report, condition_id=condition_id, simulation=simulation
    )
    payload = sample["test"]
    display_coordinates = np.asarray(
        payload["first_two_coordinates"], dtype=np.float64
    )
    selected_rows = {
        str(row["method"]): row for row in report["results"]
        if int(row["condition_id"]) == condition_id
        and int(row["simulation"]) == simulation
    }
    if any(
        "boundary_sample_predictions" not in selected_rows[method]
        for method in (unwhitened_method, whitened_method)
    ):
        # Older sweep reports retained only two coordinates and coefficient
        # norms. Refit this deterministic job to recover full-model predictions.
        from synthetic_sweep import _run_sweep_job

        condition = next(
            item for item in report["conditions"]
            if int(item["condition_id"]) == condition_id
        )
        with threadpool_limits(limits=1):
            reconstructed = _run_sweep_job({
                "config": dict(report["config"]),
                "condition": condition,
                "simulation": simulation,
            })
        reconstructed_payload = reconstructed["boundary_sample"]["test"]
        selected_rows = {row["method"]: row for row in reconstructed["rows"]}
        if not np.allclose(
            np.asarray(reconstructed_payload["first_two_coordinates"]), display_coordinates,
            rtol=1e-5, atol=1e-5,
        ):
            raise ValueError("Reconstructed boundary sample differs from the report.")
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if display_coordinates.ndim != 2 or display_coordinates.shape[1] != 2:
        raise ValueError("Stored decision-boundary coordinates are malformed.")
    lower = np.quantile(display_coordinates, 0.005, axis=0)
    upper = np.quantile(display_coordinates, 0.995, axis=0)
    padding = np.maximum(0.08 * (upper - lower), 1e-6)
    x_limits = (float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    y_limits = (float(lower[1] - padding[1]), float(upper[1] + padding[1]))

    methods = (unwhitened_method, whitened_method)
    q = int(sample["d"]) - 2
    if q <= 0:
        raise ValueError("Decision-boundary plotting requires at least one noise coordinate.")
    noise_coordinate_scale = float(report["config"]["sigma_epsilon"]) / np.sqrt(q)
    rows_by_method = {
        method: _method_rows(
            report,
            condition_id=condition_id,
            method=method,
        )
        for method in methods
    }
    simulation_sets = {
        method: {int(row["simulation"]) for row in rows}
        for method, rows in rows_by_method.items()
    }
    if simulation_sets[unwhitened_method] != simulation_sets[whitened_method]:
        raise ValueError("Boundary methods do not cover the same simulations.")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column_index, method in enumerate(methods):
        method_rows = rows_by_method[method]
        boundary_axis = axes[0, column_index]
        _draw_boundary_panel(
            boundary_axis,
            coordinates=display_coordinates,
            labels=labels,
            coefficient_summaries=_coefficient_summaries(method_rows),
            point_predictions=np.asarray(
                selected_rows[method]["boundary_sample_predictions"], dtype=bool
            ),
            noise_coordinate_scale=noise_coordinate_scale,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        boundary_axis.set_title(_boundary_title(method))
        boundary_axis.set_xlabel(
            r"Core coordinate $x_c$", fontsize=PLOT_LABEL_FONT_SIZE
        )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor=PAPER_RED, markeredgecolor="white",
            label=r"$\hat y=1$",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor=PAPER_BLUE, markeredgecolor="white",
            label=r"$\hat y=-1$",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="none", markeredgecolor=TEXT_GREY,
            label=r"$y=-1$",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=7,
            markerfacecolor="none", markeredgecolor=TEXT_GREY,
            label=r"$y=1$",
        ),
    ]
    axes[0, 1].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    figure.supylabel(
        r"Spurious coordinate $x_s$", x=0.045, fontsize=PLOT_LABEL_FONT_SIZE
    )
    figure.tight_layout(rect=(0.02, 0.0, 1.0, 1.0), w_pad=1.5)
    return figure


def _metric_summaries(
    report: Mapping[str, Any],
    *,
    method: str,
    metric: str,
    n_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    for row in report["results"]:
        if row["method"] != method:
            continue
        if n_train is not None and int(row["n_train"]) != int(n_train):
            continue
        q_over_n = (
            float(row["d"]) - 2.0
        ) / float(row["n_train"])
        value = float(row["test"][metric])
        grouped.setdefault(q_over_n, []).append(value)
    if not grouped:
        suffix = "" if n_train is None else f" at n_train={int(n_train)}"
        raise ValueError(f"Method {method!r} is absent from the sweep{suffix}.")
    x = np.asarray(sorted(grouped), dtype=np.float64)
    summaries = [_mean_and_interval(grouped[value]) for value in x]
    mean = np.asarray([value[0] for value in summaries])
    interval = np.asarray([value[1] for value in summaries])
    return x, mean, interval


def plot_performance(
    report: Mapping[str, Any],
    *,
    methods: Sequence[str],
    n_train: int | None = None,
    axes: Any = None,
) -> Any:
    """Plot overall and worst-group test accuracy against q/n."""
    standalone = axes is None
    if standalone:
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True)
    else:
        figure = axes[0].figure
    panels = (
        ("accuracy", "Overall accuracy"),
        ("worst_group_accuracy", "Worst-group accuracy"),
    )
    plotted_values: list[float] = []
    plotted_x_values: set[float] = set()
    for axis, (metric, y_label) in zip(axes, panels):
        for method in methods:
            x, mean, interval = _metric_summaries(
                report, method=method, metric=metric, n_train=n_train
            )
            plotted_x_values.update(float(value) for value in x)
            plotted_values.extend((mean - interval).tolist())
            plotted_values.extend((mean + interval).tolist())
            style = method_style(method)
            label = method_label(method)
            # Keep the synthetic-performance comparison aligned with the
            # manuscript palette: green standardization and red whitening.
            # This explicit pairing is needed because the shared style also
            # supports figures where empirical whitening is green.
            if method == "standardize":
                style = {
                    "color": PAPER_GREEN,
                    "marker": "o",
                    "linestyle": "-",
                }
            elif method == "whiten":
                style = {
                    "color": PAPER_ORANGE,
                    "marker": "o",
                    "linestyle": "-",
                }
            if method == "whiten_ledoit_wolf_nonlinear":
                style = {
                    "color": PAPER_RED,
                    "marker": "o",
                    "linestyle": "-",
                }
                label = "Whitening"
            axis.plot(
                x,
                mean,
                label=label,
                linewidth=2.4,
                **style,
            )
            axis.fill_between(
                x,
                np.clip(mean - interval, 0.0, 1.0),
                np.clip(mean + interval, 0.0, 1.0),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
        axis.axvline(1.0, color=PAPER_GREY, linestyle=":", linewidth=1.2)
        axis.set_xlabel(r"Noise-to-sample ratio $q/n$")
        axis.set_ylabel(y_label)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.tick_params(axis="y", labelleft=True)
    # The simulation points are not generally on Matplotlib's default tick
    # grid (e.g. q/n = 0.1, 0.2, 0.5, ...), so place a tick under every point.
    axes[0].set_xticks(sorted(plotted_x_values))
    axes[1].set_xticks(sorted(plotted_x_values))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for axis in axes:
        for tick_label in axis.get_xticklabels():
            tick_label.set_fontsize(PLOT_TICK_FONT_SIZE)
            tick_label.set_rotation(45)
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")
    if not plotted_values or not np.isfinite(plotted_values).all():
        raise ValueError("Performance plot contains no finite accuracy values.")
    lower = max(0.0, float(np.min(plotted_values)) - 0.03)
    upper = min(1.02, float(np.max(plotted_values)) + 0.02)
    if upper <= lower:
        upper = lower + 0.05
    axes[0].set_ylim(lower, upper)
    axes[0].legend(loc="lower left")

    if standalone:
        figure.tight_layout()
    return figure


def plot_gamma_comparison(
    reports: Sequence[Mapping[str, Any]], *,
    methods: Sequence[str] = ("identity", "standardize", "whiten_ledoit_wolf_nonlinear"),
    n_train: int | None = None,
) -> Any:
    """Repeat the q/n curves with only gamma changed and simulations matched."""
    if len(reports) < 2:
        raise ValueError("Gamma comparison requires at least two reports.")
    ignored = {"gamma", "workers", "blas_threads", "output", "overwrite",
               "run_theory_checks", "theorem2_validation", "proposition2_validation"}
    configs = [{k: v for k, v in r["config"].items() if k not in ignored} for r in reports]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Gamma reports must use identical main-sweep settings and seeds.")
    gammas = [float(r["config"]["gamma"]) for r in reports]
    if len(set(gammas)) != len(gammas):
        raise ValueError("Gamma reports must have distinct gamma values.")
    matched = [sorted((int(row["n_train"]), int(row["d"]),
                       int(row["simulation"]), str(row["method"]))
                      for row in report["results"] if row["method"] in methods)
               for report in reports]
    if any(keys != matched[0] for keys in matched[1:]):
        raise ValueError("Gamma reports must cover the same conditions and simulations.")
    sizes = sorted({int(row["n_train"]) for row in reports[0]["results"]})
    selected_n = sizes[0] if n_train is None else n_train
    configure_plot_style()
    figure, axes = plt.subplots(len(reports), 2, figsize=(11.2, 3.7 * len(reports)),
                                sharex=True, sharey=True, squeeze=False)
    for row_index, (row_axes, report) in enumerate(
        zip(axes, sorted(reports, key=lambda r: float(r["config"]["gamma"])))
    ):
        plot_performance(report, methods=methods, n_train=selected_n, axes=row_axes)
        for axis in row_axes:
            axis.set_title(fr"$\gamma={float(report['config']['gamma']):g}$")
            axis.set_ylim(0.0, 1.02)
            if row_index < len(axes) - 1:
                axis.set_xlabel("")
        row_axes[0].get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(methods))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def plot_performance_by_sample_size(
    report: Mapping[str, Any],
    *,
    method: str,
    sample_sizes: Sequence[int],
    metric: str = "accuracy",
) -> Any:
    """Compare one method across training sample sizes in a single panel."""
    sizes = tuple(int(value) for value in sample_sizes)
    if not sizes or len(sizes) != len(set(sizes)):
        raise ValueError("sample_sizes must contain unique training sizes.")
    metric_labels = {
        "accuracy": "Overall accuracy",
        "worst_group_accuracy": "Worst-group accuracy",
    }
    if metric not in metric_labels:
        raise ValueError(f"Unsupported sample-size comparison metric: {metric!r}.")

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    colors = (PAPER_RED, PAPER_PURPLE, PAPER_ORANGE, PAPER_GREEN, PAPER_BLUE)
    markers = ("D", "o", "s", "^", "v")
    plotted_values: list[float] = []
    plotted_x_values: set[float] = set()
    for index, n_train in enumerate(sizes):
        x, mean, interval = _metric_summaries(
            report, method=method, metric=metric, n_train=n_train
        )
        color = colors[index % len(colors)]
        plotted_x_values.update(float(value) for value in x)
        plotted_values.extend((mean - interval).tolist())
        plotted_values.extend((mean + interval).tolist())
        axis.plot(
            x,
            mean,
            color=color,
            marker=markers[index % len(markers)],
            linestyle="-",
            linewidth=2.4,
            label=fr"$n={n_train:,}$",
        )
        axis.fill_between(
            x,
            np.clip(mean - interval, 0.0, 1.0),
            np.clip(mean + interval, 0.0, 1.0),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    if not plotted_values or not np.isfinite(plotted_values).all():
        raise ValueError("Sample-size comparison contains no finite accuracy values.")
    lower = max(0.0, float(np.min(plotted_values)) - 0.03)
    upper = min(1.02, float(np.max(plotted_values)) + 0.02)
    if upper <= lower:
        upper = lower + 0.05
    axis.set_ylim(lower, upper)
    axis.axvline(1.0, color=PAPER_GREY, linestyle=":", linewidth=1.2)
    axis.set_xlabel(r"Noise-to-sample ratio $q/n$")
    axis.set_ylabel(metric_labels[metric])
    axis.set_xticks(sorted(plotted_x_values))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for tick_label in axis.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_horizontalalignment("right")
        tick_label.set_rotation_mode("anchor")
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure


def _coefficient_metric(
    row: Mapping[str, Any], diagnostic_key: str, metric: str, section: str
) -> float:
    value = row[diagnostic_key].get(section, {}).get(metric)
    if value is None or not np.isfinite(value):
        raise ValueError(
            f"Theorem 2 row has no finite {section}.{metric} value."
        )
    return float(value)


def _theorem2_value_limits(
    simulation_averages: np.ndarray,
    theorem_limits: np.ndarray,
    *,
    padding_fraction: float = THEOREM2_Y_PADDING_FRACTION,
) -> tuple[float, float]:
    """Pad the joint simulation-average and theorem-limit range."""
    averages = np.asarray(simulation_averages, dtype=np.float64)
    limits = np.asarray(theorem_limits, dtype=np.float64)
    values = np.concatenate((averages.reshape(-1), limits.reshape(-1)))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot set limits from empty or non-finite values.")
    if not np.isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError("Theorem 2 y-axis padding must be finite and nonnegative.")
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span <= 1e-12:
        span = max(abs(lower), 1.0)
    padding = float(padding_fraction) * span
    return lower - padding, upper + padding


def _plot_coefficient_confirmation(
    report: Mapping[str, Any],
    *,
    validation_key: str,
    diagnostic_key: str,
    condition_id: int | None,
    allow_main_fallback: bool,
    y_padding_fraction: float = THEOREM2_Y_PADDING_FRACTION,
) -> Any:
    """Plot coefficient convergence over n for a dedicated validation."""
    dedicated_rows = report.get(validation_key, {}).get("results", [])
    uses_sample_size_axis = condition_id is None and bool(dedicated_rows)
    if uses_sample_size_axis:
        candidate_rows = dedicated_rows
    elif allow_main_fallback:
        candidate_rows = report["results"]
    else:
        candidate_rows = []
    rows = [
        row
        for row in candidate_rows
        if row.get(diagnostic_key, {}).get("applicable") is True
        and (
            condition_id is None
            or int(row["condition_id"]) == int(condition_id)
        )
    ]
    if not rows:
        raise ValueError(
            f"No applicable {validation_key} rows exist for condition "
            f"{condition_id}. Run the enabled validation sweep first."
        )

    panels = (
        ("core", r"Core coefficient $\hat w_c$"),
        ("spurious", r"Spurious coefficient $\hat w_s$"),
        (
            "noise_prediction_variance",
            r"Noise contribution $\sigma_\epsilon^2"
            r"\|\hat{\mathbf{w}}_\epsilon\|_2^2/q$",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.25))
    if uses_sample_size_axis:
        series: dict[float | None, list[Mapping[str, Any]]] = {}
        for row in rows:
            target_ratio = float(row["target_q_n_ratio"])
            series.setdefault(target_ratio, []).append(row)
        x_label = r"Training sample size $n$"
    else:
        series = {None: rows}
        x_label = r"Noise-to-sample ratio $q/n$"
    colors = (PAPER_BLUE, PAPER_RED, PAPER_GREEN, PAPER_ORANGE, PAPER_PURPLE)
    if len(series) > len(colors):
        raise ValueError(
            f"At most {len(colors)} Theorem 2 ratios can be plotted clearly."
        )
    if uses_sample_size_axis:
        for target_ratio, ratio_rows in series.items():
            realized_ratios = np.asarray(
                [float(row[diagnostic_key]["psi_q_over_n"]) for row in ratio_rows]
            )
            maximum_rounding_error = max(
                0.5 / float(row["n_train"]) for row in ratio_rows
            )
            if not np.all(
                np.abs(realized_ratios - float(target_ratio))
                <= maximum_rounding_error + 1e-12
            ):
                raise ValueError(
                    "Dedicated Theorem 2 rows do not match their configured "
                    "q/n ratio."
                )

    for axis, (metric, title) in zip(axes, panels):
        panel_averages = []
        panel_limits = []
        for series_index, (target_ratio, series_rows) in enumerate(series.items()):
            grouped: dict[float, list[Mapping[str, Any]]] = {}
            for row in series_rows:
                x_value = (
                    float(row["n_train"])
                    if uses_sample_size_axis
                    else float(row[diagnostic_key]["psi_q_over_n"])
                )
                grouped.setdefault(x_value, []).append(row)
            x_grid = np.asarray(sorted(grouped), dtype=np.float64)
            observed_by_condition = []
            averages = []
            intervals = []
            limit_curve = []
            for x_value in x_grid:
                condition_rows = grouped[float(x_value)]
                observed = np.asarray(
                    [
                        _coefficient_metric(
                            row, diagnostic_key, metric, "finite_sample"
                        )
                        for row in condition_rows
                    ]
                )
                limits = np.asarray(
                    [
                        _coefficient_metric(row, diagnostic_key, metric, "limit")
                        for row in condition_rows
                    ]
                )
                if not np.allclose(limits, limits[0]):
                    raise ValueError(
                        "Theorem reference values vary within a condition."
                    )
                average, interval = _mean_and_interval(observed)
                observed_by_condition.append(observed)
                averages.append(average)
                intervals.append(interval)
                limit_curve.append(float(limits[0]))

            minimum_gap = (
                float(np.min(np.diff(x_grid)))
                if x_grid.size > 1
                else max(0.1, 0.1 * abs(float(x_grid[0])))
            )
            jitter_width = 0.04 * minimum_gap
            color = colors[series_index]
            for x_value, observed in zip(x_grid, observed_by_condition):
                offsets = np.linspace(-jitter_width, jitter_width, observed.size)
                axis.scatter(
                    x_value + offsets,
                    observed,
                    s=10,
                    color=color,
                    alpha=0.18,
                    edgecolors="none",
                )
            averages_array = np.asarray(averages)
            intervals_array = np.asarray(intervals)
            limit_array = np.asarray(limit_curve)
            axis.plot(
                x_grid,
                limit_array,
                color="#111111" if diagnostic_key == "theorem2" else color,
                linestyle="--",
            )
            axis.errorbar(
                x_grid,
                averages_array,
                yerr=intervals_array,
                color=TEXT_GREY,
                marker="o",
                capsize=3,
                linewidth=2.8,
                zorder=4,
            )
            panel_averages.append(averages_array)
            panel_limits.append(limit_array)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylim(
            *_theorem2_value_limits(
                np.concatenate(panel_averages),
                np.concatenate(panel_limits),
                padding_fraction=y_padding_fraction,
            )
        )
    axes[0].set_ylabel("Value")
    legend_handles = []
    if uses_sample_size_axis:
        legend_handles.extend(
            Line2D(
                [0], [0], color=colors[index], marker="o",
                label=fr"$q/n={float(ratio):g}$",
            )
            for index, ratio in enumerate(series)
        )
    legend_handles.extend(
        (
            Line2D(
                [0], [0], color=TEXT_GREY, marker="o", linewidth=2.8,
                label="Simulation average (95% CI)",
            ),
            Line2D(
                [0], [0],
                color="#111111" if diagnostic_key == "theorem2" else TEXT_GREY,
                linestyle="--",
                label="Theoretical limit",
            ),
        )
    )
    axes[2].legend(handles=legend_handles, loc="best")
    figure.tight_layout()
    return figure


def plot_theorem2_confirmation(
    report: Mapping[str, Any],
    *,
    condition_id: int | None,
) -> Any:
    """Plot Theorem 2 convergence over n at one fixed q/n > 1."""
    return _plot_coefficient_confirmation(
        report,
        validation_key="theorem2_validation",
        diagnostic_key="theorem2",
        condition_id=condition_id,
        allow_main_fallback=True,
    )


def plot_proposition2_confirmation(report: Mapping[str, Any]) -> Any:
    """Plot Proposition 2 validation over n for fixed ratios below one."""
    return _plot_coefficient_confirmation(
        report,
        validation_key="proposition2_validation",
        diagnostic_key="proposition2",
        condition_id=None,
        allow_main_fallback=False,
        y_padding_fraction=PROPOSITION2_Y_PADDING_FRACTION,
    )


def generate_all_plots(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
    formats: Sequence[str] = ("png",),
    overwrite: bool = False,
    boundary_ratio: float | None = None,
    boundary_condition_id: int | None = None,
    boundary_simulation: int | None = None,
    unwhitened_method: str = "identity",
    whitened_method: str = "whiten_ledoit_wolf_nonlinear",
    performance_methods: Sequence[str] | None = None,
    performance_n_train: int | None = None,
    theorem2_condition_id: int | None = None,
) -> dict[str, list[Path]]:
    """Generate and save all synthetic manuscript figures from one report."""
    configure_plot_style()
    plot_data = report["plot_data"]
    condition_id = _boundary_condition_id(
        report,
        boundary_ratio=boundary_ratio,
        boundary_condition_id=boundary_condition_id,
    )
    simulation = (
        int(plot_data["boundary_simulation"])
        if boundary_simulation is None
        else int(boundary_simulation)
    )
    available = set(_available_methods(report))
    sample_sizes = sorted(
        {int(row["n_train"]) for row in report["results"]}
    )
    selected_n_train = (
        sample_sizes[0]
        if performance_n_train is None
        else int(performance_n_train)
    )
    if selected_n_train not in sample_sizes:
        raise ValueError(
            f"No performance rows exist for n_train={selected_n_train}; "
            f"available values are {sample_sizes}."
        )
    if theorem2_condition_id is not None:
        theorem_condition: int | None = int(theorem2_condition_id)
    else:
        theorem_condition = None

    if performance_methods is None:
        default_methods = [unwhitened_method, "standardize", whitened_method]
        methods = list(dict.fromkeys(default_methods))
    else:
        methods = list(dict.fromkeys(str(value) for value in performance_methods))
    missing = [method for method in methods if method not in available]
    if missing:
        raise ValueError(
            f"Performance method(s) absent from the sweep: {missing}."
        )
    destination = Path(output_dir).expanduser().resolve()
    normalized_formats = tuple(
        str(value).lower().lstrip(".") for value in formats
    )
    if not normalized_formats:
        raise ValueError("At least one output format is required.")
    figure_names = [
        "decision_boundaries",
        "performance_vs_q_over_n",
        f"performance_empirical_n{selected_n_train}",
    ]
    has_theorem2_rows = bool(
        report.get("theorem2_validation", {}).get("results", [])
    ) or any(
        row.get("theorem2", {}).get("applicable") is True
        for row in report["results"]
    )
    has_proposition2_rows = bool(
        report.get("proposition2_validation", {}).get("results", [])
    )
    if has_theorem2_rows:
        figure_names.append("theorem2_confirmation")
    if has_proposition2_rows:
        figure_names.append("proposition2_confirmation")
    if report.get("gamma_sweep"):
        figure_names.append("performance_by_gamma")
    output_paths = [
        (destination / name).with_suffix(f".{extension}")
        for name in figure_names
        for extension in normalized_formats
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing figure(s): {names}. "
            "Pass --overwrite to replace them."
        )

    figures: dict[str, Any] = {}
    try:
        figures["decision_boundaries"] = plot_decision_boundaries(
            report,
            condition_id=condition_id,
            simulation=simulation,
            unwhitened_method=unwhitened_method,
            whitened_method=whitened_method,
        )
        figures["performance_vs_q_over_n"] = plot_performance(
            report, methods=methods, n_train=selected_n_train
        )
        if report.get("gamma_sweep"):
            figures["performance_by_gamma"] = plot_gamma_comparison(
                [report, *report["gamma_sweep"]], methods=methods,
                n_train=selected_n_train,
            )
        figures[f"performance_empirical_n{selected_n_train}"] = (
            plot_performance(
                report,
                methods=("identity", "whiten"),
                n_train=selected_n_train,
            )
        )
        if has_theorem2_rows:
            figures["theorem2_confirmation"] = plot_theorem2_confirmation(
                report, condition_id=theorem_condition
            )
        if has_proposition2_rows:
            figures["proposition2_confirmation"] = (
                plot_proposition2_confirmation(report)
            )
        saved: dict[str, list[Path]] = {}
        for name, figure in figures.items():
            saved[name] = save_figure(
                figure,
                destination / name,
                formats=normalized_formats,
                overwrite=overwrite,
            )
    finally:
        for figure in figures.values():
            plt.close(figure)
    return saved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create all manuscript figures from synthetic_sweep.py JSON."
    )
    parser.add_argument("results", help="Complete synthetic sweep JSON report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--formats", nargs="+", default=["png"])
    boundary_selection = parser.add_mutually_exclusive_group()
    boundary_selection.add_argument(
        "--boundary-ratio",
        type=float,
        help="Exact swept d/n ratio used for the averaged boundary figure.",
    )
    boundary_selection.add_argument("--boundary-condition-id", type=int)
    parser.add_argument("--boundary-simulation", type=int)
    parser.add_argument("--unwhitened-method", default="identity")
    parser.add_argument(
        "--whitened-method", default="whiten_ledoit_wolf_nonlinear"
    )
    parser.add_argument(
        "--performance-n-train",
        type=int,
        help="Training sample size used for the main performance figure.",
    )
    parser.add_argument("--theorem2-condition-id", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--gamma-reports", nargs="+",
        help="Additional matched reports differing only in gamma; render appendix comparison only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = load_report(args.results)
    if args.gamma_reports:
        figure = plot_gamma_comparison(
            [report, *(load_report(path) for path in args.gamma_reports)],
            methods=args.methods or ("identity", "standardize", args.whitened_method),
            n_train=args.performance_n_train,
        )
        try:
            paths = save_figure(figure, Path(args.output_dir) / "performance_by_gamma",
                                formats=args.formats, overwrite=args.overwrite)
        finally:
            plt.close(figure)
        for path in paths:
            print(path)
        return
    saved = generate_all_plots(
        report,
        output_dir=args.output_dir,
        formats=args.formats,
        overwrite=args.overwrite,
        boundary_ratio=args.boundary_ratio,
        boundary_condition_id=args.boundary_condition_id,
        boundary_simulation=args.boundary_simulation,
        unwhitened_method=args.unwhitened_method,
        whitened_method=args.whitened_method,
        performance_methods=args.methods,
        performance_n_train=args.performance_n_train,
        theorem2_condition_id=args.theorem2_condition_id,
    )
    for name, paths in saved.items():
        print(f"{name}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
