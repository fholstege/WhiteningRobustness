"""Shared, dependency-light plotting style for manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


PAPER_BLUE = "#4c78a8"
PAPER_RED = "#c44e52"
PAPER_GREEN = "#59a14f"
PAPER_ORANGE = "#f28e2b"
PAPER_PURPLE = "#9467bd"
PAPER_GREY = "#777777"
TEXT_GREY = "#333333"

# Shared manuscript type scale.  Figure 4 established these sizes; keeping
# them here prevents individual plotting modules from quietly drifting apart.
PLOT_LABEL_FONT_SIZE = 21
PLOT_TICK_FONT_SIZE = 19
PLOT_LEGEND_FONT_SIZE = 17
PLOT_ANNOTATION_FONT_SIZE = 16

METHOD_LABELS = {
    "identity": "None",
    "standardize": "Standardization",
    "whiten": "Whitening",
    "whiten_ledoit_wolf": "Whitening",
    "whiten_ledoit_wolf_nonlinear": "Whitening",
}

METHOD_STYLES = {
    "identity": {"color": PAPER_BLUE, "marker": "o", "linestyle": "-"},
    "standardize": {
        "color": PAPER_ORANGE,
        "marker": "s",
        "linestyle": "--",
    },
    "whiten": {"color": PAPER_GREEN, "marker": "^", "linestyle": "-"},
    "whiten_ledoit_wolf": {
        "color": PAPER_RED,
        "marker": "D",
        "linestyle": "-",
    },
    "whiten_ledoit_wolf_nonlinear": {
        "color": PAPER_PURPLE,
        "marker": "P",
        "linestyle": "-",
    },
}


def configure_plot_style() -> None:
    """Apply the paper style without requiring an external LaTeX install."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": PLOT_LABEL_FONT_SIZE,
            "axes.titlesize": 12,
            "axes.titleweight": "regular",
            "axes.edgecolor": TEXT_GREY,
            "axes.labelcolor": TEXT_GREY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "xtick.labelsize": PLOT_TICK_FONT_SIZE,
            "ytick.labelsize": PLOT_TICK_FONT_SIZE,
            "legend.fontsize": PLOT_LEGEND_FONT_SIZE,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def method_label(method: str) -> str:
    """Return a compact display label for a stored sweep method."""
    return METHOD_LABELS.get(method, method.replace("_", " "))


def method_style(method: str) -> dict[str, Any]:
    """Return a stable line style for a stored sweep method."""
    if method in METHOD_STYLES:
        return dict(METHOD_STYLES[method])
    return dict(METHOD_STYLES["identity"])


def save_figure(
    figure: Any,
    output_stem: str | Path,
    *,
    formats: Iterable[str] = ("png",),
    overwrite: bool = False,
) -> list[Path]:
    """Save one figure in each requested format with overwrite protection."""
    stem = Path(output_stem).expanduser().resolve()
    normalized_formats = tuple(str(value).lower().lstrip(".") for value in formats)
    if not normalized_formats:
        raise ValueError("At least one output format is required.")
    if "pdf" in normalized_formats:
        raise ValueError("PDF output is disabled; use PNG or SVG instead.")
    paths = [stem.with_suffix(f".{extension}") for extension in normalized_formats]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing figure(s): {names}. "
            "Pass --overwrite to replace them."
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        figure.savefig(path)
    return paths
