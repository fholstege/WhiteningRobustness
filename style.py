"""Shared, dependency-light plotting style for manuscript figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.text import Text


PAPER_BLUE = "#4c78a8"
PAPER_RED = "#c44e52"
PAPER_GREEN = "#59a14f"
PAPER_ORANGE = "#f28e2b"
PAPER_PURPLE = "#9467bd"
PAPER_GREY = "#777777"
TEXT_GREY = "#333333"

# Shared manuscript type scale for every figure and plotting module.
PLOT_LABEL_FONT_SIZE = 16
PLOT_TICK_FONT_SIZE = 12
PLOT_LEGEND_FONT_SIZE = 13
PLOT_TITLE_FONT_SIZE = 16
# Comparison-panel headings are intentionally quieter than general figure titles.
PLOT_COMPARISON_TITLE_FONT_SIZE = 12
# Dataset names need to remain legible and consistent across manuscript bar plots.
PLOT_DATASET_TICK_FONT_SIZE = 14
PLOT_ANNOTATION_FONT_SIZE = 11
PLOT_BAR_ANNOTATION_FONT_SIZE = 10
PLOT_COMPARISON_BAR_ANNOTATION_FONT_SIZE = 9
PLOT_RAYLEIGH_LABEL_FONT_SIZE = 18
PLOT_BASE_FONT_SIZE = 11
PAPER_FIGURE_WIDTH_INCHES = 11.2
PLOT_EXPORT_DPI = 300

METHOD_LABELS = {
    "identity": "No preprocessing",
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
            "font.size": PLOT_BASE_FONT_SIZE,
            "axes.labelsize": PLOT_LABEL_FONT_SIZE,
            "axes.titlesize": PLOT_TITLE_FONT_SIZE,
            "figure.labelsize": PLOT_LABEL_FONT_SIZE,
            "figure.titlesize": PLOT_TITLE_FONT_SIZE,
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
            "savefig.dpi": PLOT_EXPORT_DPI,
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
    # The manuscript includes each tightly cropped image at \textwidth. Use
    # the exported width, including padding, to preserve printed font sizes.
    if not getattr(figure, "_paper_text_scaled", False):
        artists = figure.findobj(match=Text)
        original_sizes = [artist.get_fontsize() for artist in artists]
        padding = float(plt.rcParams["savefig.pad_inches"])
        for _ in range(3):
            figure.canvas.draw()
            export_width = (
                figure.get_tightbbox(figure.canvas.get_renderer()).width
                + 2.0 * padding
            )
            scale = export_width / PAPER_FIGURE_WIDTH_INCHES
            for artist, original_size in zip(artists, original_sizes):
                artist.set_fontsize(original_size * scale)
        figure._paper_text_scaled = True
    for path in paths:
        figure.savefig(path)
    return paths
