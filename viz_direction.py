#!/usr/bin/env python3
"""Plot covariance alignment of spurious and core directions on training data.

For each target class, the spurious direction is the difference between the
mean representation at the two values of the binary spurious attribute.  The
class-specific directions are averaged, and their Rayleigh quotient under the
training covariance is normalized by its largest eigenvalue.

The core direction swaps the roles of the target and spurious attribute.  For
multiclass targets, the corresponding between-class mean subspace is used;
this reduces to the mean-difference direction for a binary target.

The stored group convention is ``g = 2*y + 1[a = 1]``.  Thus the original
``a in {-1, 1}`` is recovered from the bundle's training labels and groups.

Run the default manuscript comparison with::

    python viz_direction.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy import stats

from data import EmbeddingBundle
from evaluate import find_bundle
from style import TEXT_GREY, PLOT_ANNOTATION_FONT_SIZE, configure_plot_style, save_figure
from whitening import make_transform


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "plots" / "spurious_direction_alignment"
DEFAULT_SPECTRUM_CACHE = ROOT / "results_paper" / "spectrum_cache"
SPECTRUM_CACHE_VERSION = 1
DATASET_COLORS = {
    "WB": "#2ECC71",
    "CelebA": "#9467BD",
    "multiNLI": "#17BECF",
}
DIRECTION_ALPHA = {"spurious": 1.0, "core": 0.45}


@dataclass(frozen=True)
class DatasetSpec:
    """Identity and display settings for one embedding bundle."""

    dataset: str
    label: str
    representation: str


@dataclass(frozen=True)
class DirectionEstimate:
    """Seed-level scores and their mean 95% confidence interval."""

    scores: tuple[float, ...]
    mean: float
    lower: float
    upper: float

    @property
    def seed_count(self) -> int:
        return len(self.scores)


def summarize_scores(scores: Sequence[float]) -> DirectionEstimate:
    """Summarize seed-level scores with a two-sided 95% Student-t interval."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Seed-level scores must be a non-empty finite vector.")
    mean = float(values.mean())
    if values.size == 1:
        lower = upper = mean
    else:
        standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
        half_width = float(stats.t.ppf(0.975, values.size - 1) * standard_error)
        lower, upper = mean - half_width, mean + half_width
    return DirectionEstimate(tuple(float(value) for value in values), mean, lower, upper)


def _normalized_identifier(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return {"waterbird": "wb", "waterbirds": "wb"}.get(normalized, normalized)


def available_seeds(
    artifact_root: Path,
    *,
    dataset: str,
    representation: str,
) -> tuple[int, ...]:
    """Discover all integer representation seeds for one dataset/model pair."""
    requested_dataset = _normalized_identifier(dataset)
    requested_representation = _normalized_identifier(representation)
    seeds = set()
    for manifest_path in sorted(artifact_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            _normalized_identifier(manifest.get("dataset")) == requested_dataset
            and _normalized_identifier(manifest.get("representation"))
            == requested_representation
        ):
            seed = manifest.get("representation_seed")
            if not isinstance(seed, int):
                raise ValueError(
                    f"Expected an integer representation seed in {manifest_path}."
                )
            seeds.add(seed)
    if not seeds:
        raise FileNotFoundError(
            f"No bundles match dataset={dataset!r}, "
            f"representation={representation!r}."
        )
    return tuple(sorted(seeds))


def recover_spurious_attribute(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Recover ``a in {-1, 1}`` from bundles using ``g = 2*y + 1[a=1]``."""
    labels = np.asarray(y)
    groups = np.asarray(g)
    if labels.ndim != 1 or groups.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("y and g must be aligned one-dimensional arrays.")
    indicator = groups - 2 * labels
    if not np.all(np.isin(indicator, (0, 1))):
        raise ValueError(
            "Training groups do not follow g = 2*y + 1[a=1]; "
            "the spurious attribute cannot be recovered."
        )
    return (2 * indicator - 1).astype(np.int8, copy=False)


def _conditional_mean(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Compute a selected-row mean without materializing a large indexed copy."""
    total = np.zeros(X.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        local_mask = mask[start:stop]
        if np.any(local_mask):
            total += X[start:stop][local_mask].sum(axis=0, dtype=np.float64)
            count += int(local_mask.sum())
    if count == 0:
        raise ValueError("A target-class/spurious-attribute cell is empty.")
    return total / count


def estimate_spurious_direction(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    r"""Estimate ``mean_y(E[x|a=1,y] - E[x|a=-1,y])`` on training data."""
    features = np.asarray(X)
    labels = np.asarray(y)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("X must align with the one-dimensional label array y.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    attributes = recover_spurious_attribute(labels, g)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two target classes are required.")

    directions = []
    for target in classes:
        positive = _conditional_mean(
            features,
            (labels == target) & (attributes == 1),
            chunk_size=chunk_size,
        )
        negative = _conditional_mean(
            features,
            (labels == target) & (attributes == -1),
            chunk_size=chunk_size,
        )
        directions.append(positive - negative)
    direction = np.mean(directions, axis=0, dtype=np.float64)
    squared_norm = float(direction @ direction)
    if squared_norm <= 0.0 or not np.isfinite(squared_norm):
        raise ValueError("The estimated spurious direction has invalid zero norm.")
    return direction


def estimate_core_directions(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    r"""Estimate a balanced between-class core subspace.

    For every target class, its mean is first averaged equally over the two
    spurious-attribute values.  The returned rows are those balanced class
    means centered around their across-class mean.  With two target classes,
    their span is exactly the swapped mean-difference direction
    ``mean_a(E[x|y=1,a] - E[x|y=0,a])``.
    """
    features = np.asarray(X)
    labels = np.asarray(y)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("X must align with the one-dimensional label array y.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    attributes = recover_spurious_attribute(labels, g)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("At least two target classes are required.")

    balanced_class_means = []
    for target in classes:
        attribute_means = [
            _conditional_mean(
                features,
                (labels == target) & (attributes == attribute),
                chunk_size=chunk_size,
            )
            for attribute in (-1, 1)
        ]
        balanced_class_means.append(
            np.mean(attribute_means, axis=0, dtype=np.float64)
        )
    means = np.asarray(balanced_class_means, dtype=np.float64)
    directions = means - means.mean(axis=0, keepdims=True)
    squared_norm = float(np.sum(directions**2))
    if squared_norm <= 0.0 or not np.isfinite(squared_norm):
        raise ValueError("The estimated core subspace has invalid zero norm.")
    return directions


def normalized_rayleigh_quotient(
    X: np.ndarray,
    direction: np.ndarray,
    *,
    relative_tolerance: float | None = None,
    maximum_eigenvalue: float | None = None,
    chunk_size: int = 16_384,
) -> float:
    r"""Return ``(u^T Sigma u / u^T u) / lambda_max(Sigma)``."""
    vector = np.asarray(direction, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("The direction must be one-dimensional.")
    return normalized_subspace_rayleigh_quotient(
        X,
        vector[None, :],
        relative_tolerance=relative_tolerance,
        maximum_eigenvalue=maximum_eigenvalue,
        chunk_size=chunk_size,
    )


def normalized_subspace_rayleigh_quotient(
    X: np.ndarray,
    directions: np.ndarray,
    *,
    relative_tolerance: float | None = None,
    maximum_eigenvalue: float | None = None,
    chunk_size: int = 16_384,
) -> float:
    r"""Return the normalized covariance Rayleigh quotient of a subspace.

    For direction rows ``d_j``, this is
    ``sum_j d_j^T Sigma d_j / (lambda_1 sum_j d_j^T d_j)``.  It is invariant
    to orthogonal changes of basis within the represented subspace.
    """
    features = np.asarray(X)
    vectors = np.asarray(directions, dtype=np.float64)
    if (
        features.ndim != 2
        or vectors.ndim != 2
        or vectors.shape[0] == 0
        or vectors.shape[1] != features.shape[1]
    ):
        raise ValueError(
            "Directions must be a non-empty matrix with one column per feature."
        )

    if len(features) < 2:
        raise ValueError("At least two training observations are required.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if maximum_eigenvalue is None:
        transform = make_transform(
            "whiten",
            whitening_estimator="empirical",
            whitening_relative_tolerance=relative_tolerance,
        ).fit(features)
        if transform.eigenvalues_ is None:
            raise RuntimeError("Empirical covariance eigendecomposition is unavailable.")
        maximum_eigenvalue = float(transform.eigenvalues_[0])
    if maximum_eigenvalue <= 0.0 or not np.isfinite(maximum_eigenvalue):
        raise ValueError("The largest covariance eigenvalue must be finite and positive.")

    # The scalar projections give u^T Sigma u exactly while avoiding a second
    # feature-covariance workspace after the eigendecomposition.
    projections = np.empty((len(features), len(vectors)), dtype=np.float32)
    vectors32 = vectors.astype(np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        projections[start:stop] = features[start:stop] @ vectors32.T
    denominator = float(np.sum(vectors**2))
    rayleigh = float(
        projections.var(axis=0, ddof=1, dtype=np.float64).sum() / denominator
    )
    score = rayleigh / maximum_eigenvalue
    if not np.isfinite(score):
        raise ValueError("The normalized Rayleigh quotient is not finite.")
    return score


def _spectrum_cache_path(
    cache_dir: Path,
    *,
    bundle: EmbeddingBundle,
    relative_tolerance: float | None,
) -> Path:
    """Return the cache path shared with the eigenvalue plotting script."""
    key = {
        "version": SPECTRUM_CACHE_VERSION,
        "bundle_fingerprint": bundle.fingerprint,
        "relative_tolerance": relative_tolerance,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    manifest = bundle.manifest
    return cache_dir / (
        f"{_normalized_identifier(manifest['dataset'])}_"
        f"{_normalized_identifier(manifest['representation'])}_"
        f"seed_{manifest.get('representation_seed')}_{digest}.npz"
    )


def largest_covariance_eigenvalue(
    bundle: EmbeddingBundle,
    *,
    relative_tolerance: float | None,
    cache_dir: Path,
    refit: bool = False,
) -> float:
    """Load or fit the training eigendecomposition and return lambda one."""
    cache_path = _spectrum_cache_path(
        cache_dir,
        bundle=bundle,
        relative_tolerance=relative_tolerance,
    )
    if cache_path.is_file() and not refit:
        with np.load(cache_path, allow_pickle=False) as cached:
            eigenvalues = np.asarray(cached["eigenvalues"], dtype=np.float64)
    else:
        transform = make_transform(
            "whiten",
            whitening_estimator="empirical",
            whitening_relative_tolerance=relative_tolerance,
        ).fit(bundle.split("train").X)
        if transform.eigenvalues_ is None:
            raise RuntimeError("Empirical covariance eigendecomposition is unavailable.")
        eigenvalues = np.asarray(transform.eigenvalues_, dtype=np.float64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, eigenvalues=eigenvalues)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0 or eigenvalues[0] <= 0.0:
        raise ValueError(f"Invalid covariance eigenspectrum in {cache_path}.")
    return float(eigenvalues[0])


def make_figure(
    labels: Sequence[str],
    spurious_estimates: Sequence[DirectionEstimate],
    core_estimates: Sequence[DirectionEstimate],
) -> plt.Figure:
    """Create paired spurious/core alignment bars for three datasets."""
    if (
        len(labels) != 3
        or len(spurious_estimates) != 3
        or len(core_estimates) != 3
    ):
        raise ValueError("Exactly three dataset labels and estimate pairs are required.")
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    positions = np.arange(3)
    width = 0.34
    plotted = []
    for offset, name, estimates in (
        (-width / 2, "spurious", spurious_estimates),
        (width / 2, "core", core_estimates),
    ):
        means = np.asarray([estimate.mean for estimate in estimates])
        errors = np.asarray(
            [
                [estimate.mean - estimate.lower for estimate in estimates],
                [estimate.upper - estimate.mean for estimate in estimates],
            ]
        )
        bars = axis.bar(
            positions + offset,
            means,
            yerr=errors,
            width=width,
            color=tuple(
                DATASET_COLORS[key] for key in ("WB", "CelebA", "multiNLI")
            ),
            alpha=DIRECTION_ALPHA[name],
            edgecolor="white",
            linewidth=0.8,
            ecolor=TEXT_GREY,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        plotted.extend(zip(bars, estimates))
    axis.set_xticks(positions, labels)
    axis.set_ylabel(r"Normalized Rayleigh quotient $\mathrm{rq}(u)/\lambda_1$")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0)
    axis.legend(
        handles=[
            Patch(
                facecolor=TEXT_GREY,
                alpha=DIRECTION_ALPHA[name],
                label=name.capitalize(),
            )
            for name in ("spurious", "core")
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    for bar, estimate in plotted:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(float(estimate.upper) + 0.025, 1.02),
            f"{estimate.mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_ANNOTATION_FONT_SIZE,
            color=TEXT_GREY,
        )
    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/embeddings",
        help="Root containing NumPy embedding bundles.",
    )
    parser.add_argument(
        "--seed",
        default="all",
        help="Representation seed, or 'all' for seed means and 95%% CIs (default: all).",
    )
    parser.add_argument("--waterbirds-representation", default="resnet50")
    parser.add_argument("--celeba-representation", default="resnet50")
    parser.add_argument("--multinli-representation", default="bert")
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=None,
        help="Empirical covariance rank tolerance (default: float32 epsilon).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16_384,
        help="Rows per chunk when estimating conditional means.",
    )
    parser.add_argument(
        "--spectrum-cache",
        default=str(DEFAULT_SPECTRUM_CACHE),
        help="Fingerprint-keyed covariance eigenspectrum cache.",
    )
    parser.add_argument(
        "--refit-spectra",
        action="store_true",
        help="Ignore cached covariance eigenspectra and recompute them.",
    )
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify complete array checksums before analysis (slower).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path stem (default: plots/spurious_direction_alignment).",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "svg"),
        help="Output format; repeat for both PNG and SVG (default: png).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output figure.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact_root = Path(args.artifact_root).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    cache_dir = Path(args.spectrum_cache).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    if args.seed != "all":
        try:
            requested_seed = int(args.seed)
        except ValueError as exc:
            raise ValueError("--seed must be an integer or 'all'.") from exc
    else:
        requested_seed = None
    specs = (
        DatasetSpec("WB", "Waterbirds", args.waterbirds_representation),
        DatasetSpec("CelebA", "CelebA", args.celeba_representation),
        DatasetSpec("multiNLI", "MultiNLI", args.multinli_representation),
    )

    spurious_estimates = []
    core_estimates = []
    for index, spec in enumerate(specs, start=1):
        seeds = (
            available_seeds(
                artifact_root,
                dataset=spec.dataset,
                representation=spec.representation,
            )
            if requested_seed is None
            else (requested_seed,)
        )
        spurious_scores = []
        core_scores = []
        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"[{index}/3, seed {seed_index}/{len(seeds)}] {spec.label}: "
                f"loading {spec.representation}, seed {seed}",
                flush=True,
            )
            path = find_bundle(
                artifact_root,
                dataset=spec.dataset,
                representation=spec.representation,
                seed=seed,
            )
            bundle = EmbeddingBundle.load(path, verify_hashes=args.verify_hashes)
            train = bundle.split("train")
            direction = estimate_spurious_direction(
                train.X,
                train.y,
                train.g,
                chunk_size=args.chunk_size,
            )
            core_directions = estimate_core_directions(
                train.X,
                train.y,
                train.g,
                chunk_size=args.chunk_size,
            )
            maximum_eigenvalue = largest_covariance_eigenvalue(
                bundle,
                relative_tolerance=args.relative_tolerance,
                cache_dir=cache_dir,
                refit=args.refit_spectra,
            )
            spurious_score = normalized_rayleigh_quotient(
                train.X,
                direction,
                maximum_eigenvalue=maximum_eigenvalue,
                chunk_size=args.chunk_size,
            )
            core_score = normalized_subspace_rayleigh_quotient(
                train.X,
                core_directions,
                maximum_eigenvalue=maximum_eigenvalue,
                chunk_size=args.chunk_size,
            )
            spurious_scores.append(spurious_score)
            core_scores.append(core_score)
            print(
                f"    spurious RQ = {spurious_score:.6f}, "
                f"core RQ = {core_score:.6f}",
                flush=True,
            )
        spurious_estimate = summarize_scores(spurious_scores)
        core_estimate = summarize_scores(core_scores)
        spurious_estimates.append(spurious_estimate)
        core_estimates.append(core_estimate)
        print(
            f"{spec.label}: spurious mean={spurious_estimate.mean:.6f}, "
            f"95% CI=[{spurious_estimate.lower:.6f}, "
            f"{spurious_estimate.upper:.6f}]; core mean={core_estimate.mean:.6f}, "
            f"95% CI=[{core_estimate.lower:.6f}, {core_estimate.upper:.6f}]; "
            f"n={spurious_estimate.seed_count}",
            flush=True,
        )

    figure = make_figure(
        [spec.label for spec in specs], spurious_estimates, core_estimates
    )
    try:
        saved = save_figure(
            figure,
            args.output,
            formats=args.formats or ("png",),
            overwrite=args.overwrite,
        )
    finally:
        plt.close(figure)
    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
