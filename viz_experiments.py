#!/usr/bin/env python3
"""Plot empirical spectra and classifier simplicity for three benchmarks.

The empirical covariance is fitted on each bundle's training representations,
using exactly the same float32, ``n - 1`` covariance and rank-tolerance
conventions as :class:`whitening.WhiteningTransform`. Curves and reported
spectral entropies are averaged over all available representation seeds. The
horizontal axis evaluates the top eigenvalue and fixed fractions of the full
representation dimension so spectra with different dimensions are comparable.

Render the main representation comparison::

    python viz_experiments.py

Render the alternative-backbone comparison::

    python viz_experiments.py --preset appendix

Render the frozen-backbone simplicity comparison, selecting each head by
validation balanced log loss::

    python viz_experiments.py --plot-type frozen

Each figure places the cumulative spectrum and the simplicity of the selected
linear heads side by side. Existing plots are protected unless ``--overwrite``
is passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from aggregate import (
    _refit_selected_test_predictions_serial,
    aggregate_runs,
    load_sweep_runs,
)
from data import EmbeddingBundle
from evaluate import ROOT, find_bundle
from metric import selection_metrics
from model import LogisticConfig, fit_logistic
from methods import _neurotune_validation_partition
from style import (
    PAPER_BLUE,
    PAPER_GREEN,
    PAPER_GREY,
    PAPER_RED,
    PLOT_ANNOTATION_FONT_SIZE,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    configure_plot_style,
    save_figure,
)
from whitening import make_transform


DATASET_ORDER = ("WB", "CelebA", "multiNLI")
COMPARISON_ANNOTATION_FONT_SIZE = 12
DATASET_LABELS = {
    "WB": "Waterbirds",
    "CelebA": "CelebA",
    "multiNLI": "MultiNLI",
}
REPRESENTATION_LABELS = {
    "resnet50": "ResNet-50",
    "dino_vitb16": "DINO ViT-B/16",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "clip_openai_vitb16": "CLIP ViT-B/16",
    "bert": "BERT",
    "debertav3": "DeBERTa-v3",
}
DATASET_COLORS = {
    "WB": "#2ECC71",
    "CelebA": "#9467BD",
    "multiNLI": "#17BECF",
}
DINO_VITB16_DIMENSION = 768
ALL_SEEDS = "all"
MAX_DISPLAY_EIGENVALUES = 500
COEFFICIENT_CACHE_VERSION = 3
STANDARDIZE_CACHE_VERSION = 2
COMPARISON_COEFFICIENT_CACHE_VERSION = 4
SPECTRUM_CACHE_VERSION = 1
BARPLOT_WHITENING_METHOD = "whiten_ledoit_wolf_nonlinear"
BARPLOT_WHITENING_ESTIMATOR = "ledoit-wolf-nonlinear"
DEFAULT_COEFFICIENT_CACHE = ROOT / "results_paper" / "coefficient_head_cache"
DEFAULT_SPECTRUM_CACHE = ROOT / "results_paper" / "spectrum_cache"
ADJUST_RANDOM_BAR = -0.1
HSPACE_LEGEND = -0.1


# These presets mirror the main and appendix representation families in
# ``viz.py``. Every preset entry discovers and averages all matching seeds.
EXPERIMENT_PRESETS: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "main": (
        ("WB", "resnet50", ALL_SEEDS),
        ("CelebA", "resnet50", ALL_SEEDS),
        ("multiNLI", "BERT", ALL_SEEDS),
    ),
    "appendix": (
        ("WB", "dino_vitb16", ALL_SEEDS),
        ("CelebA", "dino_vitb16", ALL_SEEDS),
        ("multiNLI", "debertav3", ALL_SEEDS),
    ),
}

SWEEP_PRESETS: Mapping[str, tuple[str, ...]] = {
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

# Frozen sweeps use randomized train/validation splits of one fixed pretrained
# checkpoint. ``--plot-type frozen`` reconstructs each split before refitting
# the validation-selected heads, then caches those coefficients in the same
# fingerprint-keyed cache as the fine-tuned simplicity plots.
FROZEN_SWEEP_PATHS = (
    "results_frozen/WB_dinov3_vitb16_frozen.json",
    "results_frozen/CelebA_dinov3_vitb16_frozen.json",
    "results_frozen/WB_clip_openai_vitb16_frozen.json",
    "results_frozen/CelebA_clip_openai_vitb16_frozen.json",
)
FROZEN_DATASET_ORDER = ("WB", "CelebA")
FROZEN_REPRESENTATION_ORDER = ("dinov3_vitb16", "clip_openai_vitb16")

COMPARISON_SWEEP_PATHS: Mapping[str, Mapping[str, str]] = {
    "dfr": {
        "WB": "results_paper/comparisons/WB_resnet50_DFR.json",
        "CelebA": "results_paper/comparisons/CelebA_resnet50_DFR.json",
        "multiNLI": "results_paper/comparisons/multiNLI_BERT_DFR.json",
    },
    "afr": {
        "WB": "results_paper/comparisons/WB_resnet50_AFR.json",
        "CelebA": "results_paper/comparisons/CelebA_resnet50_AFR.json",
        "multiNLI": "results_paper/comparisons/multiNLI_BERT_AFR.json",
    },
    "neurotune": {
        "WB": "results_paper/comparisons/WB_resnet50_NEUROTUNE.json",
        "CelebA": "results_paper/comparisons/CelebA_resnet50_NEUROTUNE.json",
        "multiNLI": "results_paper/comparisons/multiNLI_BERT_NEUROTUNE.json",
    },
}
COMPARISON_METHOD_LABELS = {"dfr": "DFR", "afr": "AFR", "neurotune": "NeuroTune"}
COMPARISON_PLOTTED_TRANSFORMS = (
    "identity",
    "standardize",
    "whiten_ledoit_wolf_nonlinear",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One representation and either one seed or all available seeds."""

    dataset: str
    representation: str
    seed: int | None | str


@dataclass(frozen=True)
class SeedSpectrum:
    """One empirical covariance spectrum fitted on one training bundle."""

    dataset: str
    representation: str
    seed: int | str | None
    feature_dimension: int
    eigenvalues: np.ndarray
    cumulative_variance: np.ndarray
    normalized_spectral_entropy: float

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)


@dataclass(frozen=True)
class Spectrum:
    """A representation spectrum averaged over matched seed-level spectra."""

    dataset: str
    representation: str
    seeds: tuple[int | str | None, ...]
    feature_dimension: int
    retained_ranks: tuple[int, ...]
    cumulative_variance: np.ndarray
    normalized_spectral_entropy: float
    entropy_standard_error: float

    @property
    def rank(self) -> int:
        return max(self.retained_ranks)

    @property
    def seed_count(self) -> int:
        return len(self.seeds)


@dataclass(frozen=True)
class HeadSimplicity:
    """Seed-level coefficient simplicity scores for one representation."""

    dataset: str
    representation: str
    seeds: tuple[int | str | None, ...]
    identity_scores: tuple[float, ...]
    standardize_scores: tuple[float, ...]
    whiten_scores: tuple[float, ...]
    random_direction_scores: tuple[float, ...]

    @property
    def identity_mean(self) -> float:
        return float(np.mean(self.identity_scores))

    @property
    def whiten_mean(self) -> float:
        return float(np.mean(self.whiten_scores))

    @property
    def standardize_mean(self) -> float:
        return float(np.mean(self.standardize_scores))

    @staticmethod
    def _standard_error(values: tuple[float, ...]) -> float:
        array = np.asarray(values, dtype=np.float64)
        return (
            float(array.std(ddof=1) / np.sqrt(array.size))
            if array.size > 1
            else 0.0
        )

    @staticmethod
    def _bounded_interval(values: tuple[float, ...]) -> tuple[float, float]:
        """Return a logit-scale 95% t interval based on the seed-level SE."""
        mean = float(np.mean(values))
        if len(values) <= 1:
            return mean, mean
        standard_error = HeadSimplicity._standard_error(values)
        if standard_error == 0.0:
            return mean, mean
        epsilon = np.finfo(np.float64).eps
        bounded_mean = float(np.clip(mean, epsilon, 1.0 - epsilon))
        logit_mean = np.log(bounded_mean / (1.0 - bounded_mean))
        logit_se = standard_error / (bounded_mean * (1.0 - bounded_mean))
        radius = stats.t.ppf(0.975, df=len(values) - 1) * logit_se

        def inverse_logit(value: float) -> float:
            return float(1.0 / (1.0 + np.exp(-value)))

        return inverse_logit(logit_mean - radius), inverse_logit(logit_mean + radius)

    @property
    def identity_standard_error(self) -> float:
        return self._standard_error(self.identity_scores)

    @property
    def whiten_standard_error(self) -> float:
        return self._standard_error(self.whiten_scores)

    @property
    def standardize_standard_error(self) -> float:
        return self._standard_error(self.standardize_scores)

    @property
    def identity_interval(self) -> tuple[float, float]:
        return self._bounded_interval(self.identity_scores)

    @property
    def whiten_interval(self) -> tuple[float, float]:
        return self._bounded_interval(self.whiten_scores)

    @property
    def standardize_interval(self) -> tuple[float, float]:
        return self._bounded_interval(self.standardize_scores)

    @property
    def random_direction_mean(self) -> float:
        return float(np.mean(self.random_direction_scores))


def _normalized_identifier(value: Any) -> str:
    return "".join(
        character for character in str(value).lower() if character.isalnum()
    )


def _canonical_dataset(value: str) -> str:
    normalized = _normalized_identifier(value)
    aliases = {
        "wb": "WB",
        "waterbird": "WB",
        "waterbirds": "WB",
        "celeba": "CelebA",
        "multinli": "multiNLI",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {value!r}; expected WB, CelebA, or multiNLI."
        ) from exc


def _representation_label(value: str) -> str:
    return REPRESENTATION_LABELS.get(
        value.lower(), value.replace("_", " ")
    )


def _parse_seed(value: str) -> int | None | str:
    if value.lower() == ALL_SEEDS:
        return ALL_SEEDS
    if value.lower() in {"none", "null", "unseeded"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Seed must be an integer, 'all', or 'none'; received {value!r}."
        ) from exc


def _preset_specs(preset: str) -> tuple[ExperimentSpec, ...]:
    names = ("main", "appendix") if preset == "both" else (preset,)
    return tuple(
        ExperimentSpec(dataset, representation, seed)
        for name in names
        for dataset, representation, seed in EXPERIMENT_PRESETS[name]
    )


def spectral_summary(eigenvalues: np.ndarray) -> tuple[np.ndarray, float]:
    r"""Return cumulative variance and normalized spectral entropy.

    In the paper's notation, for retained empirical eigenvalues
    :math:`\hat\lambda_1,\ldots,\hat\lambda_r`, define

    .. math::

        \hat p_k = \frac{\hat\lambda_k}{\sum_{j=1}^r \hat\lambda_j},
        \qquad
        \mathcal{E}(\hat\lambda_{1:r}) =
        \frac{-\sum_{k=1}^r \hat p_k \log \hat p_k}{\log r}.

    The normalized entropy is one for a uniform retained spectrum and becomes
    smaller as the empirical eigenvalues become more unequal.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("eigenvalues must be finite and strictly positive.")
    if np.any(values[:-1] < values[1:]):
        raise ValueError("eigenvalues must be ordered from largest to smallest.")

    probabilities = values / values.sum()
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = (
        1.0 if values.size == 1 else float(entropy / np.log(values.size))
    )
    return cumulative, normalized_entropy


def fit_spectrum(
    bundle: EmbeddingBundle,
    *,
    relative_tolerance: float | None = None,
) -> SeedSpectrum:
    """Fit empirical whitening on train and retain its covariance eigenvalues."""
    train = bundle.split("train")
    transform = make_transform(
        "whiten",
        whitening_estimator="empirical",
        whitening_relative_tolerance=relative_tolerance,
    ).fit(train.X)
    eigenvalues = np.asarray(transform.eigenvalues_, dtype=np.float64).copy()
    if transform.basis_ is None:
        raise RuntimeError("Empirical whitening did not retain a covariance basis.")
    cumulative, normalized_entropy = spectral_summary(eigenvalues)
    manifest = bundle.manifest
    manifest_dataset = str(manifest["dataset"])
    try:
        dataset = _canonical_dataset(manifest_dataset)
    except ValueError:
        # Keep the numerical helper usable for synthetic/test bundles. The
        # three-panel figure applies the stricter manuscript dataset contract.
        dataset = manifest_dataset
    return SeedSpectrum(
        dataset=dataset,
        representation=str(manifest["representation"]),
        seed=manifest.get("representation_seed"),
        feature_dimension=int(train.X.shape[1]),
        eigenvalues=eigenvalues,
        cumulative_variance=cumulative,
        normalized_spectral_entropy=normalized_entropy,
    )


def _spectrum_cache_path(
    cache_dir: Path,
    *,
    bundle: EmbeddingBundle,
    relative_tolerance: float | None,
) -> Path:
    """Return a fingerprint-keyed cache location for one empirical spectrum."""
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


def _load_spectrum_cache(path: Path, bundle: EmbeddingBundle) -> SeedSpectrum:
    """Load one validated eigenspectrum cache entry."""
    with np.load(path, allow_pickle=False) as cached:
        if "eigenvalues" not in cached:
            raise ValueError(f"Spectrum cache is incomplete: {path}")
        eigenvalues = np.asarray(cached["eigenvalues"], dtype=np.float64)
    cumulative, normalized_entropy = spectral_summary(eigenvalues)
    train = bundle.split("train")
    manifest = bundle.manifest
    try:
        dataset = _canonical_dataset(str(manifest["dataset"]))
    except ValueError:
        dataset = str(manifest["dataset"])
    return SeedSpectrum(
        dataset=dataset,
        representation=str(manifest["representation"]),
        seed=manifest.get("representation_seed"),
        feature_dimension=int(train.X.shape[1]),
        eigenvalues=eigenvalues,
        cumulative_variance=cumulative,
        normalized_spectral_entropy=normalized_entropy,
    )


def average_seed_spectra(seed_spectra: Sequence[SeedSpectrum]) -> Spectrum:
    """Average normalized eigenspectra and entropy across representation seeds."""
    if not seed_spectra:
        raise ValueError("At least one seed-level spectrum is required.")
    first = seed_spectra[0]
    identities = {
        (spectrum.dataset, _normalized_identifier(spectrum.representation))
        for spectrum in seed_spectra
    }
    if len(identities) != 1:
        raise ValueError("Seed spectra must share one dataset and representation.")
    dimensions = {spectrum.feature_dimension for spectrum in seed_spectra}
    if len(dimensions) != 1:
        raise ValueError("Representation dimensions differ across seeds.")

    feature_dimension = first.feature_dimension
    proportions = np.zeros(
        (len(seed_spectra), feature_dimension), dtype=np.float64
    )
    entropies = np.empty(len(seed_spectra), dtype=np.float64)
    for index, spectrum in enumerate(seed_spectra):
        probabilities = spectrum.eigenvalues / spectrum.eigenvalues.sum()
        proportions[index, : spectrum.rank] = probabilities
        entropies[index] = spectrum.normalized_spectral_entropy
    cumulative = np.cumsum(proportions.mean(axis=0))
    cumulative[-1] = 1.0
    entropy_se = (
        float(entropies.std(ddof=1) / np.sqrt(entropies.size))
        if entropies.size > 1
        else 0.0
    )

    return Spectrum(
        dataset=first.dataset,
        representation=first.representation,
        seeds=tuple(spectrum.seed for spectrum in seed_spectra),
        feature_dimension=feature_dimension,
        retained_ranks=tuple(spectrum.rank for spectrum in seed_spectra),
        cumulative_variance=cumulative,
        normalized_spectral_entropy=float(entropies.mean()),
        entropy_standard_error=entropy_se,
    )


def _available_seeds(
    artifact_root: Path,
    *,
    dataset: str,
    representation: str,
) -> tuple[int | str | None, ...]:
    """Discover seeds at the nearest matching bundle depth."""
    if not artifact_root.is_dir():
        raise FileNotFoundError(
            f"Embedding artifact root does not exist: {artifact_root}"
        )
    requested_dataset = _normalized_identifier(dataset)
    requested_representation = _normalized_identifier(representation)
    matches: list[tuple[Path, int | str | None]] = []
    for manifest_path in sorted(artifact_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            _normalized_identifier(manifest.get("dataset")) == requested_dataset
            and _normalized_identifier(manifest.get("representation"))
            == requested_representation
        ):
            matches.append((manifest_path.parent, manifest.get("representation_seed")))
    if not matches:
        raise FileNotFoundError(
            "No embedding bundles match "
            f"dataset={dataset!r}, representation={representation!r}."
        )

    minimum_depth = min(
        len(path.relative_to(artifact_root).parts) for path, _ in matches
    )
    nearest = [
        (path, seed)
        for path, seed in matches
        if len(path.relative_to(artifact_root).parts) == minimum_depth
    ]
    paths_by_seed: dict[int | str | None, list[Path]] = {}
    for path, seed in nearest:
        paths_by_seed.setdefault(seed, []).append(path)
    ambiguous = {
        seed: paths for seed, paths in paths_by_seed.items() if len(paths) > 1
    }
    if ambiguous:
        details = "; ".join(
            f"seed={seed!r}: {', '.join(str(path) for path in paths)}"
            for seed, paths in ambiguous.items()
        )
        raise ValueError(f"Bundle identities are ambiguous: {details}")
    return tuple(
        sorted(
            paths_by_seed,
            key=lambda seed: (
                seed is None,
                isinstance(seed, str),
                str(seed) if isinstance(seed, str) else seed or 0,
            ),
        )
    )


def load_spectra(
    specs: Sequence[ExperimentSpec],
    *,
    artifact_root: str | Path,
    relative_tolerance: float | None = None,
    verify_hashes: bool = False,
    spectrum_cache: str | Path = DEFAULT_SPECTRUM_CACHE,
    refit_spectra: bool = False,
) -> list[Spectrum]:
    """Resolve, fit, and seed-average every requested representation spectrum."""
    root = Path(artifact_root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    cache_dir = Path(spectrum_cache).expanduser().resolve()
    resolved_specs = [
        (
            spec,
            _available_seeds(
                root,
                dataset=spec.dataset,
                representation=spec.representation,
            )
            if spec.seed == ALL_SEEDS
            else (spec.seed,),
        )
        for spec in specs
    ]
    total_seeds = sum(len(seeds) for _, seeds in resolved_specs)
    completed_seeds = 0
    spectra = []
    for spec, seeds in resolved_specs:
        seed_spectra = []
        for seed in seeds:
            completed_seeds += 1
            print(
                f"[spectrum {completed_seeds}/{total_seeds}] processing "
                f"{spec.dataset} / {spec.representation} / seed {seed}",
                flush=True,
            )
            bundle_path = find_bundle(
                root,
                dataset=spec.dataset,
                representation=spec.representation,
                seed=seed,
            )
            bundle = EmbeddingBundle.load(
                bundle_path, verify_hashes=verify_hashes
            )
            cache_path = _spectrum_cache_path(
                cache_dir,
                bundle=bundle,
                relative_tolerance=relative_tolerance,
            )
            if cache_path.is_file() and not refit_spectra:
                print(
                    f"[spectrum {completed_seeds}/{total_seeds}] loading cached "
                    f"{spec.dataset} / {spec.representation} / seed {seed}",
                    flush=True,
                )
                fitted = _load_spectrum_cache(cache_path, bundle)
            else:
                fitted = fit_spectrum(bundle, relative_tolerance=relative_tolerance)
                _save_coefficient_cache(
                    cache_path, eigenvalues=fitted.eigenvalues
                )
            seed_spectra.append(fitted)
            print(
                f"[spectrum {completed_seeds}/{total_seeds}] complete "
                f"(rank {fitted.rank})",
                flush=True,
            )
        spectra.append(average_seed_spectra(seed_spectra))
    return spectra


def coefficient_simplicity(
    coef: np.ndarray,
    *,
    covariance_basis: np.ndarray,
    covariance_eigenvalues: np.ndarray,
) -> float:
    r"""Return the mean coefficient Rayleigh quotient for one fitted head.

    Binary heads contribute their single logit direction. Multiclass heads
    contribute every pairwise logit-coefficient difference, which makes the
    score invariant to a common shift of all softmax coefficient vectors.
    """
    coef = np.asarray(coef, dtype=np.float64)
    basis = np.asarray(covariance_basis, dtype=np.float64)
    eigenvalues = np.asarray(covariance_eigenvalues, dtype=np.float64)
    if coef.ndim != 2 or coef.shape[1] != basis.shape[0]:
        raise ValueError("Head coefficients and covariance basis do not align.")
    if basis.ndim != 2 or eigenvalues.shape != (basis.shape[1],):
        raise ValueError("Covariance basis and eigenvalues do not align.")
    directions = (
        [coef[0]]
        if coef.shape[0] == 1
        else [
            coef[second] - coef[first]
            for first in range(coef.shape[0])
            for second in range(first + 1, coef.shape[0])
        ]
    )
    quotients = []
    for direction in directions:
        denominator = float(direction @ direction)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("A fitted coefficient direction has invalid norm.")
        coordinates = basis.T @ direction
        numerator = float(eigenvalues @ (coordinates**2))
        quotient = numerator / denominator
        if not np.isfinite(quotient):
            raise ValueError("A fitted coefficient direction has invalid simplicity.")
        quotients.append(quotient)
    return float(np.mean(quotients))


def direct_coefficient_simplicity(coef: np.ndarray, X: np.ndarray) -> float:
    """Calculate coefficient simplicity without materializing covariance."""
    coef = np.asarray(coef, dtype=np.float64)
    X = np.asarray(X, dtype=np.float32)
    if coef.ndim != 2 or X.ndim != 2 or coef.shape[1] != X.shape[1]:
        raise ValueError("Head coefficients and representation matrix do not align.")
    directions = (
        [coef[0]]
        if coef.shape[0] == 1
        else [
            coef[second] - coef[first]
            for first in range(coef.shape[0])
            for second in range(first + 1, coef.shape[0])
        ]
    )
    centered = X - X.mean(axis=0, dtype=np.float32)
    quotients = []
    for direction in directions:
        denominator = float(direction @ direction)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("A fitted coefficient direction has invalid norm.")
        projection = centered @ direction
        quotient = float(
            projection @ projection / (len(X) - 1) / denominator
        )
        if not np.isfinite(quotient):
            raise ValueError("A fitted coefficient direction has invalid simplicity.")
        quotients.append(quotient)
    return float(np.mean(quotients))


def _coefficient_cache_path(
    cache_dir: Path,
    *,
    bundle_identity: Mapping[str, Any],
    class_balanced: bool,
    identity_ridge: float,
    whiten_ridge: float,
    relative_tolerance: float | None,
    selection_criterion: str,
) -> Path:
    key = {
        "version": COEFFICIENT_CACHE_VERSION,
        "bundle_fingerprint": bundle_identity["fingerprint"],
        "class_balanced": class_balanced,
        "identity_ridge": identity_ridge,
        "whiten_ridge": whiten_ridge,
        "whitening_estimator": BARPLOT_WHITENING_ESTIMATOR,
        "relative_tolerance": relative_tolerance,
        "selection_criterion": selection_criterion,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    dataset = _normalized_identifier(bundle_identity["dataset"])
    representation = _normalized_identifier(bundle_identity["representation"])
    seed = bundle_identity.get(
        "split_seed", bundle_identity.get("representation_seed")
    )
    return cache_dir / f"{dataset}_{representation}_seed_{seed}_{digest}.npz"


def _standardize_cache_path(
    cache_dir: Path,
    *,
    bundle_identity: Mapping[str, Any],
    class_balanced: bool,
    ridge: float,
    selection_criterion: str,
) -> Path:
    key = {
        "version": STANDARDIZE_CACHE_VERSION,
        "bundle_fingerprint": bundle_identity["fingerprint"],
        "class_balanced": class_balanced,
        "ridge": ridge,
        "selection_criterion": selection_criterion,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    dataset = _normalized_identifier(bundle_identity["dataset"])
    representation = _normalized_identifier(bundle_identity["representation"])
    seed = bundle_identity.get(
        "split_seed", bundle_identity.get("representation_seed")
    )
    return cache_dir / (
        f"{dataset}_{representation}_seed_{seed}_standardize_{digest}.npz"
    )


def _load_coefficient_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        result = {name: np.asarray(cached[name]) for name in cached.files}
    required = {
        "identity_coef",
        "whiten_coef",
        "identity_simplicity",
        "whiten_simplicity",
        "maximum_eigenvalue",
        "identity_validation_metric",
        "whiten_validation_metric",
    }
    if not required.issubset(result):
        missing = ", ".join(sorted(required - result.keys()))
        raise ValueError(f"Coefficient cache is incomplete; missing {missing}: {path}")
    if any(not np.isfinite(result[name]).all() for name in required):
        raise ValueError(f"Coefficient cache contains non-finite values: {path}")
    return result


def _save_coefficient_cache(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.stem + "_",
        suffix=".npz",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **values)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_selected_bundle(bundle_identity: Mapping[str, Any]) -> EmbeddingBundle:
    """Load a saved bundle, reconstructing a frozen train/validation split."""
    source = EmbeddingBundle.load(bundle_identity["path"])
    frozen_split = bundle_identity.get("frozen_split")
    if frozen_split is not None:
        from frozen_results import resplit_bundle

        split_seed = bundle_identity.get("split_seed", frozen_split.get("split_seed"))
        if split_seed is None:
            raise ValueError("A frozen selected run is missing its split seed.")
        source = resplit_bundle(source, int(split_seed))
    if source.fingerprint != bundle_identity["fingerprint"]:
        raise ValueError("Selected sweep bundle fingerprint has changed.")
    return source


def _selected_seed(bundle_identity: Mapping[str, Any]) -> int | str | None:
    """Return the replicate identifier used by a selected sweep run."""
    return bundle_identity.get(
        "split_seed", bundle_identity.get("representation_seed")
    )


def load_head_simplicities(
    sweep_paths: Sequence[str | Path],
    *,
    criterion: str = "class_balanced_accuracy",
    relative_tolerance: float | None = None,
    coefficient_cache: str | Path = DEFAULT_COEFFICIENT_CACHE,
    refit_coefficients: bool = False,
    trace_division: bool = False,
) -> list[HeadSimplicity]:
    """Refit selected heads and normalize their Rayleigh quotients."""
    metric_name = {
        "accuracy": "accuracy",
        "class_balanced_accuracy": "balanced_accuracy",
        "log_loss": "log_loss",
        "balanced_log_loss": "balanced_log_loss",
    }.get(criterion)
    if metric_name is None:
        raise ValueError(
            "Coefficient refit verification requires a single validation criterion."
        )
    resolved = [
        path if Path(path).is_absolute() else ROOT / path
        for path in sweep_paths
    ]
    runs, _ = load_sweep_runs(resolved)
    report = aggregate_runs(runs, default_selection_rule=criterion)
    cache_dir = Path(coefficient_cache).expanduser().resolve()
    total_seeds = sum(
        len(method["selected"])
        for experiment in report["experiments"]
        for method in experiment["methods"]
        if method["method"] == BARPLOT_WHITENING_METHOD
    )
    completed_seeds = 0
    completed_standardize_seeds = 0
    summaries = []
    for experiment in report["experiments"]:
        selected_by_method = {
            method["method"]: method["selected"]
            for method in experiment["methods"]
        }
        if (
            "identity" not in selected_by_method
            or "standardize" not in selected_by_method
            or BARPLOT_WHITENING_METHOD not in selected_by_method
        ):
            raise ValueError(
                "Coefficient bar plots require selected identity, standardization, "
                "and nonlinear Ledoit--Wolf whitening runs."
            )
        identity_by_fingerprint = {
            selected["bundle"]["fingerprint"]: selected
            for selected in selected_by_method["identity"]
        }
        identity_scores = []
        standardize_scores = []
        whiten_scores = []
        random_direction_scores = []
        seeds = []
        normalization_denominator_by_fingerprint = {}
        for whiten_selected in selected_by_method[BARPLOT_WHITENING_METHOD]:
            bundle_identity = whiten_selected["bundle"]
            fingerprint = bundle_identity["fingerprint"]
            try:
                identity_selected = identity_by_fingerprint[fingerprint]
            except KeyError as exc:
                raise ValueError(
                    "Identity and whitening selections do not share matched seeds."
                ) from exc
            completed_seeds += 1
            dataset_label = DATASET_LABELS[
                _canonical_dataset(experiment["dataset"])
            ]
            identity_ridge = float(identity_selected["ridge_lambda"])
            whiten_ridge = float(whiten_selected["ridge_lambda"])
            class_balanced = bool(experiment["class_balanced"])
            cache_path = _coefficient_cache_path(
                cache_dir,
                bundle_identity=bundle_identity,
                class_balanced=class_balanced,
                identity_ridge=identity_ridge,
                whiten_ridge=whiten_ridge,
                relative_tolerance=relative_tolerance,
                selection_criterion=criterion,
            )
            progress = (
                f"[{completed_seeds}/{total_seeds}] {dataset_label} "
                f"seed {_selected_seed(bundle_identity)} "
                f"(identity ridge={identity_ridge:g}, "
                f"whiten ridge={whiten_ridge:g})"
            )
            if cache_path.is_file() and not refit_coefficients:
                print(f"{progress}: loading cached coefficients", flush=True)
                cached = _load_coefficient_cache(cache_path)
                for method, selected, cached_name in (
                    ("identity", identity_selected, "identity_validation_metric"),
                    ("whiten", whiten_selected, "whiten_validation_metric"),
                ):
                    if not np.isclose(
                        float(cached[cached_name]),
                        float(selected["validation_metric"]),
                        rtol=1e-6,
                        atol=1e-8,
                    ):
                        raise ValueError(
                            f"Cached {method} validation metric does not match "
                            f"the selected results_paper value: {cache_path}"
                        )
                maximum_eigenvalue = float(cached["maximum_eigenvalue"])
                if maximum_eigenvalue <= 0.0:
                    raise ValueError(
                        f"Cached maximum eigenvalue must be positive: {cache_path}"
                    )
                if trace_division:
                    feature_dimension = int(cached["identity_coef"].shape[-1])
                    if "random_direction_simplicity" in cached:
                        covariance_trace = (
                            float(cached["random_direction_simplicity"])
                            * feature_dimension
                            * maximum_eigenvalue
                        )
                    else:
                        bundle = _load_selected_bundle(bundle_identity)
                        train = bundle.split("train")
                        covariance_trace = float(
                            np.asarray(train.X, dtype=np.float32)
                            .var(axis=0, ddof=1, dtype=np.float32)
                            .sum(dtype=np.float64)
                        )
                    normalization_denominator = covariance_trace
                else:
                    normalization_denominator = maximum_eigenvalue
                if normalization_denominator <= 0.0:
                    raise ValueError("Covariance normalization denominator must be positive.")
                normalization_denominator_by_fingerprint[fingerprint] = (
                    normalization_denominator
                )
                identity_scores.append(
                    float(cached["identity_simplicity"]) / normalization_denominator
                )
                whiten_scores.append(
                    float(cached["whiten_simplicity"]) / normalization_denominator
                )
                if trace_division:
                    random_direction_simplicity = 1.0 / feature_dimension
                elif "random_direction_simplicity" in cached:
                    random_direction_simplicity = float(
                        cached["random_direction_simplicity"]
                    )
                else:
                    bundle = _load_selected_bundle(bundle_identity)
                    train = bundle.split("train")
                    covariance_trace = float(
                        np.asarray(train.X, dtype=np.float32)
                        .var(axis=0, ddof=1, dtype=np.float32)
                        .sum(dtype=np.float64)
                    )
                    random_direction_simplicity = covariance_trace / (
                        train.X.shape[1] * maximum_eigenvalue
                    )
                    cached["random_direction_simplicity"] = np.asarray(
                        random_direction_simplicity
                    )
                    _save_coefficient_cache(cache_path, **cached)
                random_direction_scores.append(random_direction_simplicity)
                seeds.append(_selected_seed(bundle_identity))
                continue

            print(f"{progress}: fitting selected heads", flush=True)
            bundle = _load_selected_bundle(bundle_identity)
            train = bundle.split("train")
            validation = bundle.split("val")
            whitening = make_transform(
                "whiten",
                whitening_estimator=BARPLOT_WHITENING_ESTIMATOR,
                whitening_relative_tolerance=relative_tolerance,
            ).fit(train.X)
            if whitening.basis_ is None or whitening.scales_ is None:
                raise RuntimeError("Whitening did not retain a covariance basis and scales.")
            whitened_train = whitening.transform(train.X)
            empirical_eigenvalues = (
                whitened_train.var(axis=0, ddof=1, dtype=np.float32)
                * whitening.scales_**2
            )
            identity_fit = fit_logistic(
                train.X,
                train.y,
                LogisticConfig(
                    ridge_lambda=identity_ridge,
                    class_balanced=class_balanced,
                ),
            )
            whiten_fit = fit_logistic(
                whitened_train,
                train.y,
                LogisticConfig(
                    ridge_lambda=whiten_ridge,
                    class_balanced=class_balanced,
                ),
            )
            whiten_coef, _ = whitening.inverse_head(
                whiten_fit.estimator.coef_, whiten_fit.estimator.intercept_
            )
            identity_validation_metric = selection_metrics(
                identity_fit.estimator, validation.X, validation.y
            )[metric_name]
            whiten_validation_metric = selection_metrics(
                whiten_fit.estimator,
                whitening.transform(validation.X),
                validation.y,
            )[metric_name]
            for method, observed, selected in (
                ("identity", identity_validation_metric, identity_selected),
                ("whiten", whiten_validation_metric, whiten_selected),
            ):
                expected = float(selected["validation_metric"])
                if not np.isclose(observed, expected, rtol=1e-6, atol=1e-8):
                    raise RuntimeError(
                        f"Refitted {method} validation metric for {dataset_label} "
                        f"seed {bundle_identity.get('representation_seed')} does not "
                        f"match results_paper: observed {observed:.10g}, "
                        f"expected {expected:.10g}."
                    )
            score_args = {
                "covariance_basis": whitening.basis_,
                "covariance_eigenvalues": empirical_eigenvalues,
            }
            identity_simplicity = coefficient_simplicity(
                identity_fit.estimator.coef_, **score_args
            )
            whiten_simplicity = coefficient_simplicity(whiten_coef, **score_args)
            maximum_eigenvalue = float(empirical_eigenvalues[0])
            covariance_trace = float(
                np.asarray(train.X, dtype=np.float32)
                .var(axis=0, ddof=1, dtype=np.float32)
                .sum(dtype=np.float64)
            )
            normalization_denominator = (
                covariance_trace if trace_division else maximum_eigenvalue
            )
            normalization_denominator_by_fingerprint[fingerprint] = (
                normalization_denominator
            )
            random_direction_simplicity = covariance_trace / (
                train.X.shape[1] * normalization_denominator
            )
            identity_scores.append(identity_simplicity / normalization_denominator)
            whiten_scores.append(whiten_simplicity / normalization_denominator)
            random_direction_scores.append(random_direction_simplicity)
            seeds.append(_selected_seed(bundle_identity))
            _save_coefficient_cache(
                cache_path,
                identity_coef=identity_fit.estimator.coef_,
                whiten_coef=whiten_coef,
                identity_simplicity=identity_simplicity,
                whiten_simplicity=whiten_simplicity,
                maximum_eigenvalue=maximum_eigenvalue,
                identity_validation_metric=identity_validation_metric,
                whiten_validation_metric=whiten_validation_metric,
                random_direction_simplicity=random_direction_simplicity,
            )
            print(f"{progress}: cached at {cache_path}", flush=True)

        standardize_by_fingerprint = {
            selected["bundle"]["fingerprint"]: selected
            for selected in selected_by_method["standardize"]
        }
        for fingerprint in normalization_denominator_by_fingerprint:
            completed_standardize_seeds += 1
            try:
                standardize_selected = standardize_by_fingerprint[fingerprint]
            except KeyError as exc:
                raise ValueError(
                    "Standardization and whitening selections do not share matched seeds."
                ) from exc
            bundle_identity = standardize_selected["bundle"]
            standardize_ridge = float(standardize_selected["ridge_lambda"])
            class_balanced = bool(experiment["class_balanced"])
            cache_path = _standardize_cache_path(
                cache_dir,
                bundle_identity=bundle_identity,
                class_balanced=class_balanced,
                ridge=standardize_ridge,
                selection_criterion=criterion,
            )
            dataset_label = DATASET_LABELS[
                _canonical_dataset(experiment["dataset"])
            ]
            progress = (
                f"[{completed_standardize_seeds}/{total_seeds}] "
                f"{dataset_label} seed {_selected_seed(bundle_identity)} "
                f"(standardize ridge={standardize_ridge:g})"
            )
            if cache_path.is_file() and not refit_coefficients:
                print(f"{progress}: loading cached standardization head", flush=True)
                with np.load(cache_path, allow_pickle=False) as cached_file:
                    cached = {
                        name: np.asarray(cached_file[name])
                        for name in cached_file.files
                    }
                required = {
                    "standardize_coef",
                    "standardize_simplicity",
                    "standardize_validation_metric",
                }
                if not required.issubset(cached):
                    missing = ", ".join(sorted(required - cached.keys()))
                    raise ValueError(
                        "Standardization cache is incomplete; missing "
                        f"{missing}: {cache_path}"
                    )
                if any(not np.isfinite(cached[name]).all() for name in required):
                    raise ValueError(
                        f"Standardization cache contains non-finite values: {cache_path}"
                    )
                if not np.isclose(
                    float(cached["standardize_validation_metric"]),
                    float(standardize_selected["validation_metric"]),
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError(
                        "Cached standardization validation metric does not match "
                        f"the selected results_paper value: {cache_path}"
                    )
                standardize_simplicity = float(cached["standardize_simplicity"])
            else:
                print(f"{progress}: fitting selected standardization head", flush=True)
                bundle = _load_selected_bundle(bundle_identity)
                train = bundle.split("train")
                validation = bundle.split("val")
                standardize = make_transform("standardize").fit(train.X)
                standardize_fit = fit_logistic(
                    standardize.transform(train.X),
                    train.y,
                    LogisticConfig(
                        ridge_lambda=standardize_ridge,
                        class_balanced=class_balanced,
                    ),
                )
                standardize_coef, _ = standardize.inverse_head(
                    standardize_fit.estimator.coef_,
                    standardize_fit.estimator.intercept_,
                )
                standardize_validation_metric = selection_metrics(
                    standardize_fit.estimator,
                    standardize.transform(validation.X),
                    validation.y,
                )[metric_name]
                expected = float(standardize_selected["validation_metric"])
                if not np.isclose(
                    standardize_validation_metric,
                    expected,
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise RuntimeError(
                        f"Refitted standardization validation metric for {dataset_label} "
                        f"seed {bundle_identity.get('representation_seed')} does not "
                        f"match results_paper: observed "
                        f"{standardize_validation_metric:.10g}, expected {expected:.10g}."
                    )
                standardize_simplicity = direct_coefficient_simplicity(
                    standardize_coef, train.X
                )
                _save_coefficient_cache(
                    cache_path,
                    standardize_coef=standardize_coef,
                    standardize_simplicity=standardize_simplicity,
                    standardize_validation_metric=standardize_validation_metric,
                )
                print(f"{progress}: cached at {cache_path}", flush=True)
            standardize_scores.append(
                standardize_simplicity
                / normalization_denominator_by_fingerprint[fingerprint]
            )
        summaries.append(
            HeadSimplicity(
                dataset=_canonical_dataset(experiment["dataset"]),
                representation=str(experiment["representation"]),
                seeds=tuple(seeds),
                identity_scores=tuple(identity_scores),
                standardize_scores=tuple(standardize_scores),
                whiten_scores=tuple(whiten_scores),
                random_direction_scores=tuple(random_direction_scores),
            )
        )
    return summaries


def _comparison_cache_path(
    cache_dir: Path, *, method: str, selected_run: Mapping[str, Any]
) -> Path:
    """Name a cache entry by the exact validation-selected comparison run."""
    bundle = selected_run["bundle"]
    key = {
        "version": COMPARISON_COEFFICIENT_CACHE_VERSION,
        "method": method,
        "selected_run": selected_run,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / (
        f"comparison_{method}_{_normalized_identifier(bundle['dataset'])}_"
        f"seed_{bundle.get('representation_seed')}_{digest}.npz"
    )


def _comparison_aggregate_path(relative_sweep_path: str, family: str) -> Path:
    """Return the aggregate report that supplies a comparison panel's heads."""
    sweep_path = ROOT / relative_sweep_path
    suffix = "_aggregate_cb.json" if family == "neurotune" else "_aggregate.json"
    return sweep_path.with_name(sweep_path.stem + suffix)


def _comparison_score(
    coefficient: np.ndarray, X: np.ndarray
) -> tuple[float, float]:
    """Score an original-coordinate head against its fitting covariance."""
    transform = make_transform("whiten", whitening_estimator="empirical").fit(
        X
    )
    if transform.eigenvalues_ is None or transform.eigenvalues_.size == 0:
        raise RuntimeError("Fitting covariance has no retained eigenvalues.")
    maximum_eigenvalue = float(transform.eigenvalues_[0])
    return direct_coefficient_simplicity(coefficient, X), maximum_eigenvalue


def _comparison_fitting_representations(
    bundle: EmbeddingBundle, run: Mapping[str, Any]
) -> np.ndarray:
    """Return the representations used to fit one final comparison head."""
    validation = bundle.split("val")
    if run.get("method_family") != "neurotune":
        return validation.X
    protocol = run.get("protocol", {})
    version = int(protocol.get("neurotune_protocol_version", 1))
    if version < 2:
        return bundle.split("train").X
    partition = protocol.get("validation_partition", {})
    _, tuning_indices, _ = _neurotune_validation_partition(
        validation.y, random_state=int(partition.get("random_state", 0))
    )
    return validation.X[tuning_indices]


def _raw_selected_comparison_run(
    runs: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], *, method: str
) -> dict[str, Any]:
    """Recover aggregate's selected full run from its compact report record."""
    fingerprint = str(summary["bundle"]["fingerprint"])
    ridge = float(summary["ridge_lambda"])
    matches = [
        run for run in runs
        if str(run.get("method")) == method
        and str(run.get("bundle", {}).get("fingerprint")) == fingerprint
        and np.isclose(float(run.get("ridge_lambda")), ridge, rtol=0.0, atol=0.0)
    ]
    for key, run_key in (("afr_gamma", "afr_gamma"), ("neurotune_threshold", "neurotune")):
        value = summary.get(key)
        if value is None:
            continue
        if run_key == "neurotune":
            matches = [
                run for run in matches
                if np.isclose(float(run.get("neurotune", {}).get("threshold")), float(value))
            ]
        else:
            matches = [
                run for run in matches
                if np.isclose(float(run.get(run_key)), float(value))
            ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not recover one raw selected run for {method} "
            f"(fingerprint={fingerprint}); found {len(matches)}."
        )
    return dict(matches[0])


def load_comparison_simplicities(
    *,
    coefficient_cache: str | Path = DEFAULT_COEFFICIENT_CACHE,
    refit_coefficients: bool = False,
    methods: Sequence[str] = tuple(COMPARISON_SWEEP_PATHS),
    trace_division: bool = False,
) -> dict[str, list[HeadSimplicity]]:
    """Refit and cache the aggregate-selected comparison heads for plotting.

    DFR and AFR use the selected classifiers from their ``*_aggregate.json``
    reports. NeuroTune uses the class-balanced selections in
    ``*_NEUROTUNE_aggregate_cb.json`` and retains its disjoint tuning half.
    """
    methods = tuple(methods)
    unknown = sorted(set(methods).difference(COMPARISON_SWEEP_PATHS))
    if unknown:
        raise ValueError("Unknown comparison method(s): " + ", ".join(unknown))
    cache_dir = Path(coefficient_cache).expanduser().resolve()
    bundle_by_fingerprint: dict[str, EmbeddingBundle] = {}
    trace_by_fitting_covariance: dict[tuple[Any, ...], float] = {}

    def covariance_trace(run: Mapping[str, Any]) -> float:
        """Reuse each distinct comparison fitting covariance within one render."""
        fingerprint = str(run["bundle"]["fingerprint"])
        if run.get("method_family") == "neurotune":
            protocol = run.get("protocol", {})
            partition = protocol.get("validation_partition", {})
            covariance_key = (
                fingerprint,
                "neurotune",
                int(protocol.get("neurotune_protocol_version", 1)),
                int(partition.get("random_state", 0)),
            )
        else:
            covariance_key = (fingerprint, "validation")
        cached_trace = trace_by_fitting_covariance.get(covariance_key)
        if cached_trace is not None:
            return cached_trace
        bundle = bundle_by_fingerprint.get(fingerprint)
        if bundle is None:
            bundle = EmbeddingBundle.load(run["bundle"]["path"])
            bundle_by_fingerprint[fingerprint] = bundle
        fitting_X = _comparison_fitting_representations(bundle, run)
        trace = float(
            np.asarray(fitting_X, dtype=np.float32)
            .var(axis=0, ddof=1, dtype=np.float32)
            .sum(dtype=np.float64)
        )
        if trace <= 0.0 or not np.isfinite(trace):
            raise ValueError("Comparison covariance trace must be finite and positive.")
        trace_by_fitting_covariance[covariance_key] = trace
        return trace

    selected_by_method: dict[str, list[dict[str, Any]]] = {}
    method_dataset: dict[tuple[str, str], str] = {}
    source_items = tuple(
        (family, dataset, relative_path)
        for family in methods
        for dataset, relative_path in COMPARISON_SWEEP_PATHS[family].items()
    )
    for source_index, (family, dataset, relative_path) in enumerate(
        source_items, start=1
    ):
        print(
            f"[source {source_index}/{len(source_items)}] loading {family.upper()} "
            f"selections for {DATASET_LABELS[dataset]}",
            flush=True,
        )
        runs, _ = load_sweep_runs([ROOT / relative_path])
        aggregate_path = _comparison_aggregate_path(relative_path, family)
        if not aggregate_path.is_file():
            raise FileNotFoundError(
                f"Missing comparison aggregate report: {aggregate_path}"
            )
        report = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if report.get("kind") != "ridge_lambda_aggregate":
            raise ValueError(f"Not an aggregate.py report: {aggregate_path}")
        experiments = report["experiments"]
        if len(experiments) != 1:
            raise ValueError(f"Comparison sweep must contain one experiment: {relative_path}")
        selected_count = 0
        for method_report in experiments[0]["methods"]:
            method = str(method_report["method"])
            transform_name = method.removeprefix(family + "_")
            if transform_name not in COMPARISON_PLOTTED_TRANSFORMS:
                continue
            selected_summaries = method_report["selected"]
            if not selected_summaries:
                raise ValueError(f"Comparison sweep selected no runs: {relative_path}")
            selected = [
                _raw_selected_comparison_run(
                    runs, summary, method=method
                )
                for summary in selected_summaries
            ]
            selected_by_method.setdefault(method, []).extend(selected)
            selected_count += len(selected)
            for run in selected:
                method_dataset[(method, str(run["bundle"]["fingerprint"]))] = dataset
        print(
            f"[source {source_index}/{len(source_items)}] selected "
            f"{selected_count} displayed heads",
            flush=True,
        )

    cached_scores: dict[tuple[str, str], float] = {}
    missing: dict[str, list[dict[str, Any]]] = {}
    total_selected = sum(
        len(selected_runs) for selected_runs in selected_by_method.values()
    )
    completed_selected = 0
    for method, selected_runs in selected_by_method.items():
        for run in selected_runs:
            completed_selected += 1
            fingerprint = str(run["bundle"]["fingerprint"])
            path = _comparison_cache_path(cache_dir, method=method, selected_run=run)
            dataset = method_dataset[(method, fingerprint)]
            seed = run["bundle"].get("representation_seed")
            if path.is_file() and not refit_coefficients:
                with np.load(path, allow_pickle=False) as cached:
                    required = {"simplicity", "maximum_eigenvalue"}
                    if not required.issubset(cached.files):
                        raise ValueError(f"Comparison cache is incomplete: {path}")
                    simplicity = float(cached["simplicity"])
                    maximum = float(cached["maximum_eigenvalue"])
                if not np.isfinite(simplicity) or not np.isfinite(maximum) or maximum <= 0:
                    raise ValueError(f"Comparison cache is invalid: {path}")
                if trace_division:
                    denominator = covariance_trace(run)
                else:
                    denominator = maximum
                if denominator <= 0.0 or not np.isfinite(denominator):
                    raise ValueError(f"Comparison covariance trace is invalid: {path}")
                cached_scores[(method, fingerprint)] = simplicity / denominator
                print(
                    f"[{completed_selected}/{total_selected}] loaded {method} "
                    f"for {DATASET_LABELS[dataset]} seed {seed}",
                    flush=True,
                )
            else:
                missing.setdefault(method, []).append(run)

    if missing:
        pending = [
            (method, run)
            for method in sorted(missing)
            for run in sorted(
                missing[method],
                key=lambda item: str(item["bundle"].get("fingerprint")),
            )
        ]
        print(
            f"Refitting {len(pending)} selected comparison heads for simplicity scores",
            flush=True,
        )

        completed_refits = 0

        def progress_for(
            batch_pending: Sequence[tuple[str, dict[str, Any]]],
        ) -> Callable[[int, int, str], None]:
            def report_refit_progress(
                completed: int, total: int, method: str
            ) -> None:
                nonlocal completed_refits
                if total != len(batch_pending):
                    raise RuntimeError("Comparison refit progress total changed.")
                expected_method, run = batch_pending[completed - 1]
                if method != expected_method:
                    raise RuntimeError("Comparison refit progress method order changed.")
                completed_refits += 1
                fingerprint = str(run["bundle"]["fingerprint"])
                dataset = method_dataset[(method, fingerprint)]
                seed = run["bundle"].get("representation_seed")
                print(
                    f"[refit {completed_refits}/{len(pending)}] completed {method} for "
                    f"{DATASET_LABELS[dataset]} seed {seed}",
                    flush=True,
                )

            return report_refit_progress

        # DFR and AFR report final heads refit from the full validation pool.
        # NeuroTune retains its paper-protocol split: one half identifies the
        # mask and the disjoint tuning half fits the last layer.
        records: dict[str, dict[str, Any]] = {}
        for selected_runs, refit_train_val in (
            (
                {
                    method: runs
                    for method, runs in missing.items()
                    if not method.startswith("neurotune_")
                },
                True,
            ),
            (
                {
                    method: runs
                    for method, runs in missing.items()
                    if method.startswith("neurotune_")
                },
                False,
            ),
        ):
            if not selected_runs:
                continue
            batch_pending = [
                (method, run)
                for method in sorted(selected_runs)
                for run in sorted(
                    selected_runs[method],
                    key=lambda item: str(item["bundle"].get("fingerprint")),
                )
            ]
            refitted = _refit_selected_test_predictions_serial(
                selected_runs,
                refit_train_val=refit_train_val,
                progress=progress_for(batch_pending),
            )
            for fingerprint, record in refitted.items():
                merged = records.setdefault(fingerprint, {"coefficients": {}})
                merged["coefficients"].update(record.get("coefficients", {}))
        for method, selected_runs in missing.items():
            for run in selected_runs:
                fingerprint = str(run["bundle"]["fingerprint"])
                record = records[fingerprint]
                coefficient = record.get("coefficients", {}).get(method)
                if coefficient is None:
                    raise RuntimeError(f"Refit did not expose coefficients for {method}.")
                bundle = EmbeddingBundle.load(run["bundle"]["path"])
                fitting_X = _comparison_fitting_representations(bundle, run)
                simplicity, maximum = _comparison_score(coefficient, fitting_X)
                path = _comparison_cache_path(cache_dir, method=method, selected_run=run)
                _save_coefficient_cache(
                    path,
                    simplicity=simplicity,
                    maximum_eigenvalue=maximum,
                )
                if trace_division:
                    denominator = covariance_trace(run)
                else:
                    denominator = maximum
                cached_scores[(method, fingerprint)] = simplicity / denominator
                print(f"Cached {method} seed {run['bundle'].get('representation_seed')}: {path}", flush=True)

    result: dict[str, list[HeadSimplicity]] = {}
    for family in methods:
        summaries = []
        prefix = family
        for dataset in DATASET_ORDER:
            matching = [
                (method, run)
                for method, runs in selected_by_method.items()
                for run in runs
                if method_dataset[(method, str(run["bundle"]["fingerprint"]))] == dataset
                and method.startswith(prefix + "_")
            ]
            if not matching:
                raise ValueError(f"No selected {family} runs for {dataset}.")
            matching.sort(key=lambda item: str(item[1]["bundle"].get("representation_seed")))
            by_transform: dict[str, list[float]] = {}
            seed_by_transform: dict[str, list[int | str | None]] = {}
            for method, run in matching:
                transform_name = method.removeprefix(prefix + "_")
                by_transform.setdefault(transform_name, []).append(
                    cached_scores[(method, str(run["bundle"]["fingerprint"]))]
                )
                seed_by_transform.setdefault(transform_name, []).append(
                    run["bundle"].get("representation_seed")
                )
            required = set(COMPARISON_PLOTTED_TRANSFORMS)
            if set(by_transform) != required:
                raise ValueError(f"Unexpected selected {family} transforms for {dataset}: {sorted(by_transform)}")
            summaries.append(HeadSimplicity(
                dataset=dataset,
                representation="comparison",
                seeds=tuple(seed_by_transform["identity"]),
                identity_scores=tuple(by_transform["identity"]),
                standardize_scores=tuple(by_transform["standardize"]),
                whiten_scores=tuple(by_transform["whiten_ledoit_wolf_nonlinear"]),
                random_direction_scores=tuple(0.0 for _ in by_transform["identity"]),
            ))
        result[family] = summaries
    return result



def _displayed_cumulative_variance(spectrum: Spectrum) -> np.ndarray:
    """Return cumulative variance for at most the leading 500 eigenvalues."""
    return spectrum.cumulative_variance[:MAX_DISPLAY_EIGENVALUES]


def _draw_spectrum(axis: plt.Axes, spectra: Sequence[Spectrum]) -> None:
    """Draw cumulative scree curves on one supplied axis."""
    if not spectra:
        raise ValueError("At least one spectrum is required.")
    grouped = {dataset: [] for dataset in DATASET_ORDER}
    for spectrum in spectra:
        if spectrum.dataset not in grouped:
            raise ValueError(f"Unexpected dataset in spectrum: {spectrum.dataset!r}.")
        grouped[spectrum.dataset].append(spectrum)
    missing = [dataset for dataset, values in grouped.items() if not values]
    if missing:
        raise ValueError(
            "A spectrum is required for every dataset; missing " + ", ".join(missing)
        )

    line_styles = ("-", "--", ":", "-.")
    for dataset in DATASET_ORDER:
        dataset_spectra = grouped[dataset]
        for index, spectrum in enumerate(dataset_spectra):
            label = DATASET_LABELS[dataset]
            if len(dataset_spectra) > 1:
                label += f" — {_representation_label(spectrum.representation)}"
            values = _displayed_cumulative_variance(spectrum)
            axis.plot(
                np.arange(1, values.size + 1),
                values,
                color=DATASET_COLORS[dataset],
                label=label,
                linestyle=line_styles[index % len(line_styles)],
                linewidth=2.6,
            )

    axis.set_xlim(1, MAX_DISPLAY_EIGENVALUES)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Number of top eigenvalues included", fontsize=PLOT_LABEL_FONT_SIZE)
    axis.set_ylabel(
        "Variance explained",
        fontsize=PLOT_LABEL_FONT_SIZE,
    )
    axis.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    axis.legend(loc="lower right", fontsize=PLOT_LEGEND_FONT_SIZE)
def _draw_simplicity(
    figure: plt.Figure,
    axis: plt.Axes,
    summaries: Sequence[HeadSimplicity],
    *,
    include_random_direction: bool = True,
    method_label: str | None = None,
    show_ylabel: bool = True,
    annotation_font_size: float = PLOT_ANNOTATION_FONT_SIZE,
    trace_division: bool = False,
    y_upper_limit: float | None = None,
    dataset_order: Sequence[str] = DATASET_ORDER,
) -> None:
    """Draw seed-averaged simplicity of the fitted head variants."""
    dataset_order = tuple(dataset_order)
    if len(summaries) != len(dataset_order):
        raise ValueError("A coefficient summary is required for every dataset.")
    by_dataset = {summary.dataset: summary for summary in summaries}
    if len(by_dataset) != len(summaries):
        raise ValueError("Coefficient summaries must contain each dataset once.")
    identity_values = []
    standardize_values = []
    whiten_values = []
    identity_intervals = []
    standardize_intervals = []
    whiten_intervals = []
    random_direction_values = []
    for dataset in dataset_order:
        summary = by_dataset.get(dataset)
        if summary is None:
            raise ValueError(f"Missing coefficient summary for {dataset}.")
        identity_values.append(summary.identity_mean)
        standardize_values.append(summary.standardize_mean)
        whiten_values.append(summary.whiten_mean)
        identity_intervals.append(summary.identity_interval)
        standardize_intervals.append(summary.standardize_interval)
        whiten_intervals.append(summary.whiten_interval)
        random_direction_values.append(summary.random_direction_mean)

    positions = np.arange(len(dataset_order))

    series = (
        (
            identity_values,
            identity_intervals,
            PAPER_BLUE,
            "None",
            ".2g" if trace_division else ".2f",
        ),
        (
            standardize_values,
            standardize_intervals,
            PAPER_GREEN,
            "Standardization",
            ".2g" if trace_division else ".2f",
        ),
        (
            whiten_values,
            whiten_intervals,
            PAPER_RED,
            "Whitening",
            ".2g" if trace_division else ".2f",
        ),
    )
    width = 0.19
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width)
    upper_limit = 1.12 if y_upper_limit is None else y_upper_limit
    if trace_division and y_upper_limit is None:
        plotted_upper = max(
            interval[1]
            for intervals in (identity_intervals, standardize_intervals, whiten_intervals)
            for interval in intervals
        )
        if include_random_direction:
            plotted_upper = max(plotted_upper, max(random_direction_values))
        upper_limit = max(1.15 * plotted_upper, np.finfo(np.float64).eps)
    for offset, (values, intervals, color, label, number_format) in zip(
        offsets, series
    ):
        errors = np.asarray(
            [
                [mean - interval[0] for mean, interval in zip(values, intervals)],
                [interval[1] - mean for mean, interval in zip(values, intervals)],
            ]
        )
        bars = axis.bar(
            positions + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=label,
        )
        for bar, value, interval in zip(bars, values, intervals):
            rotate_small_label = trace_division and value < 0.05
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                interval[1] + 0.012 * upper_limit,
                format(value, number_format),
                ha="center",
                va="bottom",
                fontsize=annotation_font_size,
                rotation=90 if rotate_small_label else 0,
            )
    if include_random_direction:
        random_direction_bars = axis.bar(
            positions + 1.5 * width,
            random_direction_values,
            width,
            color=PAPER_GREY,
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            label="Random direction",
        )
        # The expected score of a random direction is often orders of
        # magnitude below the fitted-head scores. Label it explicitly so it
        # remains visible on the shared 0--1 scale without exaggerating it.
        for bar, value in zip(random_direction_bars, random_direction_values):
            axis.text(
                (
                    bar.get_x() + bar.get_width() / 2
                    if trace_division
                    else bar.get_x() + bar.get_width() + ADJUST_RANDOM_BAR
                ),
                max(value, 0.0) + 0.012 * upper_limit,
                f"{value:.2g}" if trace_division else f"{value:.3f}",
                ha="center" if trace_division else "left",
                va="bottom",
                fontsize=annotation_font_size,
                rotation=90 if trace_division else 0,
            )
    axis.set_ylim(0.0, upper_limit)
    if not trace_division:
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.set_xticks(
        positions,
        [DATASET_LABELS[dataset] for dataset in dataset_order],
    )
    if show_ylabel:
        denominator = (
            r"\mathrm{tr}(\hat{\Sigma})\boldsymbol{w}^{T}\boldsymbol{w}"
            if trace_division
            else r"\hat{\lambda}_{1}\boldsymbol{w}^{T}\boldsymbol{w}"
        )
        axis.set_ylabel(
            r"$\frac{\boldsymbol{w}^{T}\hat{\Sigma}\boldsymbol{w}}"
            rf"{{{denominator}}}$",
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
    axis.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, HSPACE_LEGEND),
        ncol=2 if include_random_direction else 3,
        fontsize=PLOT_LEGEND_FONT_SIZE,
        columnspacing=0.8,
    )


def make_figure(
    spectra: Sequence[Spectrum],
    summaries: Sequence[HeadSimplicity],
    *,
    trace_division: bool = False,
) -> plt.Figure:
    """Create aligned spectrum and classifier-simplicity panels."""
    configure_plot_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.2),
        gridspec_kw={"width_ratios": (0.84, 1.16)},
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.84,
        bottom=0.32,
        wspace=0.32,
    )
    _draw_spectrum(axes[0], spectra)
    _draw_simplicity(
        figure,
        axes[1],
        summaries,
        include_random_direction=False,
        trace_division=trace_division,
    )
    return figure


def make_comparison_figure(
    summaries: Sequence[HeadSimplicity],
    *,
    method_label: str,
    trace_division: bool = False,
) -> plt.Figure:
    """Create one normalized-simplicity plot for one comparison method."""
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    figure.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.32)
    _draw_simplicity(
        figure,
        axis,
        summaries,
        include_random_direction=False,
        method_label=method_label,
        trace_division=trace_division,
    )
    return figure


def make_frozen_figure(
    summaries: Sequence[HeadSimplicity],
    *,
    trace_division: bool = False,
) -> plt.Figure:
    """Compare frozen DINOv3 and CLIP heads across the two image datasets."""
    expected = {
        (dataset, representation)
        for representation in FROZEN_REPRESENTATION_ORDER
        for dataset in FROZEN_DATASET_ORDER
    }
    by_experiment = {
        (summary.dataset, summary.representation): summary
        for summary in summaries
    }
    if len(by_experiment) != len(summaries) or set(by_experiment) != expected:
        raise ValueError(
            "Frozen plot requires one Waterbirds and CelebA summary for each "
            "configured frozen representation."
        )

    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    figure.subplots_adjust(
        left=0.085, right=0.99, top=0.88, bottom=0.32, wspace=0.14
    )
    shared_upper_limit = None
    if trace_division:
        shared_upper_limit = 1.15 * max(
            interval[1]
            for summary in summaries
            for interval in (
                summary.identity_interval,
                summary.standardize_interval,
                summary.whiten_interval,
            )
        )
    for axis, representation in zip(axes, FROZEN_REPRESENTATION_ORDER):
        representation_summaries = [
            by_experiment[(dataset, representation)]
            for dataset in FROZEN_DATASET_ORDER
        ]
        _draw_simplicity(
            figure,
            axis,
            representation_summaries,
            include_random_direction=False,
            show_ylabel=False,
            trace_division=trace_division,
            y_upper_limit=shared_upper_limit,
            dataset_order=FROZEN_DATASET_ORDER,
        )
        axis.set_title(
            _representation_label(representation), fontsize=PLOT_LABEL_FONT_SIZE
        )
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    denominator = (
        r"\mathrm{tr}(\hat{\Sigma})\boldsymbol{w}^{T}\boldsymbol{w}"
        if trace_division
        else r"\hat{\lambda}_{1}\boldsymbol{w}^{T}\boldsymbol{w}"
    )
    figure.supylabel(
        r"$\frac{\boldsymbol{w}^{T}\hat{\Sigma}\boldsymbol{w}}"
        rf"{{{denominator}}}$",
        x=0.005,
        y=0.605,
        fontsize=PLOT_LABEL_FONT_SIZE,
    )
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.08, -0.22),
        ncol=3,
        fontsize=PLOT_LEGEND_FONT_SIZE,
        columnspacing=0.8,
    )
    return figure


def make_comparison_plot(
    summaries_by_method: Mapping[str, Sequence[HeadSimplicity]],
    *,
    trace_division: bool = False,
) -> plt.Figure:
    """Create aligned normalized-Rayleigh-quotient panels for all comparisons."""
    expected = tuple(COMPARISON_SWEEP_PATHS)
    if tuple(summaries_by_method) != expected:
        raise ValueError(
            "Comparison plot requires DFR, AFR, and NeuroTune summaries in "
            "the configured method order."
        )
    configure_plot_style()
    figure, axes = plt.subplots(1, len(expected), figsize=(18.0, 5.2), sharey=True)
    figure.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.32, wspace=0.14)
    shared_upper_limit = None
    if trace_division:
        shared_upper_limit = 1.15 * max(
            interval[1]
            for summaries in summaries_by_method.values()
            for summary in summaries
            for interval in (
                summary.identity_interval,
                summary.standardize_interval,
                summary.whiten_interval,
            )
        )
    for axis, method in zip(axes, expected):
        _draw_simplicity(
            figure,
            axis,
            summaries_by_method[method],
            include_random_direction=False,
            method_label=COMPARISON_METHOD_LABELS[method],
            show_ylabel=False,
            annotation_font_size=COMPARISON_ANNOTATION_FONT_SIZE,
            trace_division=trace_division,
            y_upper_limit=shared_upper_limit,
        )
        axis.set_title(COMPARISON_METHOD_LABELS[method], fontsize=PLOT_LABEL_FONT_SIZE)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    denominator = (
        r"\mathrm{tr}(\hat{\Sigma})\boldsymbol{w}^{T}\boldsymbol{w}"
        if trace_division
        else r"\hat{\lambda}_{1}\boldsymbol{w}^{T}\boldsymbol{w}"
    )
    figure.supylabel(
        r"$\frac{\boldsymbol{w}^{T}\hat{\Sigma}\boldsymbol{w}}"
        rf"{{{denominator}}}$",
        x=0.002,
        y=0.605,
        fontsize=PLOT_LABEL_FONT_SIZE,
    )
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.62, -0.22),
        ncol=3,
        fontsize=PLOT_LEGEND_FONT_SIZE,
        columnspacing=0.8,
    )
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("main", "appendix", "both"),
        default="main",
        help="Representation family to plot (default: main).",
    )
    parser.add_argument(
        "--comparison-method",
        action="append",
        choices=tuple(COMPARISON_SWEEP_PATHS),
        help="Comparison method to render; repeat for several (default: all).",
    )
    parser.add_argument(
        "--plot-type",
        choices=("spectrum", "simplicity", "frozen", "comparison", "comparison_plot"),
        default="spectrum",
        help=(
            "Render the spectrum/ERM figure, a frozen-backbone simplicity plot, "
            "print ERM simplicity values, or three method-specific "
            "comparison simplicity figures; comparison_plot combines DFR, AFR, "
            "and NeuroTune into one three-panel figure (default: spectrum)."
        ),
    )
    parser.add_argument(
        "--sweep",
        action="append",
        help=(
            "Sweep JSON used by --plot-type simplicity or frozen; repeat for "
            "multiple representations or datasets. Frozen split sweeps are "
            "reconstructed."
        ),
    )
    parser.add_argument(
        "--selection-criterion",
        choices=("accuracy", "class_balanced_accuracy", "log_loss", "balanced_log_loss"),
        default=None,
        help=(
            "Validation rule used to select simplicity heads. Defaults to "
            "balanced_log_loss for --plot-type frozen and "
            "class_balanced_accuracy otherwise."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/embeddings",
        help="Root containing NumPy embedding bundles.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=None,
        help="Empirical rank tolerance; default is float32 machine epsilon.",
    )
    parser.add_argument(
        "--trace-division",
        action="store_true",
        help=(
            "Normalize Rayleigh quotients by the covariance trace instead of "
            "its largest eigenvalue."
        ),
    )
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify complete array checksums before fitting (slower).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path stem (default depends on --plot-type).",
    )
    parser.add_argument(
        "--coefficient-cache",
        default=str(DEFAULT_COEFFICIENT_CACHE),
        help=(
            "Temporary directory for selected fitted coefficients "
            f"(default: {DEFAULT_COEFFICIENT_CACHE})."
        ),
    )
    parser.add_argument(
        "--spectrum-cache",
        default=str(DEFAULT_SPECTRUM_CACHE),
        help=(
            "Directory for fingerprint-keyed empirical spectrum caches "
            f"(default: {DEFAULT_SPECTRUM_CACHE})."
        ),
    )
    parser.add_argument(
        "--refit-coefficients",
        action="store_true",
        help="Ignore compatible cached coefficients and refit selected heads.",
    )
    parser.add_argument(
        "--refit-spectra",
        action="store_true",
        help="Ignore compatible spectrum caches and refit every covariance.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "svg"),
        help="Output format; repeat for multiple formats (default: png).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.relative_tolerance is not None and args.relative_tolerance < 0:
        parser.error("--relative-tolerance must be nonnegative.")
    return args


def _specs_from_args(args: argparse.Namespace) -> tuple[ExperimentSpec, ...]:
    """Return the fixed spectrum inputs paired with the sweep preset."""
    return _preset_specs(args.preset)


def _selection_criterion(args: argparse.Namespace) -> str:
    """Resolve the plot-specific default without changing existing figures."""
    if args.selection_criterion is not None:
        return str(args.selection_criterion)
    return (
        "balanced_log_loss"
        if args.plot_type == "frozen"
        else "class_balanced_accuracy"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selection_criterion = _selection_criterion(args)
    if args.plot_type == "simplicity":
        if not args.sweep:
            raise ValueError("--plot-type simplicity requires at least one --sweep.")
        summaries = load_head_simplicities(
            args.sweep,
            criterion=selection_criterion,
            relative_tolerance=args.relative_tolerance,
            coefficient_cache=args.coefficient_cache,
            refit_coefficients=args.refit_coefficients,
            trace_division=args.trace_division,
        )
        for summary in summaries:
            print(json.dumps({
                "dataset": summary.dataset,
                "representation": summary.representation,
                "replicates": list(summary.seeds),
                "selection_criterion": selection_criterion,
                "normalization_denominator": (
                    "covariance_trace"
                    if args.trace_division
                    else "maximum_eigenvalue"
                ),
                "normalized_simplicity": {
                    "identity": list(summary.identity_scores),
                    "standardize": list(summary.standardize_scores),
                    "whiten_ledoit_wolf_nonlinear": list(summary.whiten_scores),
                    "random_direction": list(summary.random_direction_scores),
                },
                "means": {
                    "identity": summary.identity_mean,
                    "standardize": summary.standardize_mean,
                    "whiten_ledoit_wolf_nonlinear": summary.whiten_mean,
                    "random_direction": summary.random_direction_mean,
                },
            }, sort_keys=True))
        return 0
    if args.plot_type == "frozen":
        summaries = load_head_simplicities(
            args.sweep or FROZEN_SWEEP_PATHS,
            criterion=selection_criterion,
            relative_tolerance=args.relative_tolerance,
            coefficient_cache=args.coefficient_cache,
            refit_coefficients=args.refit_coefficients,
            trace_division=args.trace_division,
        )
        figure = make_frozen_figure(
            summaries, trace_division=args.trace_division
        )
        suffix = "_trace" if args.trace_division else ""
        output = args.output or f"plots/simplicity_frozen{suffix}"
        output_path = Path(output).expanduser()
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        try:
            paths = save_figure(
                figure,
                output_path,
                formats=args.formats or ("png",),
                overwrite=args.overwrite,
            )
        finally:
            plt.close(figure)
        for path in paths:
            print(path)
        return 0
    if args.plot_type in {"comparison", "comparison_plot"}:
        summaries_by_method = load_comparison_simplicities(
            coefficient_cache=args.coefficient_cache,
            refit_coefficients=args.refit_coefficients,
            methods=args.comparison_method or tuple(COMPARISON_SWEEP_PATHS),
            trace_division=args.trace_division,
        )
        output = args.output or "plots/simplicity_comparison"
        output_path = Path(output).expanduser()
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        if args.plot_type == "comparison_plot":
            if args.comparison_method is not None:
                raise ValueError(
                    "comparison_plot always renders DFR, AFR, and NeuroTune; "
                    "do not pass --comparison-method."
                )
            figure = make_comparison_plot(
                summaries_by_method, trace_division=args.trace_division
            )
            try:
                paths = save_figure(
                    figure,
                    output_path,
                    formats=args.formats or ("png",),
                    overwrite=args.overwrite,
                )
            finally:
                plt.close(figure)
            for path in paths:
                print(path)
            return 0
        paths = []
        for method, summaries in summaries_by_method.items():
            figure = make_comparison_figure(
                summaries,
                method_label=COMPARISON_METHOD_LABELS[method],
                trace_division=args.trace_division,
            )
            try:
                paths.extend(save_figure(
                    figure,
                    output_path.with_name(f"{output_path.name}_{method}"),
                    formats=args.formats or ("png",),
                    overwrite=args.overwrite,
                ))
            finally:
                plt.close(figure)
        for path in paths:
            print(path)
        return 0
    if args.preset == "both":
        raise ValueError("Combined plots require preset main or appendix.")
    spectra = load_spectra(
        _preset_specs(args.preset),
        artifact_root=args.artifact_root,
        relative_tolerance=args.relative_tolerance,
        verify_hashes=args.verify_hashes,
        spectrum_cache=args.spectrum_cache,
        refit_spectra=args.refit_spectra,
    )
    summaries = load_head_simplicities(
        SWEEP_PRESETS[args.preset],
        criterion=selection_criterion,
        relative_tolerance=args.relative_tolerance,
        coefficient_cache=args.coefficient_cache,
        refit_coefficients=args.refit_coefficients,
        trace_division=args.trace_division,
    )
    figure = make_figure(
        spectra, summaries, trace_division=args.trace_division
    )
    suffix = "_trace" if args.trace_division else ""
    output = args.output or f"plots/spectrum_{args.preset}{suffix}"
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    try:
        paths = save_figure(
            figure,
            output_path,
            formats=args.formats or ("png",),
            overwrite=args.overwrite,
        )
    finally:
        plt.close(figure)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
