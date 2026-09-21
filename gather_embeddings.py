#!/usr/bin/env python3
"""Offline extraction of frozen image or MultiNLI embeddings into bundles.

This script is intentionally separate from the NumPy-only evaluation path.
It reads the archival Waterbirds or CelebA image sources, or the cached
MultiNLI BERT inputs, runs a frozen backbone, and writes the ``.npy`` plus
``manifest.json`` bundle consumed by ``evaluate.py`` and
``hyper_param_sweep.py``.

Examples
--------
Validate the CelebA source without loading a model::

    python gather_embeddings.py --dataset celeba --dry-run

Extract DINOv3 ViT-L/16 embeddings for every CelebA split::

    python gather_embeddings.py \
        --dataset celeba \
        --backbone dinov3-vitl16 \
        --device auto \
        --batch-size 8

Reproduce CLS-token embeddings from a fine-tuned MultiNLI checkpoint::

    python gather_embeddings.py \
        --dataset multinli \
        --model models/multinli_run_seed4/bert_multinli_seed4.pt \
        --seed 4 \
        --device auto \
        --batch-size 32

The completed output is written to
``artifacts/embeddings/CelebA/dinov3/default``. Interrupted runs retain only
hidden, per-split staging files and can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import re
import shutil
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from data import (
    SPLITS,
    EmbeddingBundle,
    SplitArrays,
    sha256_file,
    write_embedding_bundle,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "embeddings"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MULTINLI_FEATURE_FILES = (
    "cached_train_bert-base-uncased_128_mnli",
    "cached_dev_bert-base-uncased_128_mnli",
    "cached_dev_bert-base-uncased_128_mnli-mm",
)
MULTINLI_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


# =============================================================================
# Extensible backbone registry
# =============================================================================

@dataclass(frozen=True)
class BackboneSpec:
    """One frozen image backbone and the adapter used to extract its features."""

    key: str
    model_name: str
    representation: str
    adapter: str
    feature_type: str
    per_observation_l2_normalization: bool
    minimum_transformers_version: tuple[int, int, int]


BACKBONES: dict[str, BackboneSpec] = {
    "dinov3-vitb16": BackboneSpec(
        key="dinov3-vitb16",
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        representation="dinov3_vitb16",
        adapter="huggingface_cls_token",
        feature_type="cls_token",
        per_observation_l2_normalization=False,
        minimum_transformers_version=(4, 56, 0),
    ),
    "clip-vitb16": BackboneSpec(
        key="clip-vitb16",
        model_name="openai/clip-vit-base-patch16",
        representation="clip_openai_vitb16",
        adapter="huggingface_clip_projection_l2",
        feature_type="normalized_image_projection",
        per_observation_l2_normalization=True,
        minimum_transformers_version=(4, 56, 0),
    )
}

BACKBONE_ALIASES = {
    "dino": "dinov3-vitb6",
    "dinov3": "dinov3-vitb16",
    "clip": "clip-vitlb16",
    "openai-clip": "clip-vitb16",
}


def resolve_backbone(
    name: str,
    *,
    model_override: str | None = None,
    representation_override: str | None = None,
) -> BackboneSpec:
    """Resolve a registry key while permitting an explicit checkpoint override."""
    key = BACKBONE_ALIASES.get(name.strip().lower(), name.strip().lower())
    try:
        registered = BACKBONES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKBONES))
        raise ValueError(
            f"Unknown image backbone {name!r}; choose from {choices}."
        ) from exc
    return BackboneSpec(
        key=registered.key,
        model_name=model_override or registered.model_name,
        representation=representation_override or registered.representation,
        adapter=registered.adapter,
        feature_type=registered.feature_type,
        per_observation_l2_normalization=(
            registered.per_observation_l2_normalization
        ),
        minimum_transformers_version=registered.minimum_transformers_version,
    )


# =============================================================================
# Dataset adapters
# =============================================================================

class ImageSplitSource(Protocol):
    """Minimal split interface shared by all extraction sources."""

    y: np.ndarray
    g: np.ndarray

    def __len__(self) -> int: ...

    def load_batch(self, start: int, end: int) -> Any: ...

    def release(self) -> None: ...


@dataclass
class PathImageSplit:
    paths: tuple[Path, ...]
    y: np.ndarray
    g: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)

    def load_batch(self, start: int, end: int) -> list[Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Image extraction requires Pillow. Install the embedding "
                "dependencies with `python -m pip install -e '.[embeddings]'`."
            ) from exc

        images = []
        for path in self.paths[start:end]:
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Could not read image: {path}") from exc
        return images

    def release(self) -> None:
        # Paths and labels are small compared with the image files themselves.
        return None


@dataclass
class TensorImageSplit:
    images: Any
    y: np.ndarray
    g: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def load_batch(self, start: int, end: int) -> Any:
        if self.images is None:
            raise RuntimeError("The Waterbirds image tensor has been released.")
        batch = self.images[start:end]
        mean = batch.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = batch.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (batch * std + mean).clamp_(0.0, 1.0)

    def release(self) -> None:
        self.images = None


@dataclass
class ImageDatasetSource:
    dataset: str
    splits: dict[str, ImageSplitSource]
    processor_kwargs: dict[str, Any]
    source: dict[str, Any]
    source_fingerprint: str
    group_definition: str


@dataclass
class TokenSplit:
    """A metadata-defined view into the concatenated cached MultiNLI inputs."""

    features: Sequence[Any]
    indices: np.ndarray
    y: np.ndarray
    g: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)

    def load_batch(self, start: int, end: int) -> dict[str, Any]:
        torch = _require_torch()
        selected = self.indices[start:end]
        return {
            "input_ids": torch.tensor(
                [self.features[int(index)].input_ids for index in selected],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [self.features[int(index)].input_mask for index in selected],
                dtype=torch.long,
            ),
            "token_type_ids": torch.tensor(
                [self.features[int(index)].segment_ids for index in selected],
                dtype=torch.long,
            ),
        }

    def release(self) -> None:
        # The three split views share one feature list. It is released when the
        # complete source falls out of scope, not after an individual split.
        return None


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _binary_value(raw: Any, *, column: str, row_number: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CelebA row {row_number}: {column!r} must be an integer."
        ) from exc
    if value in (0, 1):
        return value
    if value in (-1, 1):
        return int(value == 1)
    raise ValueError(
        f"CelebA row {row_number}: {column!r} must use 0/1 or -1/1 values."
    )


def _split_name(raw: Any, *, row_number: int) -> str:
    normalized = str(raw).strip().lower()
    aliases = {
        "0": "train",
        "1": "val",
        "2": "test",
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"CelebA row {row_number}: unrecognized split value {raw!r}."
        ) from exc


def load_celeba_source(
    root: str | Path,
    *,
    metadata_path: str | Path | None = None,
    images_dir: str | Path | None = None,
    target_column: str = "y",
    spurious_column: str = "Female",
    verify_images: bool = True,
) -> ImageDatasetSource:
    """Load CelebA paths and labels from the existing consolidated metadata."""
    root = Path(root).expanduser().resolve()
    metadata = (
        root / "metadata_celeba_Blond_Hair_Female.csv"
        if metadata_path is None
        else Path(metadata_path).expanduser().resolve()
    )
    image_root = (
        root / "images"
        if images_dir is None
        else Path(images_dir).expanduser().resolve()
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"CelebA metadata not found: {metadata}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"CelebA image directory not found: {image_root}")

    records: dict[str, dict[str, list[Any]]] = {
        split: {"paths": [], "y": [], "g": []} for split in SPLITS
    }
    seen_filenames: set[str] = set()
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        image_column = "img_filename" if "img_filename" in columns else "image_id"
        required = {image_column, target_column, spurious_column, "split"}
        missing = sorted(required.difference(columns))
        if missing:
            raise KeyError(
                f"CelebA metadata is missing columns: {', '.join(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            filename = str(row[image_column]).strip()
            if not filename:
                raise ValueError(f"CelebA row {row_number}: empty image filename.")
            if filename in seen_filenames:
                raise ValueError(f"CelebA metadata repeats image {filename!r}.")
            seen_filenames.add(filename)
            split = _split_name(row["split"], row_number=row_number)
            y = _binary_value(
                row[target_column], column=target_column, row_number=row_number
            )
            attribute = _binary_value(
                row[spurious_column],
                column=spurious_column,
                row_number=row_number,
            )
            records[split]["paths"].append(image_root / filename)
            records[split]["y"].append(y)
            records[split]["g"].append(2 * y + attribute)

    if verify_images:
        missing_paths = []
        for split in SPLITS:
            for path in records[split]["paths"]:
                if not path.is_file():
                    missing_paths.append(path)
                    if len(missing_paths) == 5:
                        break
            if missing_paths:
                break
        if missing_paths:
            names = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"CelebA images are missing, including: {names}")

    splits: dict[str, ImageSplitSource] = {}
    for split in SPLITS:
        values = records[split]
        if not values["paths"]:
            raise ValueError(f"CelebA metadata contains no {split} observations.")
        splits[split] = PathImageSplit(
            paths=tuple(values["paths"]),
            y=np.asarray(values["y"], dtype=np.int64),
            g=np.asarray(values["g"], dtype=np.int64),
        )

    source = {
        "format": "celeba_image_directory",
        "metadata_csv": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "images_dir": str(image_root),
        "image_count": sum(len(split) for split in splits.values()),
        "target_column": target_column,
        "spurious_column": spurious_column,
    }
    return ImageDatasetSource(
        dataset="CelebA",
        splits=splits,
        processor_kwargs={},
        source=source,
        source_fingerprint=_json_fingerprint(source),
        group_definition=f"g = 2*y + 1[{spurious_column} = 1]",
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Embedding extraction requires PyTorch. Install the optional "
            "dependencies with `python -m pip install -e '.[embeddings]'`."
        ) from exc
    return torch


def _tensor_to_int64(values: Any, *, name: str) -> np.ndarray:
    if not hasattr(values, "detach"):
        raise TypeError(f"Waterbirds {name} must be a torch tensor.")
    return values.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)


def load_waterbirds_source(path: str | Path) -> ImageDatasetSource:
    """Load the existing trusted Waterbirds pickle without modifying it."""
    _require_torch()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Waterbirds pickle not found: {path}")
    print(
        f"Loading Waterbirds source from {path} "
        f"({path.stat().st_size / 2**30:.1f} GiB)",
        flush=True,
    )
    with path.open("rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, dict):
        raise TypeError("The Waterbirds pickle must contain a dictionary.")

    required = {
        f"{prefix}_{split}"
        for split in SPLITS
        for prefix in ("X", "y", "c")
    }
    missing = sorted(required.difference(values))
    if missing:
        raise KeyError(f"Waterbirds source is missing: {', '.join(missing)}")

    splits: dict[str, ImageSplitSource] = {}
    for split in SPLITS:
        images = values.pop(f"X_{split}")
        y = _tensor_to_int64(values.pop(f"y_{split}"), name=f"y_{split}")
        c = _tensor_to_int64(values.pop(f"c_{split}"), name=f"c_{split}")
        if getattr(images, "ndim", None) != 4 or tuple(images.shape[1:]) != (
            3,
            224,
            224,
        ):
            raise ValueError(
                f"Waterbirds X_{split} must have shape (n, 3, 224, 224)."
            )
        if len(images) != len(y) or len(y) != len(c):
            raise ValueError(f"Waterbirds {split} arrays have unequal lengths.")
        if not np.all(np.isin(y, (0, 1))) or not np.all(np.isin(c, (0, 1))):
            raise ValueError("Waterbirds y and c values must be binary.")
        splits[split] = TensorImageSplit(
            images=images,
            y=y,
            g=(2 * y + c).astype(np.int64, copy=False),
        )
    del values

    file_stat = path.stat()
    source = {
        "format": "trusted_waterbirds_pickle",
        "path": str(path),
        "size_bytes": int(file_stat.st_size),
        "modified_time_ns": int(file_stat.st_mtime_ns),
    }
    return ImageDatasetSource(
        dataset="WB",
        splits=splits,
        processor_kwargs={
            "do_resize": False,
            "do_center_crop": False,
            "do_rescale": False,
        },
        source=source,
        source_fingerprint=_json_fingerprint(source),
        group_definition="g = 2*y + c",
    )


def load_multinli_source(
    root: str | Path,
    *,
    metadata_path: str | Path | None = None,
    features_dir: str | Path | None = None,
) -> ImageDatasetSource:
    """Load the cached BERT inputs in their original concatenated order.

    The archival feature files are trusted PyTorch pickles containing
    ``utils_glue.InputFeatures`` objects. They are used only by this offline
    extraction utility; the evaluation path remains NumPy-only.
    """
    torch = _require_torch()
    root = Path(root).expanduser().resolve()
    metadata = (
        root / "metadata_random.csv"
        if metadata_path is None
        else Path(metadata_path).expanduser().resolve()
    )
    feature_root = (
        root / "multiNLI_bert_features"
        if features_dir is None
        else Path(features_dir).expanduser().resolve()
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"MultiNLI metadata not found: {metadata}")
    if not feature_root.is_dir():
        raise FileNotFoundError(
            f"MultiNLI cached-feature directory not found: {feature_root}"
        )

    # Import the class named in the trusted cached pickles before unpickling.
    import utils_glue  # noqa: F401

    features: list[Any] = []
    feature_records = []
    for filename in MULTINLI_FEATURE_FILES:
        path = feature_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"MultiNLI cached features not found: {path}")
        print(f"Loading cached MultiNLI inputs: {path}", flush=True)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, list) or not loaded:
            raise TypeError(f"Expected a non-empty feature list in {path}.")
        features.extend(loaded)
        feature_records.append(
            {
                "path": str(path),
                "observations": len(loaded),
                "sha256": sha256_file(path),
            }
        )

    rows: list[tuple[int, int, int]] = []
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"gold_label", "sentence2_has_negation", "split"}
        missing = sorted(required.difference(columns))
        if missing:
            raise KeyError(
                f"MultiNLI metadata is missing columns: {', '.join(missing)}"
            )
        for row_index, row in enumerate(reader):
            if "Unnamed: 0" in columns:
                try:
                    stored_index = int(row["Unnamed: 0"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"MultiNLI metadata row {row_index + 2} has an invalid index."
                    ) from exc
                if stored_index != row_index:
                    raise ValueError(
                        "MultiNLI metadata indices must preserve cached-feature "
                        f"order; row {row_index + 2} contains {stored_index}."
                    )
            try:
                y = int(row["gold_label"])
                c = int(row["sentence2_has_negation"])
                split_id = int(row["split"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MultiNLI metadata row {row_index + 2} must be integer-valued."
                ) from exc
            if y not in (0, 1, 2):
                raise ValueError(
                    f"MultiNLI row {row_index + 2}: gold_label must be 0, 1, or 2."
                )
            if c not in (0, 1):
                raise ValueError(
                    "MultiNLI row "
                    f"{row_index + 2}: sentence2_has_negation must be binary."
                )
            if split_id not in MULTINLI_SPLIT_IDS.values():
                raise ValueError(
                    f"MultiNLI row {row_index + 2}: split must be 0, 1, or 2."
                )
            rows.append((y, c, split_id))

    if len(rows) != len(features):
        raise ValueError(
            "MultiNLI metadata and cached features have unequal lengths: "
            f"{len(rows):,} != {len(features):,}."
        )
    metadata_values = np.asarray(rows, dtype=np.int64)
    labels = metadata_values[:, 0]
    attributes = metadata_values[:, 1]
    split_ids = metadata_values[:, 2]

    sequence_length: int | None = None
    for index, feature in enumerate(features):
        required_attributes = (
            "input_ids",
            "input_mask",
            "segment_ids",
            "label_id",
        )
        if not all(hasattr(feature, name) for name in required_attributes):
            raise TypeError(
                f"MultiNLI cached feature {index} lacks required BERT fields."
            )
        lengths = {
            len(feature.input_ids),
            len(feature.input_mask),
            len(feature.segment_ids),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"MultiNLI cached feature {index} has unequal token-field lengths."
            )
        current_length = lengths.pop()
        if sequence_length is None:
            sequence_length = current_length
        elif current_length != sequence_length:
            raise ValueError("MultiNLI cached features have unequal sequence lengths.")
        if int(feature.label_id) != int(labels[index]):
            raise ValueError(
                "MultiNLI cached-feature labels do not match metadata at "
                f"observation {index}."
            )

    splits: dict[str, ImageSplitSource] = {}
    for split, split_id in MULTINLI_SPLIT_IDS.items():
        indices = np.flatnonzero(split_ids == split_id).astype(np.int64)
        if not len(indices):
            raise ValueError(f"MultiNLI metadata contains no {split} observations.")
        y = labels[indices].astype(np.int64, copy=True)
        c = attributes[indices]
        splits[split] = TokenSplit(
            features=features,
            indices=indices,
            y=y,
            g=(2 * y + c).astype(np.int64, copy=False),
        )

    source = {
        "format": "trusted_cached_multinli_bert_inputs",
        "metadata_csv": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "feature_files": feature_records,
        "concatenation_order": list(MULTINLI_FEATURE_FILES),
        "sequence_length": int(sequence_length or 0),
        "observations": len(features),
    }
    return ImageDatasetSource(
        dataset="multiNLI",
        splits=splits,
        processor_kwargs={},
        source=source,
        source_fingerprint=_json_fingerprint(source),
        group_definition="g = 2*y + sentence2_has_negation",
    )


def load_dataset_source(
    dataset: str,
    *,
    data_root: str | Path,
    metadata_path: str | Path | None,
    images_dir: str | Path | None,
    waterbirds_input: str | Path | None,
    multinli_features_dir: str | Path | None,
    target_column: str,
    spurious_column: str,
    verify_images: bool,
) -> ImageDatasetSource:
    normalized = dataset.strip().lower()
    root = Path(data_root).expanduser().resolve()
    if normalized in {"celeba", "celeb_a"}:
        return load_celeba_source(
            root / "CelebA",
            metadata_path=metadata_path,
            images_dir=images_dir,
            target_column=target_column,
            spurious_column=spurious_column,
            verify_images=verify_images,
        )
    if normalized in {"wb", "waterbirds", "waterbird"}:
        source_path = (
            root / "WB" / "data_WB_95.pkl"
            if waterbirds_input is None
            else Path(waterbirds_input).expanduser().resolve()
        )
        return load_waterbirds_source(source_path)
    if normalized in {"multinli", "multi_nli", "mnli"}:
        return load_multinli_source(
            root / "multiNLI",
            metadata_path=metadata_path,
            features_dir=multinli_features_dir,
        )
    raise ValueError("dataset must be 'celeba', 'waterbirds', or 'multinli'.")


# =============================================================================
# Frozen image encoder
# =============================================================================

def _version_tuple(value: str) -> tuple[int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", value)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])  # type: ignore[return-value]


def _require_transformers(minimum: tuple[int, int, int]) -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Image extraction requires Hugging Face Transformers. Install "
            "the optional dependencies with "
            "`python -m pip install -e '.[embeddings]'`."
        ) from exc
    if _version_tuple(installed) < minimum:
        required = ".".join(map(str, minimum))
        raise RuntimeError(
            f"transformers>={required} is required, but {installed} is installed. "
            "Upgrade with `python -m pip install -e '.[embeddings]'`."
        )


def resolve_device(requested: str) -> Any:
    torch = _require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def resolve_dtype(requested: str, device: Any) -> Any:
    torch = _require_torch()
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, requested)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 inference is not supported reliably on CPU.")
    if device.type == "mps" and dtype == torch.float16:
        raise ValueError(
            "float16 inference on MPS can produce non-finite features; "
            "use --dtype float32."
        )
    return dtype


class FrozenImageEncoder(Protocol):
    dimension: int
    device: Any
    dtype: Any

    def encode(
        self, images: Any, *, processor_kwargs: Mapping[str, Any]
    ) -> np.ndarray: ...

    def release_cache(self) -> None: ...


class HuggingFaceClsTokenEncoder:
    """Final CLS token from a Hugging Face vision-transformer checkpoint."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        device: Any,
        dtype: Any,
        local_files_only: bool = False,
    ):
        _require_transformers(spec.minimum_transformers_version)
        from transformers import AutoImageProcessor, AutoModel

        torch = _require_torch()
        print(f"Loading image processor: {spec.model_name}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(
            spec.model_name,
            local_files_only=local_files_only,
        )
        print(
            f"Loading frozen model on {device} with dtype={dtype}: "
            f"{spec.model_name}",
            flush=True,
        )
        self.model = AutoModel.from_pretrained(
            spec.model_name,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.dimension = int(getattr(self.model.config, "hidden_size"))

    def encode(
        self, images: Any, *, processor_kwargs: Mapping[str, Any]
    ) -> np.ndarray:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            **dict(processor_kwargs),
        )
        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=self.device.type == "cuda",
        )
        if not self.torch.isfinite(pixel_values).all():
            raise RuntimeError("The image processor produced non-finite pixels.")
        with self.torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3:
            raise RuntimeError("The selected backbone did not return image tokens.")
        features = hidden[:, 0, :]
        if features.shape[1] != self.dimension:
            raise RuntimeError(
                f"Expected {self.dimension} features, got {features.shape[1]}."
            )
        if not self.torch.isfinite(features).all():
            advice = " Re-run with --dtype float32." if self.dtype != self.torch.float32 else ""
            raise RuntimeError(f"The backbone produced non-finite features.{advice}")
        result = features.float().cpu().numpy()
        del inputs, pixel_values, outputs, hidden, features
        return result

    def release_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        elif self.device.type == "mps":
            self.torch.mps.empty_cache()


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    """Return float32 unit-norm rows, rejecting invalid model outputs."""
    result = np.asarray(features, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("Image features must be a two-dimensional array.")
    if not np.all(np.isfinite(result)):
        raise RuntimeError("The backbone produced non-finite features.")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise RuntimeError("The CLIP projection produced a zero-norm feature.")
    normalized = result / norms
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("CLIP feature normalization produced non-finite values.")
    return normalized.astype(np.float32, copy=False)


class HuggingFaceClipProjectionEncoder:
    """L2-normalized projected image embedding from a Hugging Face CLIP model."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        device: Any,
        dtype: Any,
        local_files_only: bool = False,
    ):
        _require_transformers(spec.minimum_transformers_version)
        from transformers import (
            AutoImageProcessor,
            CLIPVisionModelWithProjection,
        )

        torch = _require_torch()
        print(f"Loading image processor: {spec.model_name}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(
            spec.model_name,
            local_files_only=local_files_only,
        )
        print(
            f"Loading frozen model on {device} with dtype={dtype}: "
            f"{spec.model_name}",
            flush=True,
        )
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            spec.model_name,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.dimension = int(getattr(self.model.config, "projection_dim"))

    def encode(
        self, images: Any, *, processor_kwargs: Mapping[str, Any]
    ) -> np.ndarray:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            **dict(processor_kwargs),
        )
        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=self.device.type == "cuda",
        )
        if not self.torch.isfinite(pixel_values).all():
            raise RuntimeError("The image processor produced non-finite pixels.")
        with self.torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
        features = outputs.image_embeds
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise RuntimeError(
                "The CLIP model returned an unexpected projected feature shape: "
                f"{tuple(features.shape)}."
            )
        result = _l2_normalize_rows(features.float().cpu().numpy())
        del inputs, pixel_values, outputs, features
        return result

    def release_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        elif self.device.type == "mps":
            self.torch.mps.empty_cache()


def _unwrap_checkpoint_state_dict(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return tensor parameters and small JSON-compatible checkpoint metadata."""
    torch = _require_torch()
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The BERT checkpoint must contain a state dictionary.")

    state: Mapping[str, Any] | None = None
    state_key: str | None = None
    for candidate in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(candidate)
        if isinstance(value, Mapping) and value:
            state = value
            state_key = candidate
            break
    if state is None and checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        state = checkpoint
    if state is None:
        raise TypeError(
            "The BERT checkpoint must be a raw state_dict or contain "
            "'model_state_dict', 'state_dict', or 'model'."
        )
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise TypeError("Every BERT state-dictionary value must be a tensor.")

    normalized = dict(state)
    for prefix in ("module.", "_orig_mod.", "model."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {
                key.removeprefix(prefix): value for key, value in normalized.items()
            }

    metadata = {}
    if state_key is not None:
        for key, value in checkpoint.items():
            if key == state_key:
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            metadata[str(key)] = value
    return normalized, metadata


class HuggingFaceBertClsTokenEncoder:
    """Final hidden-state CLS token from a fine-tuned BERT checkpoint."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        base_model: str,
        device: Any,
        dtype: Any,
        local_files_only: bool = False,
    ):
        _require_transformers(spec.minimum_transformers_version)
        from transformers import BertModel

        torch = _require_torch()
        checkpoint_path = Path(spec.model_name).expanduser()
        self.checkpoint_metadata: dict[str, Any] = {}
        self.checkpoint_sha256: str | None = None
        if checkpoint_path.is_file():
            checkpoint_path = checkpoint_path.resolve()
            if checkpoint_path.suffix.lower() not in {".pt", ".pth", ".bin"}:
                raise ValueError(
                    "A file-based BERT checkpoint must use .pt, .pth, or .bin."
                )
            print(f"Loading BERT base model: {base_model}", flush=True)
            model = BertModel.from_pretrained(
                base_model,
                local_files_only=local_files_only,
            )
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not safely load BERT state dictionary: {checkpoint_path}"
                ) from exc
            state, self.checkpoint_metadata = _unwrap_checkpoint_state_dict(
                checkpoint
            )
            del checkpoint
            if any(key.startswith("bert.") for key in state):
                state = {
                    key.removeprefix("bert."): value
                    for key, value in state.items()
                    if key.startswith("bert.")
                }
            else:
                state = {
                    key: value
                    for key, value in state.items()
                    if not key.startswith("classifier.")
                }
            incompatible = model.load_state_dict(state, strict=False)
            missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("pooler.")
            ]
            unexpected = list(incompatible.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "The BERT checkpoint does not match the base model. "
                    f"Missing keys: {missing[:5]}; unexpected keys: {unexpected[:5]}."
                )
            self.checkpoint_sha256 = sha256_file(checkpoint_path)
        else:
            if checkpoint_path.suffix.lower() in {".pt", ".pth", ".bin"}:
                raise FileNotFoundError(f"BERT checkpoint not found: {checkpoint_path}")
            print(f"Loading BERT checkpoint: {spec.model_name}", flush=True)
            model = BertModel.from_pretrained(
                spec.model_name,
                local_files_only=local_files_only,
            )

        if str(getattr(model.config, "model_type", "")).lower() != "bert":
            raise ValueError(
                f"{spec.model_name!r} is not a BERT checkpoint "
                f"(model_type={getattr(model.config, 'model_type', None)!r})."
            )
        model.eval()
        model.requires_grad_(False)
        model.to(device=device, dtype=dtype)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.dimension = int(model.config.hidden_size)

    def encode(
        self, inputs: Any, *, processor_kwargs: Mapping[str, Any]
    ) -> np.ndarray:
        del processor_kwargs
        if not isinstance(inputs, Mapping):
            raise TypeError("BERT inputs must be a mapping of token tensors.")
        batch = {
            name: values.to(
                device=self.device,
                non_blocking=self.device.type == "cuda",
            )
            for name, values in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**batch, return_dict=True)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[2] != self.dimension:
            raise RuntimeError(
                "BERT returned an unexpected final hidden-state shape: "
                f"{tuple(hidden.shape)}."
            )
        features = hidden[:, 0, :]
        if not self.torch.isfinite(features).all():
            raise RuntimeError("BERT produced non-finite CLS-token features.")
        result = features.float().cpu().numpy()
        del batch, outputs, hidden, features
        return result

    def release_cache(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        elif self.device.type == "mps":
            self.torch.mps.empty_cache()


def load_image_encoder(
    spec: BackboneSpec,
    *,
    device: str,
    dtype: str,
    local_files_only: bool = False,
) -> FrozenImageEncoder:
    """Construct the adapter named by a registry entry."""
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    factories = {
        "huggingface_cls_token": HuggingFaceClsTokenEncoder,
        "huggingface_clip_projection_l2": HuggingFaceClipProjectionEncoder,
    }
    try:
        factory = factories[spec.adapter]
    except KeyError as exc:
        raise ValueError(f"Unsupported image adapter: {spec.adapter!r}") from exc
    return factory(
        spec,
        device=resolved_device,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
    )


def resolve_multinli_spec(
    model: str | Path,
    *,
    representation: str | None = None,
) -> BackboneSpec:
    """Describe the historical MultiNLI final-layer CLS representation."""
    model_name = str(model)
    path = Path(model_name).expanduser()
    if path.exists():
        model_name = str(path.resolve())
    return BackboneSpec(
        key="bert",
        model_name=model_name,
        representation=representation or "bert",
        adapter="huggingface_bert_final_cls_token",
        feature_type="last_hidden_state_cls_token",
        per_observation_l2_normalization=False,
        minimum_transformers_version=(4, 56, 0),
    )


def load_multinli_encoder(
    spec: BackboneSpec,
    *,
    base_model: str,
    device: str,
    dtype: str,
    local_files_only: bool = False,
) -> HuggingFaceBertClsTokenEncoder:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    return HuggingFaceBertClsTokenEncoder(
        spec,
        base_model=base_model,
        device=resolved_device,
        dtype=resolved_dtype,
        local_files_only=local_files_only,
    )


# =============================================================================
# Resumable extraction and bundle finalization
# =============================================================================

def default_destination(
    artifact_root: str | Path,
    *,
    dataset: str,
    representation: str,
) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / dataset
        / representation
        / "default"
    )


def multinli_destination(
    artifact_root: str | Path,
    *,
    representation: str,
    seed: int,
) -> Path:
    if seed < 0:
        raise ValueError("MultiNLI representation seed must be nonnegative.")
    return (
        Path(artifact_root).expanduser().resolve()
        / "multiNLI"
        / representation
        / f"seed_{seed:03d}"
    )


def _stage_signature(
    source: ImageDatasetSource,
    spec: BackboneSpec,
    encoder: FrozenImageEncoder,
) -> dict[str, Any]:
    signature = {
        "schema_version": 1,
        "dataset": source.dataset,
        "source_fingerprint": source.source_fingerprint,
        "backbone": spec.key,
        "model_name": spec.model_name,
        "representation": spec.representation,
        "feature_type": spec.feature_type,
        "per_observation_l2_normalization": (
            spec.per_observation_l2_normalization
        ),
        "embedding_dimension": int(encoder.dimension),
    }
    checkpoint_sha256 = getattr(encoder, "checkpoint_sha256", None)
    if checkpoint_sha256 is not None:
        signature["checkpoint_sha256"] = checkpoint_sha256
    return signature


def _stage_paths(staging: Path, split: str) -> dict[str, Path]:
    return {
        "X": staging / f"X_{split}.npy",
        "y": staging / f"y_{split}.npy",
        "g": staging / f"g_{split}.npy",
        "record": staging / f"{split}.complete.json",
    }


def _stage_is_current(
    staging: Path,
    *,
    split: str,
    expected_signature: Mapping[str, Any],
    expected_observations: int,
) -> bool:
    paths = _stage_paths(staging, split)
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        record = json.loads(paths["record"].read_text(encoding="utf-8"))
        if record.get("signature") != dict(expected_signature):
            return False
        X = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
        y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
        g = np.load(paths["g"], mmap_mode="r", allow_pickle=False)
        dimension = int(expected_signature["embedding_dimension"])
        return (
            X.shape == (expected_observations, dimension)
            and X.dtype == np.float32
            and y.shape == (expected_observations,)
            and y.dtype == np.int64
            and g.shape == (expected_observations,)
            and g.dtype == np.int64
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _remove_staged_split(staging: Path, split: str) -> None:
    for path in _stage_paths(staging, split).values():
        path.unlink(missing_ok=True)


def _remove_incomplete_temporary_files(staging: Path, split: str) -> None:
    """Remove small interrupted files while preserving resumable embeddings."""
    if not staging.is_dir():
        return
    prefixes = (
        f".y_{split}.tmp-",
        f".g_{split}.tmp-",
        f".{split}.complete.json.tmp-",
        f".X_{split}.partial.json.tmp-",
    )
    for path in staging.iterdir():
        if path.is_file() and path.name.startswith(prefixes):
            path.unlink()


def _partial_embedding_record(staging: Path, split: str) -> Path:
    return staging / f".X_{split}.partial.json"


def _temporary_embedding_pid(path: Path, split: str) -> int | None:
    match = re.fullmatch(rf"\.X_{re.escape(split)}\.tmp-(\d+)\.npy", path.name)
    return None if match is None else int(match.group(1))


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _completed_embedding_prefix(embeddings: np.ndarray) -> int:
    """Locate the first unwritten all-zero row in a sequential staging file."""
    lo, hi = 0, len(embeddings)
    while lo < hi:
        middle = (lo + hi) // 2
        if np.any(embeddings[middle]):
            lo = middle + 1
        else:
            hi = middle
    return lo


def _write_partial_embedding_record(
    path: Path,
    *,
    signature: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {
                "signature": dict(signature),
                "owner_pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _open_partial_embeddings(
    staging: Path,
    *,
    split: str,
    n_examples: int,
    dimension: int,
    batch_size: int,
    signature: Mapping[str, Any],
    overwrite: bool,
) -> tuple[Path, np.memmap, int]:
    """Create or recover a sequential embedding memmap."""
    pattern = f".X_{split}.tmp-*.npy"
    candidates = sorted(staging.glob(pattern))
    record_path = _partial_embedding_record(staging, split)

    active = []
    for candidate in candidates:
        owner = _temporary_embedding_pid(candidate, split)
        if owner is not None and owner != os.getpid() and _process_exists(owner):
            active.append((candidate, owner))
    if active:
        candidate, owner = active[0]
        raise RuntimeError(
            f"Embedding extraction for {split} is already running as PID {owner}: "
            f"{candidate}"
        )

    if overwrite:
        for candidate in candidates:
            candidate.unlink(missing_ok=True)
        record_path.unlink(missing_ok=True)
        candidates = []
    elif len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Multiple interrupted {split} embedding files exist: {names}. "
            "Pass --overwrite-staging to discard them."
        )

    if candidates:
        candidate = candidates[0]
        if record_path.is_file():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Could not validate partial extraction record: {record_path}"
                ) from exc
            if record.get("signature") != dict(signature):
                raise RuntimeError(
                    f"The interrupted {split} extraction has a stale signature. "
                    "Pass --overwrite-staging to discard it."
                )

        current = staging / f".X_{split}.tmp-{os.getpid()}.npy"
        if candidate != current:
            os.replace(candidate, current)
        try:
            embeddings = np.load(current, mmap_mode="r+", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not recover partial embeddings: {current}. "
                "Pass --overwrite-staging to discard them."
            ) from exc
        expected_shape = (n_examples, dimension)
        if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
            del embeddings
            raise RuntimeError(
                f"Partial {split} embeddings have the wrong shape or dtype. "
                "Pass --overwrite-staging to discard them."
            )
        completed = _completed_embedding_prefix(embeddings)
        resume_at = completed - completed % batch_size
        if resume_at < completed:
            embeddings[resume_at:completed] = 0.0
            embeddings.flush()
        _write_partial_embedding_record(record_path, signature=signature)
        print(
            f"Resuming {split} at observation {resume_at:,}/{n_examples:,}.",
            flush=True,
        )
        return current, embeddings, resume_at

    temporary = staging / f".X_{split}.tmp-{os.getpid()}.npy"
    temporary.unlink(missing_ok=True)
    embeddings = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(n_examples, dimension),
    )
    _write_partial_embedding_record(record_path, signature=signature)
    return temporary, embeddings, 0


def extract_split(
    split_source: ImageSplitSource,
    encoder: FrozenImageEncoder,
    *,
    processor_kwargs: Mapping[str, Any],
    split: str,
    batch_size: int,
    staging: str | Path,
    signature: Mapping[str, Any],
    overwrite_incomplete: bool = False,
) -> None:
    """Extract one split and mark it complete only after verification."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Embedding extraction requires tqdm. Install the optional "
            "dependencies with `python -m pip install -e '.[embeddings]'`."
        ) from exc

    staging = Path(staging).expanduser().resolve()
    staging.mkdir(parents=True, exist_ok=True)
    paths = _stage_paths(staging, split)
    n_examples = len(split_source)
    temporary_X, embeddings, resume_at = _open_partial_embeddings(
        staging,
        split=split,
        n_examples=n_examples,
        dimension=int(encoder.dimension),
        batch_size=batch_size,
        signature=signature,
        overwrite=overwrite_incomplete,
    )
    try:
        with tqdm(
            total=n_examples,
            initial=resume_at,
            desc=f"Embedding {split}",
            unit="image",
            dynamic_ncols=True,
        ) as progress:
            for start in range(resume_at, n_examples, batch_size):
                end = min(start + batch_size, n_examples)
                images = split_source.load_batch(start, end)
                values = np.asarray(
                    encoder.encode(images, processor_kwargs=processor_kwargs),
                    dtype=np.float32,
                )
                expected = (end - start, int(encoder.dimension))
                if values.shape != expected:
                    raise RuntimeError(
                        f"Backbone returned shape {values.shape}, expected {expected}."
                    )
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"Non-finite {split} features in observations {start}:{end}."
                    )
                embeddings[start:end] = values
                progress.update(end - start)
                del images, values
        embeddings.flush()
        del embeddings
        os.replace(temporary_X, paths["X"])
        _partial_embedding_record(staging, split).unlink(missing_ok=True)

        for name, values in (("y", split_source.y), ("g", split_source.g)):
            temporary = staging / f".{name}_{split}.tmp-{os.getpid()}.npy"
            temporary.unlink(missing_ok=True)
            np.save(temporary, np.asarray(values, dtype=np.int64), allow_pickle=False)
            os.replace(temporary, paths[name])

        record = {
            "signature": dict(signature),
            "split": split,
            "observations": n_examples,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_record = paths["record"].with_name(
            f".{paths['record'].name}.tmp-{os.getpid()}"
        )
        temporary_record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_record, paths["record"])
    except BaseException:
        if "embeddings" in locals():
            embeddings.flush()
            del embeddings
        raise
    finally:
        split_source.release()
        encoder.release_cache()


def finalize_bundle(
    destination: str | Path,
    *,
    staging: str | Path,
    source: ImageDatasetSource,
    spec: BackboneSpec,
    signature: Mapping[str, Any],
    device: Any,
    dtype: Any,
    representation_seed: int | str | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> EmbeddingBundle:
    """Convert three completed staged splits to one atomic repository bundle."""
    destination = Path(destination).expanduser().resolve()
    staging = Path(staging).expanduser().resolve()
    staged: dict[str, SplitArrays] = {}
    for split in SPLITS:
        split_source = source.splits[split]
        if not _stage_is_current(
            staging,
            split=split,
            expected_signature=signature,
            expected_observations=len(split_source),
        ):
            raise RuntimeError(f"The staged {split} split is missing or stale.")
        paths = _stage_paths(staging, split)
        staged[split] = SplitArrays(
            X=np.load(paths["X"], mmap_mode="r", allow_pickle=False),
            y=np.load(paths["y"], mmap_mode="r", allow_pickle=False),
            g=np.load(paths["g"], mmap_mode="r", allow_pickle=False),
        )

    metadata = {
        "backbone": spec.key,
        "model_name": spec.model_name,
        "feature_type": spec.feature_type,
        "source_fingerprint": source.source_fingerprint,
        "group_definition": source.group_definition,
        "device": str(device),
        "inference_dtype": str(dtype).removeprefix("torch."),
        "frozen_backbone": True,
        # Freezing at extraction time alone does not establish that a model
        # was never fine-tuned. Only known pretrained registry checkpoints
        # receive this declaration; overrides remain explicitly unverified.
        "task_finetuned": (
            False if spec.model_name in {item.model_name for item in BACKBONES.values()}
            else None
        ),
        "per_observation_l2_normalization": (
            spec.per_observation_l2_normalization
        ),
    }
    metadata.update(dict(metadata_extra or {}))
    bundle = write_embedding_bundle(
        destination,
        splits=staged,
        dataset=source.dataset,
        representation=spec.representation,
        representation_seed=representation_seed,
        source=source.source,
        metadata=metadata,
        link_existing_arrays=True,
    )
    shutil.rmtree(staging)
    return bundle


def compare_source_labels_and_groups(
    source: ImageDatasetSource,
    reference: str | Path | EmbeddingBundle,
) -> None:
    """Verify that extraction order, labels, and groups match a bundle."""
    bundle = (
        reference
        if isinstance(reference, EmbeddingBundle)
        else EmbeddingBundle.load(reference, verify_hashes=True)
    )
    for split in SPLITS:
        expected = bundle.split(split)
        observed = source.splits[split]
        if not np.array_equal(observed.y, expected.y):
            raise RuntimeError(
                f"MultiNLI {split} labels/order do not match {bundle.root}."
            )
        if not np.array_equal(observed.g, expected.g):
            raise RuntimeError(
                f"MultiNLI {split} groups/order do not match {bundle.root}."
            )
    print(
        f"Source labels, groups, and split order exactly match: {bundle.root}",
        flush=True,
    )


def compare_embedding_bundles(
    candidate: str | Path | EmbeddingBundle,
    reference: str | Path | EmbeddingBundle,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    chunk_size: int = 4096,
) -> dict[str, dict[str, Any]]:
    """Compare two bundles without loading their full feature arrays into RAM."""
    if atol < 0 or rtol < 0:
        raise ValueError("Comparison tolerances must be nonnegative.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    candidate_bundle = (
        candidate
        if isinstance(candidate, EmbeddingBundle)
        else EmbeddingBundle.load(candidate, verify_hashes=True)
    )
    reference_bundle = (
        reference
        if isinstance(reference, EmbeddingBundle)
        else EmbeddingBundle.load(reference, verify_hashes=True)
    )
    report: dict[str, dict[str, Any]] = {}
    equivalent = True
    for split in SPLITS:
        observed = candidate_bundle.split(split)
        expected = reference_bundle.split(split)
        labels_equal = np.array_equal(observed.y, expected.y)
        groups_equal = np.array_equal(observed.g, expected.g)
        shape_equal = observed.X.shape == expected.X.shape
        exact = shape_equal
        close = shape_equal
        max_absolute_error = 0.0
        squared_error = 0.0
        values_compared = 0
        if shape_equal:
            for start in range(0, len(observed.X), chunk_size):
                end = min(start + chunk_size, len(observed.X))
                observed_chunk = np.asarray(observed.X[start:end])
                expected_chunk = np.asarray(expected.X[start:end])
                difference = observed_chunk.astype(np.float64) - expected_chunk
                exact = exact and np.array_equal(observed_chunk, expected_chunk)
                close = close and np.allclose(
                    observed_chunk,
                    expected_chunk,
                    atol=atol,
                    rtol=rtol,
                )
                if difference.size:
                    max_absolute_error = max(
                        max_absolute_error,
                        float(np.max(np.abs(difference))),
                    )
                    squared_error += float(np.sum(difference * difference))
                    values_compared += difference.size
        split_equivalent = labels_equal and groups_equal and close
        equivalent = equivalent and split_equivalent
        report[split] = {
            "shape_equal": shape_equal,
            "labels_equal": labels_equal,
            "groups_equal": groups_equal,
            "embeddings_exact": exact,
            "embeddings_allclose": close,
            "max_absolute_error": max_absolute_error,
            "rmse": (
                float(np.sqrt(squared_error / values_compared))
                if values_compared
                else 0.0
            ),
            "equivalent": split_equivalent,
        }
    if not equivalent:
        raise RuntimeError(
            "Extracted embeddings do not match the reference bundle within "
            f"atol={atol:g}, rtol={rtol:g}: {json.dumps(report, sort_keys=True)}"
        )
    return report


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen image or MultiNLI embeddings into a NumPy bundle."
    )
    parser.add_argument(
        "--dataset",
        choices=("celeba", "waterbirds", "wb", "multinli", "mnli"),
        default="celeba",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--backbone",
        choices=tuple(sorted(set(BACKBONES) | set(BACKBONE_ALIASES))),
        default="dinov3-vitb16",
    )
    parser.add_argument(
        "--model",
        help=(
            "Image checkpoint override, or required fine-tuned BERT checkpoint "
            "(.pt state_dict, Hugging Face directory, or model identifier) for "
            "MultiNLI extraction."
        ),
    )
    parser.add_argument(
        "--base-model",
        default="bert-base-uncased",
        help="Base model used to initialize a file-based MultiNLI state_dict.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Required representation seed for a MultiNLI output bundle.",
    )
    parser.add_argument(
        "--representation",
        help="Optional manifest representation name for a checkpoint override.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load a previously cached checkpoint without contacting the hub.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to extract now; the bundle is finalized once all exist.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="CelebA or MultiNLI metadata CSV override.",
    )
    parser.add_argument("--images-dir", type=Path, help="CelebA image directory.")
    parser.add_argument("--target-column", default="y")
    parser.add_argument("--spurious-column", default="Female")
    parser.add_argument("--waterbirds-input", type=Path)
    parser.add_argument(
        "--multinli-features-dir",
        type=Path,
        help="Directory containing the three cached MultiNLI BERT feature files.",
    )
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        help=(
            "Validate source ordering and compare completed embeddings against "
            "an existing bundle."
        ),
    )
    parser.add_argument("--comparison-atol", type=float, default=1e-6)
    parser.add_argument("--comparison-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip the up-front CelebA check that every image path exists.",
    )
    parser.add_argument(
        "--overwrite-staging",
        action="store_true",
        help="Replace stale or current requested staging splits, never a bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and print split counts without loading a model.",
    )
    return parser.parse_args(argv)


def _print_source_summary(
    source: ImageDatasetSource,
    *,
    spec: BackboneSpec,
    destination: Path,
) -> None:
    print(f"Dataset: {source.dataset}")
    print(f"Backbone: {spec.key} ({spec.model_name})")
    print(f"Representation: {spec.representation}")
    print(f"Destination: {destination}")
    for split in SPLITS:
        values = source.splits[split]
        labels = {
            int(value): int(count)
            for value, count in zip(*np.unique(values.y, return_counts=True))
        }
        groups = {
            int(value): int(count)
            for value, count in zip(*np.unique(values.g, return_counts=True))
        }
        print(
            f"  {split}: n={len(values):,} labels={labels} groups={groups}"
        )


def _print_storage_summary(
    source: ImageDatasetSource,
    *,
    embedding_dimension: int,
    destination: Path,
) -> None:
    observations = sum(len(source.splits[split]) for split in SPLITS)
    feature_bytes = observations * embedding_dimension * np.dtype(np.float32).itemsize
    label_bytes = observations * 2 * np.dtype(np.int64).itemsize
    expected_bytes = feature_bytes + label_bytes

    disk_path = destination.parent
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    free_bytes = shutil.disk_usage(disk_path).free
    print(
        "Expected final bundle size: "
        f"{expected_bytes / 2**30:.2f} GiB "
        f"({observations:,} x {embedding_dimension} float32 features)."
    )
    print(f"Available disk space: {free_bytes / 2**30:.2f} GiB.")


def _existing_bundle_is_current(
    destination: Path,
    *,
    source: ImageDatasetSource,
    spec: BackboneSpec,
) -> bool:
    if not destination.exists():
        return False
    bundle = EmbeddingBundle.load(destination, verify_hashes=False)
    metadata = bundle.manifest.get("metadata", {})
    current = (
        bundle.manifest.get("dataset") == source.dataset
        and bundle.manifest.get("representation") == spec.representation
        and metadata.get("model_name") == spec.model_name
        and metadata.get("feature_type") == spec.feature_type
        and metadata.get("per_observation_l2_normalization")
        == spec.per_observation_l2_normalization
        and metadata.get("source_fingerprint") == source.source_fingerprint
    )
    if not current:
        return False
    if spec.adapter == "huggingface_bert_final_cls_token":
        checkpoint = Path(spec.model_name).expanduser()
        if checkpoint.is_file():
            expected = metadata.get("checkpoint_sha256")
            return isinstance(expected, str) and sha256_file(checkpoint) == expected
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    is_multinli = args.dataset in {"multinli", "mnli"}
    if is_multinli:
        if args.seed is None:
            raise ValueError("MultiNLI extraction requires --seed.")
        if args.seed < 0:
            raise ValueError("--seed must be nonnegative.")
        if args.model is None and not args.dry_run:
            raise ValueError("MultiNLI extraction requires --model.")
        spec = resolve_multinli_spec(
            args.model or args.base_model,
            representation=args.representation,
        )
    else:
        spec = resolve_backbone(
            args.backbone,
            model_override=args.model,
            representation_override=args.representation,
        )
    source = load_dataset_source(
        args.dataset,
        data_root=args.data_root,
        metadata_path=args.metadata,
        images_dir=args.images_dir,
        waterbirds_input=args.waterbirds_input,
        multinli_features_dir=args.multinli_features_dir,
        target_column=args.target_column,
        spurious_column=args.spurious_column,
        verify_images=not args.skip_image_validation,
    )
    if args.destination is None:
        destination = (
            multinli_destination(
                args.artifact_root,
                representation=spec.representation,
                seed=int(args.seed),
            )
            if is_multinli
            else default_destination(
                args.artifact_root,
                dataset=source.dataset,
                representation=spec.representation,
            )
        )
    else:
        destination = args.destination.expanduser().resolve()
    _print_source_summary(source, spec=spec, destination=destination)
    reference = (
        None
        if args.reference_bundle is None
        else args.reference_bundle.expanduser().resolve()
    )
    if reference is not None:
        compare_source_labels_and_groups(source, reference)
    if args.dry_run:
        print("Dry run complete; no model was loaded and no files were written.")
        return

    if destination.exists():
        if _existing_bundle_is_current(destination, source=source, spec=spec):
            print(f"Skipping current embedding bundle: {destination}")
            if reference is not None:
                report = compare_embedding_bundles(
                    destination,
                    reference,
                    atol=args.comparison_atol,
                    rtol=args.comparison_rtol,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
            return
        raise FileExistsError(
            f"Refusing to overwrite existing embedding bundle: {destination}. "
            "Use a distinct --representation or --destination."
        )

    requested_splits = tuple(dict.fromkeys(args.splits))
    encoder = (
        load_multinli_encoder(
            spec,
            base_model=args.base_model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        if is_multinli
        else load_image_encoder(
            spec,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    )
    _print_storage_summary(
        source,
        embedding_dimension=encoder.dimension,
        destination=destination,
    )
    signature = _stage_signature(source, spec, encoder)
    staging = destination.parent / f".{destination.name}.gather"

    for split in requested_splits:
        split_source = source.splits[split]
        _remove_incomplete_temporary_files(staging, split)
        current = _stage_is_current(
            staging,
            split=split,
            expected_signature=signature,
            expected_observations=len(split_source),
        )
        if current and not args.overwrite_staging:
            print(f"Skipping current staged split: {split}")
            split_source.release()
            continue
        staged_paths = _stage_paths(staging, split)
        if any(path.exists() for path in staged_paths.values()):
            incomplete = not staged_paths["record"].is_file()
            if incomplete:
                print(f"Replacing interrupted staged split: {split}")
                _remove_staged_split(staging, split)
            elif not args.overwrite_staging:
                raise FileExistsError(
                    f"Stale staging files exist for {split} in {staging}. "
                    "Pass --overwrite-staging to replace that split."
                )
            else:
                _remove_staged_split(staging, split)
        extract_split(
            split_source,
            encoder,
            processor_kwargs=source.processor_kwargs,
            split=split,
            batch_size=args.batch_size,
            staging=staging,
            signature=signature,
            overwrite_incomplete=args.overwrite_staging,
        )

    missing = [
        split
        for split in SPLITS
        if not _stage_is_current(
            staging,
            split=split,
            expected_signature=signature,
            expected_observations=len(source.splits[split]),
        )
    ]
    if missing:
        print(
            "Staged requested splits. Run the remaining split(s) to finalize: "
            + ", ".join(missing)
        )
        return

    bundle = finalize_bundle(
        destination,
        staging=staging,
        source=source,
        spec=spec,
        signature=signature,
        device=encoder.device,
        dtype=encoder.dtype,
        representation_seed=args.seed if is_multinli else None,
        metadata_extra=(
            {
                "base_model": args.base_model,
                "checkpoint_sha256": getattr(
                    encoder, "checkpoint_sha256", None
                ),
                "checkpoint_metadata": getattr(
                    encoder, "checkpoint_metadata", {}
                ),
            }
            if is_multinli
            else None
        ),
    )
    print(f"Saved validated embedding bundle: {bundle.root}")
    if reference is not None:
        report = compare_embedding_bundles(
            bundle,
            reference,
            atol=args.comparison_atol,
            rtol=args.comparison_rtol,
        )
        print("Embedding comparison passed:")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
