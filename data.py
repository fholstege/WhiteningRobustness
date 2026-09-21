"""Validated NumPy embedding loading and bundle creation.

The evaluator never needs to know how a representation was originally
generated. It receives one directory ("bundle") with three splits and a
manifest describing their files, shapes, dtypes, and checksums.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from scipy.special import expit, softmax

from numerics import STORED_EMBEDDING_DTYPE


SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
SAVED_HEAD_SCHEMA_VERSION = 1


# =============================================================================
# Small data containers and checksum helper
# =============================================================================

def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file without loading it into memory."""
    # Read in chunks because embedding files may be several gigabytes.
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitArrays:
    """One immutable embedding split."""

    X: np.ndarray
    y: np.ndarray
    g: np.ndarray


@dataclass(frozen=True)
class SavedLinearHead:
    """Read-only NumPy copy of one archived trained linear classifier."""

    root: Path
    weight: np.ndarray
    bias: np.ndarray
    classes_: np.ndarray
    manifest: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        classes: np.ndarray,
        input_features: int,
        verify_hashes: bool = False,
    ) -> "SavedLinearHead":
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Saved-head manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SAVED_HEAD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported saved-head schema {manifest.get('schema_version')!r}; "
                f"expected {SAVED_HEAD_SCHEMA_VERSION}."
            )
        arrays = {}
        for name in ("weight", "bias"):
            spec = manifest.get("parameters", {}).get(name)
            if not isinstance(spec, dict):
                raise KeyError(f"Saved-head manifest is missing parameters.{name}.")
            path = EmbeddingBundle._safe_child(root, spec.get("file"))
            if path.suffix != ".npy" or not path.is_file():
                raise ValueError(f"Saved-head parameter must be an existing .npy file: {path}")
            if verify_hashes and sha256_file(path) != spec.get("sha256"):
                raise ValueError(f"Checksum mismatch for {path}.")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.dtype != np.float32 or list(value.shape) != spec.get("shape"):
                raise ValueError(f"Saved-head parameter metadata mismatch: {path}")
            if not np.isfinite(value).all():
                raise ValueError(f"Saved-head parameter contains non-finite values: {path}")
            arrays[name] = value
        classes = np.asarray(classes).reshape(-1)
        weight, bias = arrays["weight"], arrays["bias"]
        expected_outputs = 1 if classes.size == 2 else classes.size
        if weight.shape != (expected_outputs, int(input_features)):
            raise ValueError(
                f"Saved-head weight shape {weight.shape} is incompatible with "
                f"{classes.size} classes and {input_features} features."
            )
        if bias.shape != (expected_outputs,):
            raise ValueError(f"Saved-head bias shape {bias.shape} is incompatible with weight.")
        return cls(root, weight, bias, classes.copy(), manifest)

    @property
    def coef_(self) -> np.ndarray:
        return self.weight

    @property
    def intercept_(self) -> np.ndarray:
        return self.bias

    @property
    def fit_intercept(self) -> bool:
        return True

    @property
    def n_features_in_(self) -> int:
        return int(self.weight.shape[1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(X) @ np.asarray(self.weight).T + np.asarray(self.bias)
        return scores[:, 0] if self.classes_.size == 2 else scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self.classes_.size == 2:
            p1 = expit(scores)
            return np.column_stack((1.0 - p1, p1))
        return softmax(scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        indices = (scores >= 0).astype(np.int64) if self.classes_.size == 2 else np.argmax(scores, axis=1)
        return self.classes_[indices]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "saved_trained_last_layer",
            "path": str(self.root),
            "classes": self.classes_.tolist(),
            "input_features": self.n_features_in_,
            "output_logits": int(self.weight.shape[0]),
            "source_pt_path": self.manifest.get("source", {}).get("path"),
            "source_pt_sha256": self.manifest.get("source", {}).get("sha256"),
            "parameter_sha256": {
                name: self.manifest["parameters"][name]["sha256"]
                for name in ("weight", "bias")
            },
        }


def write_saved_linear_head(
    destination: str | Path,
    *,
    weight: np.ndarray,
    bias: np.ndarray,
    dataset: str,
    representation: str,
    representation_seed: int | str | None,
    source: Mapping[str, Any],
) -> Path:
    """Atomically store a NumPy runtime copy of an archived linear head."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite saved head: {destination}")
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("Saved-head weight must be 2D with one bias per output logit.")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Saved-head parameters must be finite.")

    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary saved-head directory exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        parameters: dict[str, dict[str, Any]] = {}
        for name, value in (("weight", weight), ("bias", bias)):
            filename = f"{name}.npy"
            path = temporary / filename
            np.save(path, value, allow_pickle=False)
            parameters[name] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_file(path),
            }
        manifest = {
            "schema_version": SAVED_HEAD_SCHEMA_VERSION,
            "dataset": dataset,
            "representation": representation,
            "representation_seed": representation_seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": dict(source),
            "parameters": parameters,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


@dataclass(frozen=True)
class EmbeddingBundle:
    """A validated directory of NumPy embedding arrays and metadata."""

    root: Path
    manifest: Mapping[str, Any]
    splits: Mapping[str, SplitArrays]
    fingerprint: str

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        mmap_mode: str | None = "r",
        verify_hashes: bool = False,
    ) -> "EmbeddingBundle":
        # Resolve once so every later path check is against an absolute root.
        root = Path(root).expanduser().resolve()
        if root.suffix.lower() in {".pt", ".pth"}:
            raise ValueError("Production code only loads NumPy embedding bundles.")

        # The manifest is the source of truth for all array filenames and
        # expected metadata. Array contents are never unpickled.
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported embedding schema {manifest.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}."
            )
        declared_splits = manifest.get("splits")
        if not isinstance(declared_splits, dict):
            raise TypeError("manifest['splits'] must be an object.")

        loaded: dict[str, SplitArrays] = {}
        for split in SPLITS:
            spec = declared_splits.get(split)
            if not isinstance(spec, dict):
                raise KeyError(f"Manifest is missing the {split!r} split.")
            arrays: dict[str, np.ndarray] = {}
            for logical_name in ("X", "y", "g"):
                array_spec = spec.get(logical_name)
                if not isinstance(array_spec, dict):
                    raise KeyError(f"Manifest is missing {split}.{logical_name}.")
                path = cls._safe_child(root, array_spec.get("file"))
                if path.suffix != ".npy":
                    raise ValueError(f"Expected a .npy array, received: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if verify_hashes:
                    # Hashing is optional because it reads the complete file.
                    expected = array_spec.get("sha256")
                    observed = sha256_file(path)
                    if not expected or observed != expected:
                        raise ValueError(f"Checksum mismatch for {path}.")
                # Memory mapping lets evaluation work without copying the full
                # representation into RAM. Pickled NumPy objects are forbidden.
                array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
                expected_shape = tuple(array_spec.get("shape", ()))
                expected_dtype = np.dtype(array_spec.get("dtype"))
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path}: shape {array.shape} != manifest {expected_shape}."
                    )
                if array.dtype != expected_dtype:
                    raise ValueError(
                        f"{path}: dtype {array.dtype} != manifest {expected_dtype}."
                    )
                arrays[logical_name] = array
            cls._validate_split(split, arrays)
            loaded[split] = SplitArrays(**arrays)

        cls._validate_across_splits(loaded)
        fingerprint = sha256(raw_manifest).hexdigest()
        return cls(root=root, manifest=manifest, splits=loaded, fingerprint=fingerprint)

    @staticmethod
    def _safe_child(root: Path, value: Any) -> Path:
        """Reject absolute paths and ``..`` escapes in the manifest."""
        if not isinstance(value, str) or not value:
            raise TypeError("Array filenames must be non-empty strings.")
        child = (root / value).resolve()
        if child.parent != root:
            raise ValueError(f"Array path escapes its bundle: {value!r}.")
        return child

    @staticmethod
    def _validate_split(split: str, arrays: Mapping[str, np.ndarray]) -> None:
        """Check the storage contract for one train/val/test split."""
        X, y, g = arrays["X"], arrays["y"], arrays["g"]
        if X.ndim != 2:
            raise ValueError(f"{split}: X must have shape (observations, features).")
        if y.ndim != 1 or g.ndim != 1:
            raise ValueError(f"{split}: y and g must be one-dimensional.")
        if len(X) == 0 or len(X) != len(y) or len(X) != len(g):
            raise ValueError(f"{split}: X, y, and g have inconsistent lengths.")
        if X.dtype != np.dtype(STORED_EMBEDDING_DTYPE):
            raise ValueError(
                f"{split}: stored features must use "
                f"{np.dtype(STORED_EMBEDDING_DTYPE).name}."
            )
        if y.dtype != np.int64 or g.dtype != np.int64:
            raise ValueError(f"{split}: y and g must use int64.")
        if not np.isfinite(X).all():
            raise ValueError(f"{split}: X contains non-finite values.")
        if np.any(g < 0):
            raise ValueError(f"{split}: group identifiers must be nonnegative.")

    @staticmethod
    def _validate_across_splits(splits: Mapping[str, SplitArrays]) -> None:
        """Ensure all splits can be evaluated by one fitted prediction head."""
        dimensions = {values.X.shape[1] for values in splits.values()}
        if len(dimensions) != 1:
            raise ValueError("Feature dimensions differ across splits.")
        train_classes = np.unique(splits["train"].y)
        train_groups = np.unique(splits["train"].g)
        expected_groups = np.arange(train_groups.size)
        if not np.array_equal(train_groups, expected_groups):
            raise ValueError("Training group identifiers must be contiguous from zero.")
        for split, values in splits.items():
            if not np.all(np.isin(values.y, train_classes)):
                raise ValueError(f"{split}: contains a class absent from training.")
            if not np.all(np.isin(values.g, train_groups)):
                raise ValueError(f"{split}: contains a group absent from training.")

    def split(self, name: str) -> SplitArrays:
        """Return one validated split by its conventional name."""
        try:
            return self.splits[name]
        except KeyError as exc:
            raise KeyError(f"Unknown split {name!r}; choose from {SPLITS}.") from exc


def write_embedding_bundle(
    destination: str | Path,
    *,
    splits: Mapping[str, SplitArrays],
    dataset: str,
    representation: str,
    representation_seed: int | str | None,
    source: Mapping[str, Any],
    metadata: Mapping[str, Any],
    link_existing_arrays: bool = False,
) -> EmbeddingBundle:
    """Atomically write a new bundle without modifying its source artifact.

    When ``link_existing_arrays`` is true, every input must be a memory-mapped
    ``.npy`` file with the required storage dtype. The bundle then hard-links
    those files instead of copying them. This is useful for promoting validated
    extraction staging files without temporarily doubling their disk usage.
    """
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite embedding bundle: {destination}")
    if set(splits) != set(SPLITS):
        raise ValueError(f"splits must contain exactly {SPLITS}.")

    # Write into a sibling temporary directory. Only rename it into place after
    # every array and the manifest have been written successfully.
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary bundle already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        manifest_splits: dict[str, dict[str, Any]] = {}
        normalized_splits: dict[str, SplitArrays] = {}
        for split in SPLITS:
            values = splits[split]
            # Normalize storage dtypes once at the bundle boundary.
            normalized = SplitArrays(
                X=np.asarray(values.X, dtype=STORED_EMBEDDING_DTYPE),
                y=np.asarray(values.y, dtype=np.int64).reshape(-1),
                g=np.asarray(values.g, dtype=np.int64).reshape(-1),
            )
            EmbeddingBundle._validate_split(
                split,
                {"X": normalized.X, "y": normalized.y, "g": normalized.g},
            )
            normalized_splits[split] = normalized
            manifest_splits[split] = {}
            for name, array in (
                ("X", normalized.X),
                ("y", normalized.y),
                ("g", normalized.g),
            ):
                filename = f"{name}_{split}.npy"
                path = temporary / filename
                # .npy preserves shape/dtype and supports safe memory mapping.
                if link_existing_arrays:
                    source_array = getattr(values, name)
                    if not isinstance(source_array, np.memmap):
                        raise TypeError(
                            "link_existing_arrays requires memory-mapped .npy "
                            f"inputs; {split}.{name} is not memory-mapped."
                        )
                    original = getattr(source_array, "filename", None)
                    if original is None:
                        raise TypeError(
                            "link_existing_arrays requires memory-mapped .npy "
                            f"inputs; {split}.{name} is not memory-mapped."
                        )
                    original = Path(original).expanduser().resolve()
                    if original.suffix != ".npy" or not original.is_file():
                        raise ValueError(
                            f"Cannot hard-link invalid .npy source: {original}"
                        )
                    if getattr(values, name).dtype != array.dtype:
                        raise ValueError(
                            f"Cannot hard-link {split}.{name} because its "
                            "storage dtype requires conversion."
                        )
                    stored = np.load(
                        original, mmap_mode="r", allow_pickle=False
                    )
                    if (
                        stored.shape != array.shape
                        or stored.dtype != array.dtype
                        or source_array.shape != stored.shape
                        or source_array.strides != stored.strides
                        or source_array.offset != stored.offset
                    ):
                        raise ValueError(
                            f"Cannot hard-link {split}.{name}: the complete "
                            "source file does not match the supplied array."
                        )
                    del stored
                    os.link(original, path)
                else:
                    np.save(path, array, allow_pickle=False)
                manifest_splits[split][name] = {
                    "file": filename,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": sha256_file(path),
                }
        EmbeddingBundle._validate_across_splits(normalized_splits)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "representation": representation,
            "representation_seed": representation_seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": dict(source),
            "metadata": dict(metadata),
            "splits": manifest_splits,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        # A failed conversion must not leave a bundle that looks complete.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return EmbeddingBundle.load(destination, verify_hashes=True)
