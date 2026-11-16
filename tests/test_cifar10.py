"""Offline checks for split isolation, pooled features, and the paper protocol."""
import numpy as np
import pytest

import gather_embeddings as gather
from data import EmbeddingBundle, SplitArrays, write_embedding_bundle
from hyper_param_sweep import _resolved_representation, parse_args, resolved_config, sweep_bundle
from paper_protocol import RIDGE_LAMBDAS


@pytest.mark.parametrize('dataset,n_classes', [('CIFAR10', 10), ('CIFAR100', 100)])
def test_cifar10_split_is_stratified_disjoint_and_reproducible(monkeypatch, tmp_path, dataset, n_classes):
    from torchvision import datasets
    from PIL import Image

    calls = []

    class FakeCIFAR10:
        base_folder = "cifar-10-batches-py"
        train_list = [("train", "unused")]
        test_list = [("test", "unused")]
        meta = {"filename": "meta"}
        classes = [str(i) for i in range(n_classes)]

        def __init__(self, root, train, download):
            calls.append((train, download))
            self.targets = np.repeat(np.arange(n_classes), 100 if train else 10).tolist()

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return Image.new("RGB", (32, 32)), self.targets[index]

    monkeypatch.setattr(datasets, dataset, FakeCIFAR10)
    monkeypatch.setattr(gather, "sha256_file", lambda path: "test-hash")
    source = gather.load_cifar_source(tmp_path, dataset=dataset)
    repeat = gather.load_cifar_source(tmp_path, dataset=dataset)
    other = gather.load_cifar_source(tmp_path, dataset=dataset, split_seed=1, download=False)
    assert source.dataset == dataset
    assert calls[:2] == [(True, True), (False, True)]
    assert calls[-2:] == [(True, False), (False, False)]
    assert source.source_fingerprint == repeat.source_fingerprint
    assert source.source_fingerprint != other.source_fingerprint
    train, val, test = [source.splits[name] for name in ("train", "val", "test")]
    assert not np.intersect1d(train.indices, val.indices).size
    np.testing.assert_array_equal(np.sort(np.r_[train.indices, val.indices]), np.arange(n_classes*100))
    np.testing.assert_array_equal(test.indices, np.arange(n_classes*10))
    np.testing.assert_array_equal(test.indices, other.splits["test"].indices)
    for split, count in ((train, 90), (val, 10), (test, 10)):
        np.testing.assert_array_equal(np.bincount(split.y), np.full(n_classes, count))
        np.testing.assert_array_equal(split.g, split.y)
    assert train.load_batch(0, 1)[0].size == (32, 32)
    assert source.source["spurious_attribute"] is None


def test_resnet_features_are_exactly_the_fc_input(monkeypatch):
    import torch
    from torchvision.models import resnet50
    from PIL import Image

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    reference = resnet50(weights=None).eval()
    state = reference.state_dict()
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **kw: state)
    monkeypatch.setattr(gather, "sha256_file", lambda path: "test-hash")
    encoder = gather.TorchvisionResNet50Encoder(
        gather.resolve_backbone("resnet50-imagenet"), device=torch.device("cpu"),
        dtype=torch.float32,
    )
    images = [Image.new("RGB", (32, 32), color=(100, 150, 200))]
    captured = []
    hook = reference.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.inference_mode():
        reference(torch.stack([encoder.processor(image) for image in images]))
    hook.remove()
    actual = encoder.encode(images, processor_kwargs={})
    torch.set_num_threads(previous_threads)
    assert actual.shape == (1, 2048)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, captured[0].numpy(), rtol=1e-5, atol=1e-6)
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_unseeded_paper_grid_cli():
    config = resolved_config(parse_args([
        "--dataset", "CIFAR10", "--representation", "resnet50_imagenet_v2",
        "--seed", "none", "--transform", "identity", "whiten", "--paper-grid",
    ]))
    assert _resolved_representation(config) == ("resnet50_imagenet_v2", (None,))
    assert config["ridge_lambdas"] == RIDGE_LAMBDAS
    assert len(RIDGE_LAMBDAS) == 24
    assert RIDGE_LAMBDAS[0] == 1e-5 and RIDGE_LAMBDAS[-1] == 5e6
    with pytest.raises(SystemExit):
        parse_args(["--paper-grid", "--ridge-lambda", "1"])


def test_ten_class_bundle_runs_both_transforms(tmp_path):
    rng = np.random.default_rng(3)
    splits = {}
    for name in ("train", "val", "test"):
        y = np.tile(np.arange(10), 5)
        X = np.eye(10)[y] + rng.normal(scale=0.1, size=(50, 10))
        splits[name] = SplitArrays(X=X.astype(np.float32), y=y, g=y.copy())
    bundle = write_embedding_bundle(
        tmp_path / "bundle", splits=splits, dataset="CIFAR10",
        representation="test", representation_seed=None, source={}, metadata={},
    )
    bundle = EmbeddingBundle.load(bundle.root, verify_hashes=True)
    runs = sweep_bundle(bundle, transforms=("identity", "whiten"),
                        ridge_lambdas=(0.1,), class_balanced=False)
    assert len(runs) == 2
    for run in runs:
        assert run["status"] == "complete"
        assert run["fit"]["solver"] == "lbfgs"
        assert run["protocol"]["group_labels_used_for_model_selection"] is False
        assert run["splits"]["test"]["accuracy"] == 1.0
