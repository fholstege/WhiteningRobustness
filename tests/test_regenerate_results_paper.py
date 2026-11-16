from regenerate_results_paper import _comparison_config, _finetune_config, expected_paths


def test_default_result_plan_matches_the_paper_plot_inputs():
    paths = expected_paths(include_theory_checks=False)
    names = {path.name for path in paths}
    assert len(paths) == 30
    assert "synthetic_gamma_sweep.json" in names
    assert "WB_resnet50_finetune.json" in names
    assert "multiNLI_debertav3_finetune.json" in names
    assert "WB_resnet50_AFR_aggregate.json" in names
    assert "WB_resnet50_NEUROTUNE_aggregate_cb.json" in names
    assert "synthetic_theory_checks.json" not in names


def test_theory_result_is_an_explicit_optional_input():
    paths = expected_paths(include_theory_checks=True)
    assert len(paths) == 31
    assert paths[-1].name == "synthetic_theory_checks.json"


def test_result_plan_can_target_an_isolated_output_directory(tmp_path):
    paths = expected_paths(results_dir=tmp_path / "paper_test", include_theory_checks=False)
    assert len(paths) == 30
    assert all(path.is_relative_to(tmp_path / "paper_test") for path in paths)
    assert any(path.parent.name == "comparisons" for path in paths)


def test_paper_specific_hyperparameter_exceptions_are_preserved():
    finetune = _finetune_config(overwrite=True, workers=1)
    bert = next(item for item in finetune["experiments"] if item["embedding"] == "BERT")
    assert [item["whitening_estimator"] for item in bert["whitening_variants"]] == [
        "empirical", "ledoit-wolf", "ledoit-wolf-nonlinear",
    ]

    afr = _comparison_config("AFR", overwrite=True, workers=1)
    multinli = next(item for item in afr["experiments"] if item["dataset"] == "multiNLI")
    assert multinli["afr_gammas"] == [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    assert afr["sweep_defaults"]["afr_gammas"][:2] == [0.01, 0.05]


def test_controls_use_the_paper_seeds_and_report_paths(tmp_path):
    config = _finetune_config(overwrite=False, workers=1, controls=True)
    assert [(item["dataset"], item["seeds"], item["class_balanced"])
            for item in config["experiments"]] == [
        ("WB_no_spurious", [1, 2, 3, 4, 5], True), ("CIFAR10", [0, 1, 2], False),
    ]
    paths = expected_paths(results_dir=tmp_path, section="controls", include_theory_checks=False)
    assert {path.name for path in paths} == {
        f"{dataset}_resnet50_finetune{suffix}.json"
        for dataset in ("WB_no_spurious", "CIFAR10")
        for suffix in ("", "_aggregate")
    }


def test_synthetic_paper_config_pins_main_and_appendix_scales(tmp_path):
    from regenerate_results_paper import _synthetic_config

    common = dict(output=tmp_path / "report.json", overwrite=False, workers=1, theory=False)
    main = _synthetic_config(**common)
    appendix = _synthetic_config(**common, appendix=True)
    assert main["gamma"] == 5.0 and "gamma_values" not in main
    assert appendix["gamma"] == 3.0 and appendix["gamma_values"] == [0.5, 1.0, 3.0]
    for key in ("base_seed", "n_train", "d_n_ratio", "n_test", "n_sim"):
        assert main[key] == appendix[key]
    assert main["n_sim"] == 100
