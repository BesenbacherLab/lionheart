import numpy.testing as npt
from generalize.evaluate.roc_curves import ROCCurves

from lionheart.utils.global_vars import INCLUDED_MODELS


def test_customize_thresholds_shared_resources(
    run_cli,
    tmp_path,
    resource_path,
):
    output_subdir = "roc_output"

    command_args = [
        "lionheart",
        "customize_thresholds",
        "--resources_dir",
        resource_path,
        "--model_name",
        "detect_cancer__001__25_11_24",
        "--use_included_features",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = ["ROC_curves.json"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    roc = ROCCurves.load(tmp_path / output_subdir / "ROC_curves.json").get("Custom ROC")

    npt.assert_almost_equal(roc.auc, 0.8837, decimal=4)


def test_customize_thresholds_single_dataset_custom_model(
    run_cli,
    tmp_path,
    resource_path,
):
    output_subdir = "roc_output"

    command_args = [
        "lionheart",
        "customize_thresholds",
        "--dataset_paths",
        resource_path / "shared_features" / "GECOCA" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "GECOCA" / "meta_data.csv",
        "--custom_model_dir",
        resource_path / "models" / INCLUDED_MODELS[0],
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = ["ROC_curves.json"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    roc = ROCCurves.load(tmp_path / output_subdir / "ROC_curves.json").get("Custom ROC")

    npt.assert_almost_equal(roc.auc, 0.8479, decimal=4)
