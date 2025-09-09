import numpy as np
import numpy.testing as npt
import pandas as pd
from utipy import mk_dir

from lionheart.utils.global_vars import INCLUDED_MODELS


def test_predict(run_cli, tmp_path, resource_path, lionheart_features):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "prediction_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    assert scores.shape == (10, 898)
    np.save(sample_dir / "dataset" / "feature_dataset.npy", scores)

    command_args = [
        "lionheart",
        "predict_sample",
        "--sample_dir",
        tmp_path / "test_sample",
        "--resources_dir",
        resource_path,
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = ["prediction.csv", "README.txt"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert expected_file in generated_files, (
            f"Expected file {expected_file} not found."
        )

    # Check prediction
    prediction = pd.read_csv(tmp_path / output_subdir / "prediction.csv")
    print(prediction)

    assert prediction["Prediction"].tolist() == ["Cancer"] * 6
    assert np.round(prediction["P(Cancer)"], decimals=4).tolist() == [0.9481] * 6


def test_predict_with_custom_model_and_roc(
    run_cli, tmp_path, resource_path, lionheart_features
):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "prediction_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    assert scores.shape == (10, 898)
    np.save(sample_dir / "dataset" / "feature_dataset.npy", scores)

    ## Create custom ROC curve:

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
        output_subdir="roc_output",
    )

    custom_threshold_dir = tmp_path / "roc_output"

    # Predict

    command_args = [
        "lionheart",
        "predict_sample",
        "--sample_dir",
        tmp_path / "test_sample",
        "--resources_dir",
        resource_path,
        "--custom_threshold_dirs",
        custom_threshold_dir,
        "--custom_model_dirs",
        resource_path / "models" / INCLUDED_MODELS[0],
        "--model_names",
        "none",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = ["prediction.csv", "README.txt"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert expected_file in generated_files, (
            f"Expected file {expected_file} not found."
        )

    # Check prediction
    prediction = pd.read_csv(tmp_path / output_subdir / "prediction.csv")
    pd.set_option("display.max_columns", None)
    print(prediction)

    assert (
        prediction["Prediction"].tolist() == ["Cancer"] * 12
    )  # Also tests size of data frame

    npt.assert_equal(
        prediction.loc[[0, 6], "ROC Curve"].tolist(),
        ["Average (training data)", "Custom 0"],
    )
    # Max. J differs between the thresholds
    npt.assert_almost_equal(
        prediction.loc[[0, 6], "Threshold"].tolist(),
        [0.4888, 0.6229],
        decimal=4,
    )
    # Exp. accuracy differs between probability density files
    npt.assert_almost_equal(
        prediction.loc[[0, 6], "Exp. Accuracy for Class at Probability"].tolist(),
        [0.9542, 0.9597],
        decimal=4,
    )
