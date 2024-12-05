import numpy as np
import pandas as pd
from utipy import mk_dir

from lionheart.utils.global_vars import INCLUDED_MODELS


def test_predict(run_cli, tmp_path, resource_path, lionheart_features):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "prediction_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    assert scores.shape == (10, 489)
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
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    # Check prediction
    prediction = pd.read_csv(tmp_path / output_subdir / "prediction.csv")
    print(prediction)

    assert prediction["Prediction"].tolist() == ["Cancer"] * 6
    assert np.round(prediction["P(Cancer)"], decimals=4).tolist() == [0.9932] * 6


def test_predict_with_custom_model_and_roc(
    run_cli, tmp_path, resource_path, lionheart_features
):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "prediction_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    assert scores.shape == (10, 489)
    np.save(sample_dir / "dataset" / "feature_dataset.npy", scores)

    ## Create custom ROC curve:

    command_args = [
        "lionheart",
        "extract_roc",
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

    custom_roc_path = tmp_path / "roc_output" / "ROC_curves.json"

    # Predict

    command_args = [
        "lionheart",
        "predict_sample",
        "--sample_dir",
        tmp_path / "test_sample",
        "--resources_dir",
        resource_path,
        "--custom_roc_paths",
        custom_roc_path,
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
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    # Check prediction
    prediction = pd.read_csv(tmp_path / output_subdir / "prediction.csv")
    print(prediction)

    assert prediction["Prediction"].tolist() == ["Cancer"] * 6
    #assert np.round(prediction["P(Cancer)"], decimals=4).tolist() == [0.9932] * 6
    assert False
