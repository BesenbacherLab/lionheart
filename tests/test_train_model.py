import json
import pandas as pd
from utipy import mk_dir


def test_train_model_two_shared_datasets(run_cli, tmp_path, resource_path):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "model_output"

    command_args = [
        "lionheart",
        "train_model",
        "--dataset_paths",
        resource_path / "shared_features" / "Cristiano" / "feature_dataset.npy",
        resource_path / "shared_features" / "Jiang" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "Cristiano" / "meta_data.csv",
        resource_path / "shared_features" / "Jiang" / "meta_data.csv",
        "--dataset_names",
        "Cristiano 2019",
        "Jiang 2015",
        "--resources_dir",
        resource_path,
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = [
        "predictions.csv",
        "model.joblib",
        "confusion_matrices.json",
        "evaluation_scores.csv",
        "feature_contributions.csv",
        "feature_effects_on_probability.csv",
        "probability_densities.csv",
        "ROC_curves.json",
        "threshold_versions.txt",
        "training_info.json",
        "warnings.csv",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    # Check prediction
    with open(tmp_path / output_subdir / "training_info.json") as f:
        training_info = json.load(f)

    print(training_info)

    expected_training_info = {
        "Task": "Cancer Detection",
        "Modeling Task": "binary_classification",
        "Package Versions": {
            "lionheart": "0.1.0",
            "generalize": "0.1.0",
            "joblib": "1.2.0",
            "sklearn": "1.0.2",
            "Min. Required lionheart": "N/A",
        },
        "Labels": {
            "Labels to Use": ["0_Control(control)", "1_Cancer(cancer)"],
            "Positive Label": 1,
            "New Label Index to New Label": {"0": "Control", "1": "Cancer"},
            "New Label to New Label Index": {"Control": 0, "Cancer": 1},
        },
        "Data": {
            "Shape": [586, 489],
            "Target counts": {"0": 282, "1": 304},
            "Datasets": {
                "Names": [
                    "Cristiano 2019",
                    "Jiang 2015",
                ],
                "Number of Samples": {
                    "Cristiano 2019": 474,
                    "Jiang 2015": 112,
                },
            },
        },
    }

    assert list(expected_training_info.keys()) == list(training_info.keys())

    for key in training_info.keys():
        if key == "Package Versions":
            continue
        assert training_info[key] == expected_training_info[key]

    predictions = pd.read_csv(tmp_path / output_subdir / "predictions.csv")
    predictions.iloc[0, 0] = 0.11913294


def test_train_model_one_shared_dataset(run_cli, tmp_path, resource_path):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "model_output"

    command_args = [
        "lionheart",
        "train_model",
        "--dataset_paths",
        resource_path / "shared_features" / "Cristiano" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "Cristiano" / "meta_data.csv",
        "--dataset_names",
        "Cristiano 2019",
        "--resources_dir",
        resource_path,
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = [
        "predictions.csv",
        "model.joblib",
        "confusion_matrices.json",
        "evaluation_scores.csv",
        "feature_contributions.csv",
        "feature_effects_on_probability.csv",
        "probability_densities.csv",
        "ROC_curves.json",
        "threshold_versions.txt",
        "training_info.json",
        "warnings.csv",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    # Check prediction
    with open(tmp_path / output_subdir / "training_info.json") as f:
        training_info = json.load(f)

    print(training_info)

    expected_training_info = {
        "Task": "Cancer Detection",
        "Modeling Task": "binary_classification",
        "Package Versions": {
            "lionheart": "0.1.0",
            "generalize": "0.1.0",
            "joblib": "1.2.0",
            "sklearn": "1.0.2",
            "Min. Required lionheart": "N/A",
        },
        "Labels": {
            "Labels to Use": ["0_Control(control)", "1_Cancer(cancer)"],
            "Positive Label": 1,
            "New Label Index to New Label": {"0": "Control", "1": "Cancer"},
            "New Label to New Label Index": {"Control": 0, "Cancer": 1},
        },
        "Data": {
            "Shape": [474, 489],
            "Target counts": {"0": 244, "1": 230},
            "Datasets": {
                "Names": [
                    "Cristiano 2019",
                ],
                "Number of Samples": {
                    "Cristiano 2019": 474,
                },
            },
        },
    }

    assert list(expected_training_info.keys()) == list(training_info.keys())

    for key in training_info.keys():
        if key == "Package Versions":
            continue
        assert training_info[key] == expected_training_info[key]

    predictions = pd.read_csv(tmp_path / output_subdir / "predictions.csv")
    predictions.iloc[0, 0] = 0.11913294
