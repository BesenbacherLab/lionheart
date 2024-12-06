import numpy as np
import numpy.testing as npt
import pandas as pd
from utipy import mk_dir

from lionheart.utils.global_vars import INCLUDED_MODELS

# --out_dir path/to/model_validation
# --resources_dir path/to/resource/directory
# --model_dir path/to/new_model
# --use_included_validation
# """,
# )
# examples.add_example(
#     description="Validate included model on your dataset:",
#     example=f"""--dataset_paths path/to/dataset_1/feature_dataset.npy
# --meta_data_paths path/to/dataset_1/meta_data.csv
# --out_dir path/to/output/directory
# --resources_dir path/to/resource/directory
# --model_name {INCLUDED_MODELS[0]}


def test_validate_custom_dataset(run_cli, tmp_path, resource_path, lionheart_features):
    mk_dir(tmp_path / "dataset")
    output_subdir = "validate_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    scores = np.expand_dims(np.array(scores), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    assert scores.shape == (10, 10, 489)
    np.save(tmp_path / "dataset" / "feature_dataset.npy", scores)

    meta = pd.DataFrame(
        {
            "Sample ID": range(10),
            "Cancer Status": ["cancer"] * 5 + ["control"] * 5,
            "Cancer Type": ["cancer"] * 5 + ["control"] * 5,
            "Subject ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    print(meta)
    meta.to_csv(tmp_path / "dataset" / "meta_data.csv", index=False)

    command_args = [
        "lionheart",
        "validate",
        "--dataset_name",
        "'the_dataset'",
        "--dataset_path",
        tmp_path / "dataset" / "feature_dataset.npy",
        "--meta_data_path",
        tmp_path / "dataset" / "meta_data.csv",
        "--resources_dir",
        resource_path,
        "--model_name",
        INCLUDED_MODELS[0],
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


def test_validate_reproducibility(run_cli, tmp_path, resource_path):
    output_subdir = "validate_output"

    command_args = [
        "lionheart",
        "validate",
        "--resources_dir",
        resource_path,
        "--model_name",
        INCLUDED_MODELS[0],
        "--use_included_validation",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = ["predictions.csv", "evaluation_scores.csv"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    pd.set_option("display.max_columns", None)

    # Check prediction
    prediction = pd.read_csv(tmp_path / output_subdir / "predictions.csv")
    print(prediction.iloc[:3])

    assert len(prediction) == 2106

    # Note: loc includes the end index for some reason...
    assert prediction.loc[:2, "Prediction"].tolist() == ["No Cancer"] * 3
    npt.assert_almost_equal(
        prediction.loc[:2, "P(Cancer)"],
        [
            0.371896,
            0.348255,
            0.363712,
        ],
        decimal=4,
    )

    # Check evaluation scores
    eval_scores = pd.read_csv(tmp_path / output_subdir / "evaluation_scores.csv")
    print(eval_scores)

    assert eval_scores.loc[:, "Threshold Name"].tolist() == [
        "Max. Youden's J",
        "Sensitivity ~0.95",
        "Sensitivity ~0.99",
        "Specificity ~0.95",
        "Specificity ~0.99",
        "Threshold ~0.5",
    ]
    npt.assert_almost_equal(
        eval_scores.loc[:, "Threshold"],
        [0.476824, 0.273344, 0.179274, 0.602376, 0.712597, 0.500000],
        decimal=4,
    )
    assert np.round(eval_scores["AUC"], decimals=4).tolist() == [0.858754] * 6


# def test_predict_with_custom_model_and_roc(
#     run_cli, tmp_path, resource_path, lionheart_features
# ):
#     sample_dir = tmp_path / "test_sample"
#     mk_dir(sample_dir / "dataset")
#     output_subdir = "prediction_output"

#     scores = np.expand_dims(np.array(lionheart_features), 0)
#     scores = np.concatenate([scores for _ in range(10)], axis=0)
#     assert scores.shape == (10, 489)
#     np.save(sample_dir / "dataset" / "feature_dataset.npy", scores)

#     ## Create custom ROC curve:

#     command_args = [
#         "lionheart",
#         "customize_thresholds",
#         "--dataset_paths",
#         resource_path / "shared_features" / "GECOCA" / "feature_dataset.npy",
#         "--meta_data_paths",
#         resource_path / "shared_features" / "GECOCA" / "meta_data.csv",
#         "--custom_model_dir",
#         resource_path / "models" / INCLUDED_MODELS[0],
#     ]
#     generated_files, output_dir = run_cli(
#         command_args=command_args,
#         tmp_path=tmp_path,
#         output_subdir="roc_output",
#     )

#     custom_threshold_dir = tmp_path / "roc_output"

#     # Predict

#     command_args = [
#         "lionheart",
#         "predict_sample",
#         "--sample_dir",
#         tmp_path / "test_sample",
#         "--resources_dir",
#         resource_path,
#         "--custom_threshold_dirs",
#         custom_threshold_dir,
#         "--custom_model_dirs",
#         resource_path / "models" / INCLUDED_MODELS[0],
#         "--model_names",
#         "none",
#     ]
#     generated_files, output_dir = run_cli(
#         command_args=command_args,
#         tmp_path=tmp_path,
#         output_subdir=output_subdir,
#     )

#     # Expected files
#     expected_files = ["prediction.csv", "README.txt"]

#     # Check that expected files are generated
#     for expected_file in expected_files:
#         assert (
#             expected_file in generated_files
#         ), f"Expected file {expected_file} not found."

#     # Check prediction
#     prediction = pd.read_csv(tmp_path / output_subdir / "prediction.csv")
#     pd.set_option("display.max_columns", None)
#     print(prediction)

#     assert (
#         prediction["Prediction"].tolist() == ["Cancer"] * 12
#     )  # Also tests size of data frame

#     npt.assert_equal(
#         prediction.loc[[0, 6], "ROC Curve"].tolist(),
#         ["Average (training data)", "Custom 0"],
#     )
#     # Max. J differs between the thresholds
#     npt.assert_almost_equal(
#         prediction.loc[[0, 6], "Threshold"].tolist(),
#         [0.476824, 0.506503],
#         decimal=4,
#     )
#     # Exp. accuracy differs between probability density files
#     npt.assert_almost_equal(
#         prediction.loc[[0, 6], "Exp. Accuracy for Class at Probability"].tolist(),
#         [0.990602, 0.999808],
#         decimal=4,
#     )
