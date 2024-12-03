import numpy as np
import pandas as pd
from utipy import mk_dir
import numpy.testing as npt


def test_cross_validate_three_shared_datasets(run_cli, tmp_path, resource_path):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "model_output"

    command_args = [
        "lionheart",
        "cross_validate",
        "--dataset_paths",
        resource_path / "shared_features" / "Cristiano" / "feature_dataset.npy",
        resource_path / "shared_features" / "Jiang" / "feature_dataset.npy",
        resource_path / "shared_features" / "EndoscopyII" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "Cristiano" / "meta_data.csv",
        resource_path / "shared_features" / "Jiang" / "meta_data.csv",
        resource_path / "shared_features" / "EndoscopyII" / "meta_data.csv",
        "--dataset_names",
        "Cristiano 2019",
        "Jiang 2015",
        "EndoscopyII",
        "--resources_dir",
        resource_path,
        "--k_inner",
        "5",
        "--pca_target_variance",
        "0.996",
        "0.997",
        "--lasso_c",
        "0.04",
        "0.05",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # print(generated_files)

    # Expected files
    expected_files = [
        "warnings.csv",
        "predictions.csv",
        "evaluation_scores.csv",
        "evaluation_summary.csv",
        "confusion_matrices.json",
        "total_confusion_matrices.json",
        "ROC_curves.json",
        "splits_summary.csv",
        "inner_results.csv",
        "best_coefficients.csv",
        "inner_cv_Score_HP_C.png",
        "inner_cv_Score_HP_Target_Variance.png",
        "inner_cv_Time_HP_C.png",
        "inner_cv_Time_HP_Target_Variance.png",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    predictions = pd.read_csv(tmp_path / output_subdir / "predictions.csv")
    print(predictions.iloc[0])
    assert predictions.iloc[0, 0] == 0.525994

    splits_summary = pd.read_csv(tmp_path / output_subdir / "splits_summary.csv")
    print(splits_summary)
    print(splits_summary["AUC"])
    npt.assert_equal(
        np.unique(splits_summary["Fold"]),
        ["Cristiano 2019", "EndoscopyII", "Jiang 2015"],  # alphabetical order
    )
    assert splits_summary.loc[0, "AUC"] == 0.817962
    assert splits_summary.loc[4, "AUC"] == 0.654003


def test_cross_validate_single_shared_datasets(run_cli, tmp_path, resource_path):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "model_output"

    command_args = [
        "lionheart",
        "cross_validate",
        "--dataset_paths",
        resource_path / "shared_features" / "Cristiano" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "Cristiano" / "meta_data.csv",
        "--dataset_names",
        "Cristiano 2019",
        "--resources_dir",
        resource_path,
        "--k_outer",
        "5",
        "--k_inner",
        "5",
        "--pca_target_variance",
        "0.996",
        "0.997",
        "--lasso_c",
        "0.04",
        "0.05",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    print(generated_files)

    # Expected files
    expected_files = [
        "warnings.csv",
        "predictions.csv",
        "evaluation_scores.csv",
        "confusion_matrices.json",
        "ROC_curves.json",
        "inner_results.csv",
        "best_coefficients.csv",
        "inner_cv_Score_HP_C.png",
        "inner_cv_Score_HP_Target_Variance.png",
        "inner_cv_Time_HP_C.png",
        "inner_cv_Time_HP_Target_Variance.png",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    predictions = pd.read_csv(tmp_path / output_subdir / "predictions.csv")
    print(predictions.iloc[0])
    assert predictions.iloc[0, 0] == 0.525994

    # splits_summary = pd.read_csv(tmp_path / output_subdir / "splits_summary.csv")
    # print(splits_summary)
    # print(splits_summary["AUC"])
    # npt.assert_equal(
    #     np.unique(splits_summary["Fold"]),
    #     ["Cristiano 2019", "EndoscopyII", "Jiang 2015"],  # alphabetical order
    # )
    # assert splits_summary.loc[0, "AUC"] == 0.817962
    # assert splits_summary.iloc[4, "AUC"] == 0.654003
