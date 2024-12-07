import numpy as np
import numpy.testing as npt
import pandas as pd


def test_train_customize_validate(run_cli, tmp_path, resource_path, lionheart_features):
    #### Train model ####
    model_output_subdir = "model_output"

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
        output_subdir=model_output_subdir,
    )

    #### Customize thresholds ####

    roc_output_subdir = "roc_output"

    command_args = [
        "lionheart",
        "customize_thresholds",
        "--dataset_paths",
        resource_path / "shared_features" / "GECOCA" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "GECOCA" / "meta_data.csv",
        "--custom_model_dir",
        model_output_subdir,
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=roc_output_subdir,
    )

    # Expected files
    # expected_files = ["ROC_curves.json"]

    #### Validate model ####

    validate_output_subdir = "validate_output"

    command_args = [
        "lionheart",
        "validate",
        "--resources_dir",
        resource_path,
        "--custom_model_dir",
        model_output_subdir,
        "--custom_threshold_dirs",
        roc_output_subdir,
        "--thresholds",
        "max_j",
        "spec_0.99",
        "--use_included_validation",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=validate_output_subdir,
    )

    # Expected files
    # expected_files = ["predictions.csv", "evaluation_scores.csv"]

    eval_scores = pd.read_csv(
        tmp_path / validate_output_subdir / "evaluation_scores.csv"
    )

    pd.set_option("display.max_columns", None)
    print(eval_scores)

    assert eval_scores.loc[:, "Threshold Name"].tolist() == [
        "Max. Youden's J",
        "Specificity ~0.99",
        "Max. Youden's J",
        "Specificity ~0.99",
    ]
    assert eval_scores.loc[:, "ROC Curve"].tolist() == [
        "Average (training data)",
        "Average (training data)",
        "Custom 0",
        "Custom 0",
    ]
    npt.assert_almost_equal(
        eval_scores.loc[:, "Threshold"],
        [0.373910, 0.706941, 0.562187, 0.944799],
        decimal=4,
    )
    assert np.round(eval_scores["AUC"], decimals=4).tolist() == [0.8364] * 4
