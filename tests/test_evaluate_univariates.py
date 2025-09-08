import numpy as np
import pandas as pd
import numpy.testing as npt


def test_evaluate_univariates_shared_only(run_cli, tmp_path, resource_path):
    output_subdir = "univar_output"

    command_args = [
        "lionheart",
        "evaluate_univariates",
        "--resources_dir",
        resource_path,
        "--use_included_features",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = [
        "univariate_evaluations.csv",
        "univariate_evaluations.README.txt",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert expected_file in generated_files, (
            f"Expected file {expected_file} not found."
        )

    pd.set_option("display.max_columns", None)

    # Check prediction
    evals = pd.read_csv(tmp_path / output_subdir / "univariate_evaluations.csv")

    # print(evals)

    assert len(evals) == 898

    npt.assert_almost_equal(
        evals.loc[:, "AUC"].max(),
        0.72596,
        decimal=2,
    )

    npt.assert_almost_equal(
        evals.loc[0:3, "num_tests"],
        [898] * 4,
        decimal=2,
    )

    npt.assert_equal(np.unique(evals.loc[:, "Seq Type"], return_counts=True)[1], [355, 544])


def test_evaluate_univariates_single_shared(run_cli, tmp_path, resource_path):
    output_subdir = "univar_output"

    command_args = [
        "lionheart",
        "evaluate_univariates",
        "--resources_dir",
        resource_path,
        "--dataset_paths",
        resource_path / "shared_features" / "Cristiano" / "feature_dataset.npy",
        "--meta_data_paths",
        resource_path / "shared_features" / "Cristiano" / "meta_data.csv",
        "--dataset_names",
        "Cristiano 2019",
    ]
    generated_files, output_dir = run_cli(
        command_args=command_args,
        tmp_path=tmp_path,
        output_subdir=output_subdir,
    )

    # Expected files
    expected_files = [
        "univariate_evaluations.csv",
        "univariate_evaluations.README.txt",
    ]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert expected_file in generated_files, (
            f"Expected file {expected_file} not found."
        )

    pd.set_option("display.max_columns", None)

    # Check prediction
    evals = pd.read_csv(tmp_path / output_subdir / "univariate_evaluations.csv")

    # print(evals)

    assert len(evals) == 898

    npt.assert_almost_equal(
        evals.loc[:, "AUC"].max(),
        0.80564,
        decimal=4,
    )

    npt.assert_almost_equal(
        evals.loc[0:3, "num_tests"],
        [898] * 4,
        decimal=2,
    )

    npt.assert_equal(np.unique(evals.loc[:, "Seq Type"], return_counts=True)[1], [355, 544])
