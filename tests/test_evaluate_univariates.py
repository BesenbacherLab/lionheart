import json
import pandas as pd
import numpy.testing as npt
from utipy import mk_dir

from lionheart.utils.global_vars import LABELS_TO_USE


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
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    pd.set_option("display.max_columns", None)

    # Check prediction
    evals = pd.read_csv(tmp_path / output_subdir / "univariate_evaluations.csv")

    print(evals)

    assert False
    
    npt.assert_almost_equal(
        evals.loc[:, "Threshold"],
        [0.373910, 0.706941, 0.562187, 0.944799],
        decimal=4,
    )


    