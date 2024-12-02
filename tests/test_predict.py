import numpy as np
import pandas as pd
from utipy import mk_dir


def test_predict(run_cli, tmp_path, resource_path, lionheart_features):
    sample_dir = tmp_path / "test_sample"
    mk_dir(sample_dir / "dataset")
    output_subdir = "prediction_output"

    scores = np.expand_dims(np.array(lionheart_features), 0)
    assert scores.shape == (1, 489)
    np.save(sample_dir / "dataset" / "feature_dataset.npy", scores)

    command_args = [
        "lionheart",
        "predict_sample",
        "--sample_dir",
        tmp_path / "test_sample",
        "--resources_dir",
        resource_path,
    ]
    generated_files, output_dir = run_cli(command_args, tmp_path)

    # Expected files
    expected_files = ["prediction.csv", "README.txt"]

    # Check that expected files are generated
    for expected_file in expected_files:
        assert (
            expected_file in generated_files
        ), f"Expected file {expected_file} not found."

    # Check prediction
    print(pd.read_csv(tmp_path / output_subdir))

    assert False
