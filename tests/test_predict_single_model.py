from typing import List
import numpy as np
import numpy.testing as npt
import pandas as pd
from utipy import IOPaths, Messenger, StepTimer
from generalize.evaluate.roc_curves import ROCCurves
from lionheart.modeling.run_predict_single_model import run_predict_single_model
from lionheart.utils.cli_utils import parse_thresholds
from lionheart.utils.global_vars import INCLUDED_MODELS
from lionheart.utils.utils import load_json


def test_predict_single_model(resource_path, lionheart_features):
    scores = np.expand_dims(np.array(lionheart_features), 0)
    assert scores.shape == (1, 898)
    three_scores = np.concatenate([scores for _ in range(3)], axis=0)
    assert three_scores.shape == (3, 898)

    model_name = INCLUDED_MODELS[0]
    model_dir = resource_path / "models" / INCLUDED_MODELS[0]

    paths = IOPaths(
        in_dirs={},
        in_files={
            f"model_{model_name}": model_dir / "model.joblib",
            f"roc_curve_{model_name}": model_dir / "ROC_curves.json",
            f"prob_densities_{model_name}": model_dir / "probability_densities.csv",
            f"training_info_{model_name}": model_dir / "training_info.json",
        },
    )

    model_name_to_training_info = {
        model_name: load_json(paths[f"training_info_{model_name}"])
    }

    threshold_names = ["max_j", "spec_0.99", "0.5"]
    thresholds_to_calculate = parse_thresholds(threshold_names)

    # Single sample

    predictions_list_single_sample = run_predict_single_model(
        features=scores,
        sample_identifiers=None,
        model_name=INCLUDED_MODELS[0],
        model_name_to_training_info=model_name_to_training_info,
        custom_roc_paths={},
        custom_prob_density_paths={},
        thresholds_to_calculate=thresholds_to_calculate,
        paths=paths,
        messenger=Messenger(verbose=False),
        timer=StepTimer(verbose=False),
        model_idx=0,
    )
    predictions_single_sample = pd.concat(predictions_list_single_sample, axis=0)

    pd.set_option("display.max_columns", None)
    print(predictions_single_sample)

    npt.assert_almost_equal(
        predictions_single_sample.loc[:, "P(Cancer)"].tolist(),
        [0.9481] * len(threshold_names),  # 2 x num thresholds
        decimal=4,
    )

    training_roc = ROCCurves.load(paths[f"roc_curve_{model_name}"]).get("Average")
    numeric_threshold = 0.5
    expected_fpr = np.interp(
        numeric_threshold,
        training_roc.thresholds[::-1],
        training_roc.fpr[::-1],
    )
    expected_sensitivity = np.interp(
        numeric_threshold,
        training_roc.thresholds[::-1],
        training_roc.tpr[::-1],
    )
    numeric_threshold_row = predictions_single_sample.loc[
        predictions_single_sample["Threshold Name"] == "Threshold ~0.5"
    ].iloc[0]

    npt.assert_almost_equal(
        numeric_threshold_row["Exp. Specificity"],
        1 - expected_fpr,
        decimal=6,
    )
    npt.assert_almost_equal(
        numeric_threshold_row["Exp. Sensitivity"],
        expected_sensitivity,
        decimal=6,
    )

    # Three samples

    predictions_list_three_samples = run_predict_single_model(
        features=three_scores,
        sample_identifiers=None,
        model_name=INCLUDED_MODELS[0],
        model_name_to_training_info=model_name_to_training_info,
        custom_roc_paths={},
        custom_prob_density_paths={},
        thresholds_to_calculate=thresholds_to_calculate,
        paths=paths,
        messenger=Messenger(verbose=False),
        timer=StepTimer(verbose=False),
        model_idx=0,
    )
    predictions_three_samples = pd.concat(predictions_list_three_samples, axis=0)

    print(predictions_three_samples)

    npt.assert_almost_equal(
        predictions_three_samples.loc[:, "P(Cancer)"].tolist(),
        [0.9481] * 3 * len(threshold_names),
        decimal=4,
    )

    # Comparing single sample and multi sample output

    # Ensure that the colnames are the same

    npt.assert_equal(
        list(predictions_single_sample.columns),
        list(predictions_three_samples.columns),
    )

    # Ensure that the first row of each version is the same

    def round_numerics(l: List[str]) -> List[str]:  # noqa: E741
        def as_numeric(s: str, decimals=5):
            try:
                # Round if numeric
                i = float(s)
                i = np.round(i, decimals=decimals)
                i = str(i)
            except:  # noqa: E722
                i = s
            return i

        return [as_numeric(s) for s in l]

    npt.assert_equal(
        round_numerics(list(predictions_single_sample.loc[0])),
        round_numerics(list(predictions_three_samples.loc[0])),
    )
