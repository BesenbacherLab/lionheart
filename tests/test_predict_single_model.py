import numpy as np
import pandas as pd
from utipy import IOPaths, Messenger, StepTimer
from lionheart.modeling.run_predict_single_model import run_predict_single_model
from lionheart.utils.cli_utils import parse_thresholds
from lionheart.utils.global_vars import INCLUDED_MODELS
from lionheart.utils.utils import load_json


def test_predict_single_model(resource_path, lionheart_features):
    scores = np.expand_dims(np.array(lionheart_features), 0)
    scores = np.concatenate([scores for _ in range(10)], axis=0)
    scores = np.expand_dims(np.array(scores), 0)
    assert scores.shape == (1, 10, 489)
    three_scores = np.concatenate([scores for _ in range(3)], axis=0)
    assert three_scores.shape == (3, 10, 489)

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

    thresholds_to_calculate = parse_thresholds(["max_j", "spec_0.99"])

    # Single sample

    predictions_single_sample = run_predict_single_model(
        features=scores,
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

    pd.set_option("display.max_columns", None)
    print(predictions_single_sample)

    # Three samples

    predictions_three_samples = run_predict_single_model(
        features=scores,
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

    print(predictions_three_samples)
    assert False
