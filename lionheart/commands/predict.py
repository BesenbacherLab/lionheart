"""
Script that applies the model to the features of a singe sample and returns the probability of cancer.

"""

from typing import Dict
import logging
import pathlib
import warnings
import json
import joblib
from joblib import load as joblib_load
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from packaging import version
from utipy import Messenger, StepTimer, IOPaths
from generalize.dataset import assert_shape
from generalize.evaluate.roc_curves import ROCCurves, ROCCurve
from generalize.evaluate.probability_densities import ProbabilityDensities
from lionheart.modeling.run_predict_single_model import run_predict_single_model
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import parse_thresholds, Examples
from lionheart.utils.global_vars import INCLUDED_MODELS, ENABLE_SUBTYPING
from lionheart import __version__ as lionheart_version

if not ENABLE_SUBTYPING:
    INCLUDED_MODELS = [m for m in INCLUDED_MODELS if "subtype" not in m]


def setup_parser(parser):
    parser.add_argument(
        "--sample_dir",
        required=True,
        type=str,
        help="Path to directory for sample specified as `--out_dir` during feature extraction."
        "\nShould contain the `dataset` sub folder with the `feature_dataset.npy` files.",
    )
    parser.add_argument(
        "--resources_dir",
        required=True,
        type=str,
        help="Path to directory with framework resources such as the trained model.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory to store the output at. "
        "\nThis directory should be exclusive to the current sample. "
        "\nIt may be within the `--sample_dir`. "
        "\nWhen not supplied, the predictions are stored in `--sample_dir`."
        "\nA `log` directory will be placed in the same directory.",
    )
    models_string = "', '".join(INCLUDED_MODELS + ["none"])
    parser.add_argument(
        "--model_names",
        choices=INCLUDED_MODELS + ["none"],
        default=[INCLUDED_MODELS[0]],  # Newest model should be first in the list
        type=str,
        nargs="*",
        help="Name(s) of included trained model(s) to run. "
        "\nSet to `none` to only use a custom model (see --custom_model_dir)."
        "\nOne of {"
        f"'{models_string}'"
        "}.",
    )
    parser.add_argument(
        "--custom_model_dirs",
        type=str,
        nargs="*",
        help="Path(s) to a directory with a custom model to use. "
        "\nThe directory must include the files `model.joblib` and `ROC_curves.json`."
        "\nThe directory name will be used to identify the predictions in the `model` column of the output.",
    )
    parser.add_argument(
        "--custom_threshold_dirs",
        type=str,
        nargs="*",
        help="Path(s) to a directory with `ROC_curves.json` and `probability_densities.csv` "
        "files made with `lionheart customize_thresholds` "
        "for extracting the probability thresholds."
        "\nThe output will have predictions for thresholds "
        "based on each of the available ROC curves and probability densities "
        "from the training data, the custom models, and these directories."
        + (
            "\n<b>NOTE></b>: These are ignored for subtyping models."
            if ENABLE_SUBTYPING
            else ""
        ),
    )
    threshold_defaults = [
        "max_j",
        "spec_0.95",
        "spec_0.99",
        "sens_0.95",
        "sens_0.99",
        "0.5",
    ]
    parser.add_argument(
        "--thresholds",
        type=str,
        nargs="*",
        default=threshold_defaults,
        help="The probability thresholds to use in cancer detection."
        f"\nDefaults to these {len(threshold_defaults)} thresholds:\n  {', '.join(threshold_defaults)}"
        "\n'max_j' is the threshold at the max. of Youden's J (`sensitivity + specificity + 1`)."
        "\nPrefix a specificity-based threshold with <b>'spec_'</b>. \n  The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "\nPrefix a sensitivity-based threshold with <b>'sens_'</b>. \n  The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "\nWhen passing specific float thresholds, the nearest threshold "
        "in the ROC curve is used. "
        "\n<b>NOTE</b>: The thresholds are extracted from each of the specified ROC curves,"
        "namely the ROC curve fitted on the model's training data and those in --custom_threshold_dirs."
        + ("\n<b>NOTE></b>: Ignored for subtyping models." if ENABLE_SUBTYPING else ""),
    )
    parser.add_argument(
        "--identifier",
        type=str,
        help="A string to add to the output data frame in an ID column. "
        "E.g., the subject ID. Optional.",
    )
    parser.set_defaults(func=main)


examples = Examples()
examples.add_example(
    description="Simplest example:",
    example="""--sample_dir path/to/subject_1/features
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/predictions""",
)
examples.add_example(
    description="Using a custom model (trained with `lionheart train_model`):",
    example="""--sample_dir path/to/subject_1/features
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/predictions
--custom_model_dirs path/to/model/directory""",
)

examples.add_example(
    description="""Using a custom ROC curve for calculating probability thresholds (created with `lionheart extract_roc`).
This lets you use probability thresholds optimized for your own data.""",
    example="""--sample_dir path/to/subject_1/features
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/predictions
--custom_roc_paths path/to/ROC_curves.json""",
)
examples.add_example(
    description="""Specifying custom probability thresholds for 1) a specificity of ~0.975 and 2) a sensitivity of ~0.8.""",
    example="""--sample_dir path/to/subject_1/features
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/predictions
--thresholds spec_0.975 sens_0.8""",
)
EPILOG = examples.construct()


def main(args):
    sample_dir = pathlib.Path(args.sample_dir)
    out_path = pathlib.Path(args.out_dir) if args.out_dir is not None else sample_dir
    resources_dir = pathlib.Path(args.resources_dir)

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="predict-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running model prediction on a single sample")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    model_name_to_dir = {
        model_name: resources_dir / "models" / model_name
        for model_name in args.model_names
        if model_name != "none"
    }
    if args.custom_model_dirs is not None and args.custom_model_dirs:
        for custom_model_path in args.custom_model_dirs:
            custom_model_path = pathlib.Path(custom_model_path)
            if not custom_model_path.is_dir():
                raise ValueError(
                    "A path in --custom_model_dirs was not a directory: "
                    f"{custom_model_path}"
                )
            model_name = custom_model_path.stem
            if model_name in model_name_to_dir.keys():
                raise ValueError(f"Got a duplicate model name: {model_name}")
            model_name_to_dir[model_name] = custom_model_path

    if not model_name_to_dir:
        raise ValueError(
            "No models where selected. Select one or more models to predict the sample."
        )

    training_info_paths = {
        f"training_info_{model_name}": model_dir / "training_info.json"
        for model_name, model_dir in model_name_to_dir.items()
    }

    model_paths = {
        f"model_{model_name}": model_dir / "model.joblib"
        for model_name, model_dir in model_name_to_dir.items()
    }

    custom_threshold_dirs = {}
    custom_roc_paths = {}
    custom_prob_density_paths = {}
    if args.custom_threshold_dirs is not None and args.custom_threshold_dirs:
        custom_threshold_dirs = {
            f"custom_threshold_dir_{idx}": pathlib.Path(path)
            for idx, path in enumerate(args.custom_threshold_dirs)
        }
        custom_roc_paths = {
            f"custom_roc_curve_{idx}": pathlib.Path(path) / "ROC_curves.json"
            for idx, path in enumerate(args.custom_threshold_dirs)
        }
        custom_prob_density_paths = {
            f"custom_prob_densities_{idx}": pathlib.Path(path)
            / "probability_densities.csv"
            for idx, path in enumerate(args.custom_threshold_dirs)
        }

    paths = IOPaths(
        in_files={
            "features": sample_dir / "dataset" / "feature_dataset.npy",
            **model_paths,
            **custom_roc_paths,
            **custom_prob_density_paths,
            **training_info_paths,
        },
        in_dirs={
            "resources_dir": resources_dir,
            "dataset_dir": sample_dir / "dataset",
            "sample_dir": sample_dir,
            **model_name_to_dir,
            **custom_threshold_dirs,
        },
        out_dirs={
            "out_path": out_path,
        },
        out_files={
            "prediction_path": out_path / "prediction.csv",
            "readme_path": out_path / "README.txt",
        },
    )

    messenger("Start: Loading training info", indent=4)
    model_name_to_training_info = {
        model_name: _load_json(paths[f"training_info_{model_name}"])
        for model_name in model_name_to_dir.keys()
    }

    training_roc_paths = {
        f"roc_curve_{model_name}": model_dir / "ROC_curves.json"
        for model_name, model_dir in model_name_to_dir.items()
        if model_name_to_training_info[model_name]["Modeling Task"]
        == "binary_classification"
    }
    if training_roc_paths:
        paths.set_paths(training_roc_paths, collection="in_files")

    training_probability_densities_paths = {
        f"prob_densities_{model_name}": model_dir / "probability_densities.csv"
        for model_name, model_dir in model_name_to_dir.items()
        if model_name_to_training_info[model_name]["Modeling Task"]
        == "binary_classification"
    }
    if training_probability_densities_paths:
        paths.set_paths(training_probability_densities_paths, collection="in_files")

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Interpreting `--thresholds`")
    thresholds_to_calculate = parse_thresholds(args.thresholds)

    messenger("Start: Loading features")
    try:
        features = np.load(paths["features"])
    except:
        messenger("Failed to load features.")
        raise

    # Check shape of sample dataset
    # 10 feature sets, 489 cell types
    assert_shape(
        features,
        expected_n_dims=2,
        expected_dim_sizes={0: 10, 1: 489},
        x_name="Loaded features",
    )

    features = np.expand_dims(features, axis=0)
    # Get first feature set (correlations)
    features = features[:, 0, :]

    prediction_dfs = []

    for model_idx, model_name in enumerate(model_name_to_dir.keys()):
        messenger(f"Model: {model_name}")

        prediction_dfs += run_predict_single_model(
            features=features,
            model_name=model_name,
            model_name_to_training_info=model_name_to_training_info,
            custom_roc_paths=custom_roc_paths,
            custom_prob_density_paths=custom_prob_density_paths,
            thresholds_to_calculate=thresholds_to_calculate,
            paths=paths,
            messenger=messenger,
            timer=timer,
            model_idx=model_idx,
        )

    # Combine data frames and clean it up a bit
    all_predictions_df = pd.concat(prediction_dfs, axis=0, ignore_index=True)

    # Reorder columns
    prob_columns = [col_ for col_ in all_predictions_df.columns if col_[:2] == "P("]
    first_columns = [
        "Model",
        "Task",
        "Threshold Name",
        "ROC Curve",
        "Prediction",
    ] + prob_columns
    remaining_columns = [
        col_ for col_ in all_predictions_df.columns if col_ not in first_columns
    ]
    all_predictions_df = all_predictions_df.loc[:, first_columns + remaining_columns]

    if args.identifier is not None:
        all_predictions_df["ID"] = args.identifier

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        messenger("Final predictions:")
        messenger(all_predictions_df)

    messenger("Saving predicted probability to disk")
    all_predictions_df.to_csv(paths["prediction_path"], index=False)

    messenger("Writing README to explain output")
    _write_output_explanation(all_predictions_df, paths["readme_path"])

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def _load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def _write_output_explanation(df: pd.DataFrame, path: pathlib.Path) -> None:
    # Define the explanations for each column in your output

    column_explanations = {
        "Model": "Name of the applied model used for predictions.",
        "Task": "The task performed by the model.",
        "Threshold Name": "The name of the threshold (i.e. probability cutoff) used for decision making.",
        "ROC Curve": "Name of the Receiver Operating Characteristic curve used to calculate the probability threshold.",
        "Prediction": "The prediction.",
        "P(Cancer)": "The predicted probability of cancer. From an uncalibrated logistic regression model.",
        "Threshold": "The actual probability cutoff used to determine the predicted class.",
        "Exp. Specificity": "The expected specificity at the probability threshold.",
        "Exp. Sensitivity": "The expected sensitivity at the probability threshold.",
        "Exp. Accuracy for Class at Probability": (
            "The expected accuracy of predicting the specific class at the specific probability."
            "\n    I.e., for all samples with this specific probability (interpolated), what percentage were from the predicted class?"
            "\n    Given a new prediction of the class with this probability, we would expect it to be correct that percentage of the time."
            "\n    Calculated based on probability density estimates from the training data."
            "\n    Informs about the reliability of the class prediction (in addition to the probability)."
        ),
        "ID": "A unique sample identifier.",
    }

    # Write the explanations to the readme file
    with open(path, "w") as file:
        file.write("Explanations of columns in `prediction.csv`\n")
        file.write("===========================================\n\n")
        for column, explanation in column_explanations.items():
            if column in df.columns:
                file.write(f"{column}: {explanation}\n\n")
