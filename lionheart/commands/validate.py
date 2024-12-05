"""
Script that validates a model on one or more specified validation datasets.

"""

from typing import Dict
import logging
import pathlib
import numpy as np
import pandas as pd
import joblib
import warnings
from joblib import load as joblib_load
from utipy import Messenger, StepTimer, IOPaths, move_column_inplace
from generalize.dataset import assert_shape
from generalize.evaluate.roc_curves import ROCCurves, ROCCurve
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.global_vars import JOBLIB_VERSION
from lionheart.utils.cli_utils import parse_thresholds, Examples
from lionheart.utils.global_vars import INCLUDED_MODELS
from lionheart.modeling.prepare_modeling_command import prepare_validation_command

# TODO Not implemented
# - Add the figure of sens/spec thresholds in train and test ROCs


def setup_parser(parser):
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to `feature_dataset.npy` file(s) containing the collected features. "
        "\nExpects shape <i>(?, 10, 489)</i> (i.e., <i># samples, # feature sets, # features</i>). "
        "\nOnly the first feature set is used.",
    )
    parser.add_argument(
        "--meta_data_paths",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to csv file(s) where:"
        "\n  1) the first column contains the <b>sample IDs</b>"
        "\n  2) the second column contains the <b>cancer status</b>\n      One of: {<i>'control', 'cancer', 'exclude'</i>}"
        "\n  3) the third column contains the <b>cancer type</b> "
        + (
            (
                "for subtyping (see --subtype)"
                "\n     Either one of:"
                "\n       {<i>'control', 'colorectal cancer', 'bladder cancer', 'prostate cancer',"
                "\n       'lung cancer', 'breast cancer', 'pancreatic cancer', 'ovarian cancer',"
                "\n       'gastric cancer', 'bile duct cancer', 'hepatocellular carcinoma',"
                "\n       'head and neck squamous cell carcinoma', 'nasopharyngeal carcinoma',"
                "\n       'exclude'</i>} (Must match exactly (case-insensitive) when using included features!) "
                "\n     or a custom cancer type."
                "\n     <b>NOTE</b>: When not running subtyping, any character value is fine."
            )
            if False  # ENABLE_SUBTYPING
            else "[NOTE: Not currently used so can be any string value!]."
        )
        + "\n  4) the (optional) fourth column contains the <b>subject ID</b> "
        "(for when subjects have more than one sample)"
        "\nWhen --dataset_paths has multiple paths, there must be "
        "one meta data path per dataset, in the same order."
        "\nSamples with the <i>'exclude'</i> label are excluded from the training.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the validation outputs in."
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--use_included_validation",
        action="store_true",
        help="Whether to use the included validation dataset."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--resources_dir",
        required=True,
        type=str,
        help="Path to directory with framework resources such as the validation dataset.",
    )
    parser.add_argument(
        "--model_name",
        choices=INCLUDED_MODELS,
        type=str,
        help="Name of the included model to validate."
        "\nNOTE: only one of `--model_name` and `--model_dir` can be specified.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to a directory with a custom model to use. "
        "\nThe directory must include the files `model.joblib` and `ROC_curves.json`."
        "\nThe directory name will be used to identify the predictions in the `model` column of the output.",
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
        "\n<b>NOTE</b>: The thresholds are extracted from the included ROC curve,"
        "\nwhich was fitted to the <b>training</b> data during model training."
        + ("\n<b>NOTE></b>: Ignored for subtyping models." if False else ""),
    )
    parser.set_defaults(func=main)


# TODO: Add more examples
examples = Examples()
examples.add_example(
    description="Validate your model on included validation dataset:",
    example="""--out_dir path/to/model_validation
--resources_dir path/to/resource/directory
--model_dir path/to/new_model
--use_included_validation
""",
)
examples.add_example(
    description="Validate included model on your dataset:",
    example=f"""--dataset_paths path/to/dataset_1/feature_dataset.npy 
--meta_data_paths path/to/dataset_1/meta_data.csv 
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory
--model_name {INCLUDED_MODELS[0]}
""",
)
EPILOG = examples.construct()


def main(args):
    sample_dir = pathlib.Path(args.sample_dir)
    out_path = pathlib.Path(args.out_dir) if args.out_dir is not None else sample_dir
    resources_dir = pathlib.Path(args.resources_dir)

    if sum([args.model_name is not None, args.model_dir is not None]) != 1:
        raise ValueError(
            "Exactly one of {`--model_name`, `--model_dir`} "
            "should be specified at a time."
        )

    model_dir = (
        (resources_dir / "models" / args.model_name)
        if args.model_name is not None
        else args.model_dir
    )
    model_name = (
        args.model_name
        if args.model_name is not None
        else pathlib.Path(args.model_dir).name
    )

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="validate-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running model validation")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    paths = IOPaths(
        in_files={
            "model_file": model_dir / "model.joblib",
            "input_roc_curve": model_dir / "ROC_curves.json",
        },
        in_dirs={
            "resources_dir": resources_dir,
            "dataset_dir": sample_dir / "dataset",
            "model_dir": model_dir,
        },
        out_dirs={
            "out_path": out_path,
        },
    )

    dataset_paths, meta_data_paths = prepare_validation_command(
        args=args,
        paths=paths,
        messenger=messenger,
    )

    # Specificy ROC curve paths
    roc_paths = {
        dataset_name + "_roc_curve": out_path / dataset_name / "ROC_curves.json"
        for dataset_name in dataset_paths.keys()
    }

    if len(dataset_paths) > 1:
        # Add combined ROC curve path
        roc_paths["combined_roc_curve"] = out_path / "Combined" / "ROC_curves.json"
        paths.set_path("combined_dir", out_path / "Combined", "out_dirs")

    paths.set_paths(roc_paths, "out_files")

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

    if joblib.__version__ != JOBLIB_VERSION:
        # joblib sometimes can't load objects
        # pickled with a different joblib version
        messenger(
            f"Model was pickled with joblib=={JOBLIB_VERSION}. "
            f"The installed version is {joblib.__version__}. "
            "Model loading *may* fail.",
            add_msg_fn=warnings.warn,
        )

    prediction_dfs = []

    messenger("Start: Loading ROC Curve", indent=4)
    with timer.time_step(indent=8, name_prefix="load_roc_curves"):
        with messenger.indentation(add_indent=8):
            roc_curves: Dict[str, ROCCurve] = {}
            # Load training-data-based ROC curve collection
            try:
                rocs = ROCCurves.load(paths[f"roc_curve_{model_name}"])
            except:
                messenger(
                    "Failed to load ROC curve collection at: "
                    f"{paths[f'roc_curve_{model_name}']}"
                )
                raise

            try:
                roc = rocs.get("Average")  # TODO: Fix path
            except:
                messenger(
                    "`ROCCurves` collection did not have the expected `Average` ROC curve. "
                    f"File: {paths[f'roc_curve_{model_name}']}"
                )
                raise

            roc_curves["Average (training data)"] = roc

            # Load custom ROC curves
            if custom_roc_paths:
                for roc_key in custom_roc_paths.keys():
                    # Load training-data-based ROC curve collection
                    try:
                        rocs = ROCCurves.load(paths[roc_key])
                    except:
                        messenger(
                            "Failed to load ROC curve collection at: "
                            f"{paths[roc_key]}"
                        )
                        raise

                    try:
                        roc = rocs.get("Validation")  # TODO: Fix path
                    except:
                        messenger(
                            "`ROCCurves` collection did not have the expected "
                            f"`Validation` ROC curve. File: {paths[roc_key]}"
                        )
                        raise
                    roc_curves[f"Validation {roc_key.split('_')[-1]}"] = roc

    messenger("Start: Calculating probability threshold(s)", indent=4)
    with timer.time_step(indent=8, name_prefix="threshold_calculation"):
        with messenger.indentation(add_indent=8):
            roc_to_thresholds = {}

            for roc_name, roc_curve in roc_curves.items():
                roc_to_thresholds[roc_name] = []

                if thresholds_to_calculate["max_j"]:
                    max_j = roc_curve.get_threshold_at_max_j()
                    max_j["Name"] = "Max. Youden's J"
                    roc_to_thresholds[roc_name].append(max_j)

                for s in thresholds_to_calculate["sensitivity"]:
                    thresh = roc_curve.get_threshold_at_sensitivity(
                        above_sensitivity=s
                    )
                    thresh["Name"] = f"Sensitivity ~{s}"
                    roc_to_thresholds[roc_name].append(thresh)

                for s in thresholds_to_calculate["specificity"]:
                    thresh = roc_curve.get_threshold_at_specificity(
                        above_specificity=s
                    )
                    thresh["Name"] = f"Specificity ~{s}"
                    roc_to_thresholds[roc_name].append(thresh)

                for t in thresholds_to_calculate["numerics"]:
                    thresh = roc_curve.get_nearest_threshold(threshold=t)
                    thresh["Name"] = f"Threshold ~{t}"
                    roc_to_thresholds[roc_name].append(thresh)

                messenger(f"ROC curve: {roc_name}")
                messenger(
                    "Calculated the following thresholds: \n",
                    pd.DataFrame(roc_to_thresholds[roc_name]),
                    add_indent=4,
                )

    messenger("Start: Loading and applying model pipeline", indent=4)
    with timer.time_step(indent=8, name_prefix="model_inference"):
        with messenger.indentation(add_indent=8):
            try:
                pipeline = joblib_load(paths[f"model_{model_name}"])
                messenger("Pipeline:\n", pipeline)
            except:
                messenger("Model failed to be loaded.")
                raise

            predicted_probability = pipeline.predict_proba(features).flatten()
            if len(predicted_probability) == 1:
                predicted_probability = float(predicted_probability[0])
            elif len(predicted_probability) == 2:
                predicted_probability = float(predicted_probability[1])
            else:
                raise NotImplementedError(
                    f"The predicted probability had the wrong shape: {predicted_probability}. "
                    "Multiclass is not currently supported."
                )
            messenger(f"Predicted probability: {predicted_probability}")

            for roc_name, thresholds in roc_to_thresholds.items():
                # Calculate predicted classes based on cutoffs
                for thresh_info in thresholds:
                    thresh_info["Prediction"] = (
                        "Cancer"
                        if predicted_probability > thresh_info["Threshold"]
                        else "No Cancer"
                    )
                prediction_df = pd.DataFrame(thresholds)

                prediction_df["Probability"] = predicted_probability
                prediction_df.columns = [
                    "Threshold",
                    "Exp. Specificity",
                    "Exp. Sensitivity",
                    "Threshold Name",
                    "Prediction",
                    "Probability",
                ]
                prediction_df["ROC Curve"] = roc_name
                prediction_df["Model"] = model_name
            prediction_dfs.append(prediction_df)

    # Combine data frames and clean it up a bit
    all_predictions_df = pd.concat(prediction_dfs, axis=0, ignore_index=True)
    move_column_inplace(all_predictions_df, "Threshold Name", 0)
    move_column_inplace(all_predictions_df, "ROC Curve", 1)
    move_column_inplace(all_predictions_df, "Model", 0)
    if args.identifier is not None:
        all_predictions_df["ID"] = args.identifier

    messenger("Saving predicted probability to disk")
    all_predictions_df.to_csv(paths["prediction_path"], index=False)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
