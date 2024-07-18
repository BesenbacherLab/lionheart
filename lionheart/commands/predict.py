"""
Script that applies the model to the features of a singe sample and returns the probability of cancer.

"""

from typing import Dict, List, Union
import logging
import pathlib
import numpy as np
import pandas as pd
import joblib
import warnings
from joblib import load as joblib_load
from utipy import Messenger, StepTimer, IOPaths, move_column_inplace
from generalize.dataset import assert_shape
from lionheart.modeling.roc_curves import ROCCurves
from lionheart.utils.dual_log import setup_logging


joblib_dump_version = "1.4.0"


# TODO Allow using a custom model!


def parse_thresholds(thresholds: List[str]) -> Dict[str, Union[bool, List[float]]]:
    """
    Parse the threshold names.
    """
    thresh_dict = {
        "max_j": False,
        "sensitivity": [],
        "specificity": [],
        "numerics": [],
    }

    for thresh_name in thresholds:
        if thresh_name == "max_j":
            thresh_dict["max_j"] = True
        elif thresh_name[:4] == "sens":
            thresh_dict["sensitivity"].append(float(thresh_name.split("_")[1]))
        elif thresh_name[:4] == "spec":
            thresh_dict["specificity"].append(float(thresh_name.split("_")[1]))
        elif thresh_name.replace(".", "").isnumeric():
            thresh_dict["numerics"].append(float(thresh_name))
        else:
            raise ValueError(f"Could not parse passed threshold: {thresh_name}")
    return thresh_dict


def setup_parser(parser):
    parser.add_argument(
        "--sample_dir",
        required=True,
        type=str,
        help="Path to directory for sample specified as `--out_dir` during feature extraction."
        "Should contain the `dataset` sub folder with the `feature_dataset.npy` files.",
    )
    parser.add_argument(
        "--resources_dir",
        required=True,
        type=str,
        help="Path to directory with framework resources, such as the trained model.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to directory to store the output at. "
        "This directory should be exclusive to the current sample. "
        "It may be within the `--sample_dir`. "
        "When not supplied, the predictions are stored in `--sample_dir`."
        "A `log` directory will be placed in the same directory.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        nargs="*",
        default=["max_j", "spec_0.95", "spec_0.99", "sens_0.95", "sens_0.99", "0.5"],
        help="The probability thresholds to use. "
        "'max_j' is the threshold at max. of Youden's J (`sensitivity + specificity + 1`). "
        "Prefix a specificity-based threshold with 'spec_'. The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "Prefix a sensitivity-based threshold with 'sens_'. The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "When passing specific float thresholds, the nearest threshold "
        "in the ROC curve is used. "
        "Note: The thresholds are taken from the included ROC curve, "
        "which was fitted to the *training* data during model training.",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        help="A string to add to the output data frame in an ID column. "
        "E.g. the subject ID. Optional.",
    )
    parser.set_defaults(func=main)


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

    model_name = "CRUK_CUHK_C2i_D1_D2_DL_DLV_PRSTT"

    paths = IOPaths(
        in_files={
            "features": sample_dir / "dataset" / "feature_dataset.npy",
            "model": resources_dir / "models" / model_name / "model.joblib",
            "roc_curves": resources_dir / "models" / model_name / "ROC_curves.json",
        },
        in_dirs={
            "resources_dir": resources_dir,
            "dataset_dir": sample_dir / "dataset",
            "models_dir": resources_dir / "models",
            "full_model_dir": resources_dir / "models" / model_name,
            "sample_dir": sample_dir,
        },
        out_dirs={
            "out_path": out_path,
        },
        out_files={"prediction_path": out_path / "prediction.csv"},
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Interpreting `--thresholds`")
    thresholds_to_calculate = parse_thresholds(args.thresholds)

    messenger("Start: Loading ROC Curve and calculating thresholds")
    with timer.time_step(indent=4, name_prefix="threshold_calculation"):
        with messenger.indentation(add_indent=4):
            # Load roc curve collection
            try:
                rocs = ROCCurves.load(paths["roc_curves"])
            except:
                messenger("Failed to load ROC curve collection")
                raise

            try:
                roc = rocs.get("Average")  # TODO: Fix path
            except:
                messenger("ROCCurves collection did not have the expected ROC curve.")
                raise

            thresholds = []

            if thresholds_to_calculate["max_j"]:
                max_j = roc.get_threshold_at_max_j()
                max_j["Name"] = "Max. Youden's J"
                thresholds.append(max_j)

            for s in thresholds_to_calculate["sensitivity"]:
                thresh = roc.get_threshold_at_sensitivity(above_sensitivity=s)
                thresh["Name"] = f"Sensitivity ~{s}"
                thresholds.append(thresh)

            for s in thresholds_to_calculate["specificity"]:
                thresh = roc.get_threshold_at_specificity(above_specificity=s)
                thresh["Name"] = f"Specificity ~{s}"
                thresholds.append(thresh)

            for t in thresholds_to_calculate["numerics"]:
                thresh = roc.get_nearest_threshold(threshold=t)
                thresh["Name"] = f"Threshold ~{t}"
                thresholds.append(thresh)

            messenger(
                "Calculated the following thresholds: \n", pd.DataFrame(thresholds)
            )

    messenger("Start: Loading and applying model pipeline")
    with timer.time_step(indent=4, name_prefix="model_inference"):
        with messenger.indentation(add_indent=4):
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

            if joblib.__version__ != joblib_dump_version:
                # joblib sometimes can't load objects
                # pickled with a different joblib version
                messenger(
                    f"Model was pickled with joblib=={joblib_dump_version}. "
                    f"The installed version is {joblib.__version__}. "
                    "Model loading *may* fail.",
                    add_msg_fn=warnings.warn,
                )

            try:
                pipeline = joblib_load(paths["model"])
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

            # Calculate predicted classes based on cutoffs
            for thresh_info in thresholds:
                thresh_info["Prediction"] = (
                    "Cancer"
                    if predicted_probability > thresh_info["Threshold"]
                    else "Healthy"
                )

            prediction_df = pd.DataFrame(thresholds)
            prediction_df["ROC Curve"] = "Average (training data)"
            prediction_df["Probability"] = predicted_probability
            prediction_df.columns = [
                "Threshold",
                "Exp. Specificity",
                "Exp. Sensitivity",
                "Name",
                "Prediction",
                "Probability",
            ]
            move_column_inplace(prediction_df, "Name", 0)

            if args.identifier is not None:
                prediction_df["ID"] = args.identifier

            messenger("Saving predicted probability to disk")
            prediction_df.to_csv(paths["prediction_path"], index=False)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
