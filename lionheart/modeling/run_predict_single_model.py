import pathlib
from typing import Dict, Optional
import warnings
import joblib
from joblib import load as joblib_load
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from packaging import version
from utipy import Messenger, StepTimer, IOPaths
from generalize.evaluate.roc_curves import ROCCurves, ROCCurve
from generalize.evaluate.probability_densities import ProbabilityDensities
from lionheart.utils.global_vars import INCLUDED_MODELS, ENABLE_SUBTYPING
from lionheart import __version__ as lionheart_version

if not ENABLE_SUBTYPING:
    INCLUDED_MODELS = [m for m in INCLUDED_MODELS if "subtype" not in m]


def run_predict_single_model(
    features: np.ndarray,
    sample_identifiers: Optional[pd.DataFrame],
    model_name: str,
    model_name_to_training_info: Dict[str, dict],
    custom_roc_paths: dict,
    custom_prob_density_paths: dict,
    thresholds_to_calculate: dict,
    paths: IOPaths,
    messenger: Messenger,
    timer: StepTimer,
    model_idx=0,
):
    prediction_dfs = []

    messenger("Start: Extracting training info", indent=4)
    with timer.time_step(indent=8, name_prefix=f"{model_idx}_training_info"):
        with messenger.indentation(add_indent=8):
            # Check package versioning
            training_info = model_name_to_training_info[model_name]
            for pkg, present_pkg_version, pkg_verb in [
                ("joblib", joblib.__version__, "pickled"),
                ("sklearn", sklearn_version, "fitted"),
            ]:
                model_pkg_version = training_info["Package Versions"][pkg]
                if present_pkg_version != model_pkg_version:
                    # joblib sometimes can't load objects
                    # pickled with a different joblib version
                    messenger(
                        f"Model ({model_name}) was {pkg_verb} with `{pkg}=={model_pkg_version}`. "
                        f"The installed version is {present_pkg_version}. "
                        "Using the model *may* fail.",
                        add_msg_fn=warnings.warn,
                    )
            min_lionheart_requirement = training_info["Package Versions"][
                "Min. Required lionheart"
            ]
            if min_lionheart_requirement != "N/A" and version.parse(
                min_lionheart_requirement
            ) > version.parse(lionheart_version):
                raise RuntimeError(
                    f"Model ({model_name}) requires a newer version "
                    f"({min_lionheart_requirement}) of LIONHEART."
                )

            # Whether model is binary or multiclass
            modeling_task = training_info["Modeling Task"]
            cancer_task = training_info["Task"]
            if modeling_task not in [
                "binary_classification",
                "multiclass_classification",
            ]:
                raise ValueError(
                    f"The `training_info.json` 'Modeling Task' was invalid: {modeling_task}"
                )
            messenger(
                f"Modeling task: {cancer_task} ({modeling_task.replace('_', ' ').title()})",
                indent=8,
            )

    if modeling_task == "binary_classification":
        messenger("Start: Loading ROC Curve(s)", indent=4)
        with timer.time_step(indent=8, name_prefix=f"{model_idx}_load_roc_curves"):
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
                            "Failed to load ROC curve collection from: "
                            f"{paths[roc_key]}"
                        )
                        raise

                    try:
                        roc = rocs.get("Custom ROC")
                    except:
                        messenger(
                            "`ROCCurves` collection did not have the expected "
                            f"`Custom ROC` curve. File: {paths[roc_key]}"
                        )
                        raise
                    roc_curves[f"Custom {roc_key.split('_')[-1]}"] = roc

        messenger("Start: Loading Probability Densities", indent=4)
        with timer.time_step(
            indent=8, name_prefix=f"{model_idx}_load_probability_densities"
        ):
            probability_densitites: Dict[str, ProbabilityDensities] = {}

            # Load training-data-based ROC curve collection
            try:
                probability_densitites["Average (training data)"] = (
                    ProbabilityDensities.from_file(
                        paths[f"prob_densities_{model_name}"]
                    )
                )
            except:
                messenger(
                    "Failed to read probability densities file: "
                    f"{paths[f'prob_densities_{model_name}']}"
                )
                raise

            # Load custom probability densities
            if custom_prob_density_paths:
                for prob_key in custom_prob_density_paths.keys():
                    # Load training-data-based ROC curve collection
                    try:
                        probability_densitites[f"Custom {prob_key.split('_')[-1]}"] = (
                            ProbabilityDensities.from_file(paths[prob_key])
                        )
                    except:
                        messenger(
                            "Failed to load probability densities from: "
                            f"{paths[prob_key]}"
                        )
                        raise

        assert sorted(probability_densitites.keys()) == sorted(roc_curves.keys()), (
            "Did not get matching ROC curves and probability density files. "
            f"Keys: [{', '.join(sorted(roc_curves.keys()))}] != "
            f"[{', '.join(sorted(probability_densitites.keys()))}]."
        )

        messenger("Start: Calculating probability threshold(s)", indent=4)
        with timer.time_step(
            indent=8, name_prefix=f"{model_idx}_threshold_calculation"
        ):
            with messenger.indentation(add_indent=8):
                roc_to_thresholds = {}

                for roc_name, roc_curve in roc_curves.items():
                    roc_to_thresholds[roc_name] = []

                    if thresholds_to_calculate["max_j"]:
                        max_j = roc_curve.get_threshold_at_max_j(interpolate=True)
                        max_j["Name"] = "Max. Youden's J"
                        roc_to_thresholds[roc_name].append(max_j)

                    for s in thresholds_to_calculate["sensitivity"]:
                        thresh = roc_curve.get_threshold_at_sensitivity(
                            above_sensitivity=s, interpolate=True
                        )
                        thresh["Name"] = f"Sensitivity ~{s}"
                        roc_to_thresholds[roc_name].append(thresh)

                    for s in thresholds_to_calculate["specificity"]:
                        thresh = roc_curve.get_threshold_at_specificity(
                            above_specificity=s, interpolate=True
                        )
                        thresh["Name"] = f"Specificity ~{s}"
                        roc_to_thresholds[roc_name].append(thresh)

                    for t in thresholds_to_calculate["numerics"]:
                        thresh = roc_curve.get_interpolated_threshold(threshold=t)
                        thresh["Name"] = f"Threshold ~{t}"
                        roc_to_thresholds[roc_name].append(thresh)

                    messenger(f"ROC curve: {roc_name}")
                    messenger(
                        "Calculated the following (interpolated) thresholds: \n",
                        pd.DataFrame(roc_to_thresholds[roc_name]),
                        add_indent=4,
                    )

    messenger("Start: Loading and applying model pipeline", indent=4)
    with timer.time_step(indent=8, name_prefix=f"{model_idx}_model_inference"):
        with messenger.indentation(add_indent=8):
            try:
                pipeline = joblib_load(paths[f"model_{model_name}"])
                messenger("Pipeline:\n", pipeline)
            except:
                messenger("Model failed to be loaded.")
                raise

            # Load and prepare `New Label Index to New Label` mapping
            label_idx_to_label = training_info["Labels"]["New Label Index to New Label"]
            # Ensure keys are integers
            label_idx_to_label = {
                int(key): val for key, val in label_idx_to_label.items()
            }

            if modeling_task == "binary_classification":
                predicted_probabilities = pipeline.predict_proba(features)

                if features.shape[0] == 1:
                    predicted_probabilities = predicted_probabilities.flatten()
                    if len(predicted_probabilities) == 1:
                        predicted_probabilities = [float(predicted_probabilities[0])]
                    elif len(predicted_probabilities) == 2:
                        predicted_probabilities = [float(predicted_probabilities[1])]
                    else:
                        raise NotImplementedError(
                            f"The predicted probability had the wrong shape: {predicted_probabilities}. "
                            f"Model ({model_name}) is expected to be a binary classifier."
                        )
                else:
                    if len(predicted_probabilities.shape) not in [1, 2]:
                        raise NotImplementedError(
                            f"The predicted probability had the wrong shape: {predicted_probabilities.shape}. "
                            "Multiclass is not currently supported."
                        )
                    if (
                        len(predicted_probabilities.shape) == 2
                        and predicted_probabilities.shape[1] > 2
                    ):
                        raise NotImplementedError(
                            f"The predicted probability had the wrong shape: {predicted_probabilities.shape}. "
                            "Multiclass is not currently supported."
                        )

                    if (
                        len(predicted_probabilities.shape) == 2
                        and predicted_probabilities.shape[1] == 2
                    ):
                        # If two columns, get for second column
                        predicted_probabilities = predicted_probabilities[:, 1]

                    predicted_probabilities = predicted_probabilities.flatten().tolist()

                # Get label of predicted class
                positive_label = label_idx_to_label[
                    int(training_info["Labels"]["Positive Label"])
                ]
                probability_colname = f"P({positive_label})"

                if len(predicted_probabilities) == 1:
                    messenger(
                        f"Predicted probability {probability_colname}: "
                        f"{predicted_probabilities}"
                    )

                for roc_name, thresholds in roc_to_thresholds.items():
                    # Calculate predicted classes based on cutoffs
                    if features.shape[0] == 1:
                        # Single sample prediction

                        for thresh_info in thresholds:
                            thresh_info["Prediction"] = (
                                "Cancer"
                                if predicted_probabilities[0] > thresh_info["Threshold"]
                                else "No Cancer"
                            )
                            # Get the expected accuracy for the given prediction
                            # at this probability (based on the training data)
                            thresh_info["Expected Accuracy"] = probability_densitites[
                                roc_name  # Same keys to ensure we use the right densities per ROC
                            ].get_expected_accuracy(
                                new_probability=predicted_probabilities[0]
                            )[
                                "Cancer"
                                if predicted_probabilities[0] > thresh_info["Threshold"]
                                else "Control"  # Label during model training
                            ]

                        prediction_df = pd.DataFrame(thresholds)

                        prediction_df[probability_colname] = predicted_probabilities[0]

                        if sample_identifiers is not None:
                            for col in sample_identifiers.columns:
                                prediction_df[col] = sample_identifiers[col][0]

                    else:
                        # Multi-sample prediction
                        prediction_sets = []
                        for thresh_info in thresholds:
                            pred_set = pd.DataFrame(
                                {
                                    **thresh_info,
                                    "Prediction": [
                                        "Cancer"
                                        if prob > thresh_info["Threshold"]
                                        else "No Cancer"
                                        for prob in predicted_probabilities
                                    ],
                                    "Expected Accuracy": [
                                        probability_densitites[
                                            roc_name  # Same keys to ensure we use the right densities per ROC
                                        ].get_expected_accuracy(new_probability=prob)[
                                            "Cancer"
                                            if prob > thresh_info["Threshold"]
                                            else "Control"  # Label during model training
                                        ]
                                        for prob in predicted_probabilities
                                    ],
                                    probability_colname: predicted_probabilities,
                                }
                            )

                            if sample_identifiers is not None:
                                assert len(pred_set) == len(sample_identifiers)
                                pred_set = pd.concat(
                                    [pred_set, sample_identifiers], axis=1
                                )

                            prediction_sets.append(pred_set)

                        prediction_df = pd.concat(prediction_sets, axis=0).reset_index(
                            drop=True
                        )

                    # NOTE: Assumes insertion order is intact!
                    # TODO: Make more robust!
                    new_cols = [
                        "Threshold",
                        "Exp. Specificity",
                        "Exp. Sensitivity",
                        "Threshold Name",
                        "Prediction",
                        "Exp. Accuracy for Class at Probability",
                        probability_colname,
                    ]

                    if sample_identifiers is not None:
                        new_cols += list(sample_identifiers.columns)

                    prediction_df.columns = new_cols

                    prediction_df["ROC Curve"] = roc_name
                    prediction_df["Model"] = model_name
                    prediction_df["Task"] = cancer_task
                    prediction_dfs.append(prediction_df)

            elif modeling_task == "multiclass_classification":
                # Predict samples
                predicted_probabilities = pipeline.predict_proba(features)
                predictions = pipeline.predict(features).flatten()
                assert len(predictions) == 1
                prediction = label_idx_to_label[predictions[0]]

                messenger(f"Predicted class: {prediction}")

                # Combine to data frame
                prediction_df = pd.DataFrame(
                    predicted_probabilities,
                    columns=[
                        f"P({label_idx_to_label[int(i)]})"
                        for i in sorted(label_idx_to_label.keys(), key=lambda k: int(k))
                    ],
                )
                prediction_df["Prediction"] = prediction
                prediction_df["Model"] = model_name
                prediction_df["Task"] = cancer_task
                prediction_dfs.append(prediction_df)

    return prediction_dfs


def extract_custom_threshold_paths(args):
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

    return custom_threshold_dirs, custom_roc_paths, custom_prob_density_paths
