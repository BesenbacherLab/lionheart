from typing import Callable, Dict, List, Optional, Tuple, Union
import pathlib
import warnings
import json
import joblib
from joblib import load as joblib_load
from sklearn import __version__ as sklearn_version
from packaging import version
from utipy import Messenger, StepTimer, check_messenger
from generalize.evaluate.evaluate import Evaluator
from generalize.evaluate.roc_curves import ROCCurves
from generalize.evaluate.probability_densities import ProbabilityDensities

from lionheart.modeling.prepare_modeling import prepare_modeling
from lionheart import __version__ as lionheart_version
from lionheart.utils.utils import load_json


def run_customize_thresholds(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    model_dir: Union[str, pathlib.Path],
    labels_to_use: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,  # None for 2D
    feature_indices: Optional[List[Union[Tuple[int, int], int]]] = None,
    aggregate_by_groups: bool = False,
    expected_shape: Optional[Dict[int, int]] = None,
    exp_name: str = "",
    timer: StepTimer = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> None:
    """
    Run calculation of the ROC curve on a datasets.


    Parameters
    ----------
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more datasets. When multiple datasets are specified (and they are
        not merged to 1 dataset), leave-one-dataset-out cross-validation is performed.
        When multiple datasets, pass as a dict mapping dataset name -> dataset path.
        TODO: Add requirements for dataset shape.
    out_path: Union[str, pathlib.Path]
        Path to the directory where the results will be saved.
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more meta data files (one per dataset path, see `dataset_paths`).
        When multiple datasets, pass as a dict mapping dataset name -> meta data path.
        The meta data should be a .csv file containing 2 or 3 columns:
        {sample id, target, group (optional)}.
    model_dir: Union[str, pathlib.Path]
        Path to a model directory with a trained model and training info.
    labels_to_use: Optional[List[str]], default=None
        The labels to use in classification. When specified, at least two labels/groups
        should be specified (separated by a whitespace). When more than two labels are specified,
        multiclass classification is used. When no labels are specified, all labels are used.
        Combine multiple labels to a single label/group (e.g., cancer <- colon,rectal,prostate)
        by giving a name and the paranthesis-wrapped, comma-separated labels. E.g.
        'cancer(colon,rectal,prostate)'.
    feature_sets: Optional[List[int]], default=None
        List of feature sets to use (only for 3D datasets). Default is to use all available feature sets.
    feature_indices: Optional[List[Union[Tuple[int, int], int]]], default=None
        List of feature indices to use. If 2D dataset, indices should be integers.
        If 3D dataset, tuples with indices of the feature set and the index of the feature.
    exp_name: str, default=""
        Name of experiment to add to some of the output data frames.
    messenger : `utipy.Messenger` or `None`
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g., `print`)
        and potential indentation.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)
    messenger("Preparing to extract ROC curve")

    # Init timestamp handler
    # When using the messenger as msg_fn, messages are indented properly
    if timer is None:
        timer = StepTimer(msg_fn=messenger, verbose=messenger.verbose)

    # Start timer for total runtime
    timer.stamp("Running cross-validation")

    # Create paths container with checks
    out_path = pathlib.Path(out_path)

    prepared_modeling_dict = prepare_modeling(
        dataset_paths=dataset_paths,
        out_path=out_path,
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=None,
        task="binary_classification",
        labels_to_use=labels_to_use,
        feature_sets=feature_sets,
        feature_indices=feature_indices,
        aggregate_by_groups=aggregate_by_groups,
        expected_shape=expected_shape,
        mk_plots_dir=False,
        exp_name=exp_name,
        timer=timer,
        messenger=messenger,
    )

    paths = prepared_modeling_dict["paths"]
    paths.set_paths(
        {
            "model_path": model_dir / "model.joblib",
            "training_info": model_dir / "training_info.json",
        },
        "in_files",
    )
    paths.set_paths(
        {
            "roc_path": out_path / "ROC_curves.json",
            "prob_densities_path": out_path / "probability_densities.csv",
        },
        collection="out_files",
    )

    # Create output directories
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Checking training info")
    _check_train_info(
        paths,
        messenger,
        timer,
    )

    messenger("Start: Predicting with model")
    with timer.time_step(indent=2, message="Prediction took:"):
        with messenger.indentation(add_indent=8):
            try:
                pipeline = joblib_load(paths["model_path"])
                messenger("Pipeline:\n", pipeline)
            except:
                messenger("Model failed to be loaded.")
                raise

            predicted_probabilities = pipeline.predict_proba(
                prepared_modeling_dict["dataset"]
            )

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

            predicted_probabilities = predicted_probabilities.flatten()

    messenger("Start: Calculating ROC curve")
    with timer.time_step(indent=2, message="ROC curve calculation took:"):
        eval = Evaluator.evaluate(
            targets=prepared_modeling_dict["labels"],
            predictions=predicted_probabilities,
            task="binary_classification",
            groups=prepared_modeling_dict["groups"]
            if prepared_modeling_dict["aggregate_by_groups"]
            else None,
            positive=1,
        )

    messenger("Start: Saving ROC curve")
    rocs = ROCCurves()
    rocs.add(path="Custom ROC", roc_curve=eval["ROC"])
    rocs.save(paths["roc_path"])

    # Calculate probability densities
    combined_predictions = Evaluator.combine_predictions(
        predictions_list=[predicted_probabilities],
        targets=prepared_modeling_dict["labels"],
        groups=prepared_modeling_dict["groups"]
        if prepared_modeling_dict["aggregate_by_groups"]
        else None,
        positive_class="Cancer",
        target_idx_to_target_label_map={0: "Control", 1: "Cancer"},
    )

    messenger("Start: Calculating probability densities")
    prob_densities = ProbabilityDensities().calculate_densities(
        combined_predictions,
        probability_col="P(Cancer)",
        target_col="Target Label",
        group_cols="Split" if "Split" in combined_predictions.columns else None,
    )
    messenger("Start: Saving probability densities")
    prob_densities.save(path=paths["prob_densities_path"])


def _check_train_info(
    paths,
    messenger,
    timer,
):
    messenger("Start: Extracting training info", indent=4)
    with timer.time_step(indent=8, name_prefix="model_training_info"):
        with messenger.indentation(add_indent=8):
            training_info = load_json(paths["training_info"])

            # Check package versioning
            for pkg, present_pkg_version, pkg_verb in [
                ("joblib", joblib.__version__, "pickled"),
                ("sklearn", sklearn_version, "fitted"),
            ]:
                model_pkg_version = training_info["Package Versions"][pkg]
                if present_pkg_version != model_pkg_version:
                    # joblib sometimes can't load objects
                    # pickled with a different joblib version
                    messenger(
                        f"Model was {pkg_verb} with `{pkg}=={model_pkg_version}`. "
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
                    f"Model requires a newer version "
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
