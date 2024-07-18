import pathlib
from typing import Callable, List, Optional, Union, Dict
from joblib import dump
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utipy import StepTimer, Messenger, check_messenger
from generalize import Evaluator, train_full_model
from generalize.evaluate.roc_curves import ROCCurves

from lionheart.modeling.prepare_modeling import prepare_modeling

# TODO: Rename labels to targets (Make it clear when these are class indices / strings!)
# TODO: Make this work with regression
# TODO: Test single-dataset works
# TODO: Write docstring for function


def run_full_model_training(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    task: str,
    model_dict: dict,
    labels_to_use: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,  # None for 2D
    train_only_datasets: Optional[List[str]] = None,
    merge_datasets: Optional[Dict[str, List[str]]] = None,
    k: int = 10,
    transformers: Optional[Union[List[tuple], Callable]] = None,
    train_test_transformers: List[str] = [],
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_per_dataset: bool = False,
    expected_shape: Optional[Dict[int, int]] = None,
    num_jobs: int = 1,
    seed: Optional[int] = 1,
    exp_name: str = "",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Fit model to full dataset and test on the training set.
    Finds optimal hyperparameters with grid search (cross-validation).

    Load model with:
    >>> from joblib import load
    >>> clf = load('<out_path>/model.joblib')

    Parameters
    ----------

    aggregate_by_groups : bool
        Whether to aggregate predictions per group, prior to evaluation.
        For regression predictions and predicted probabilities,
        the values are averaged per group.
        For class predictions, we use majority vote. In ties, the
        lowest class index is selected.
        **Ignored** when no groups are present in the meta data.
    weight_loss_by_groups : bool
        Whether to weight samples by their group's size in training loss.
        Each sample in a group gets the weight `1 / group_size`.
        Passed to model's `.fit(sample_weight=)` method.
        **Ignored** when no groups are present in the meta data.


    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)
    messenger("Preparing to run full model training")

    # Init timestamp handler
    # Note: Does not handle nested timing!
    # When using the messenger as msg_fn, messages are indented properly
    timer = StepTimer(msg_fn=messenger, verbose=messenger.verbose)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks
    out_path = pathlib.Path(out_path)

    prepared_modeling_dict = prepare_modeling(
        dataset_paths=dataset_paths,
        out_path=out_path,
        meta_data_paths=meta_data_paths,
        task=task,
        model_dict=model_dict,
        labels_to_use=labels_to_use,
        feature_sets=feature_sets,
        train_only_datasets=train_only_datasets,
        merge_datasets=merge_datasets,
        aggregate_by_groups=aggregate_by_groups,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_per_dataset=weight_per_dataset,
        expected_shape=expected_shape,
        mk_plots_dir=False,
        seed=seed,
        exp_name=exp_name,
        timer=timer,
        messenger=messenger,
    )

    # Unpack parts of the prepared modeling objects
    model_dict = prepared_modeling_dict["model_dict"]
    task = prepared_modeling_dict["task"]

    # Add to paths
    paths = prepared_modeling_dict["paths"]
    paths.set_path(
        name="model_path", path=out_path / "model.joblib", collection="out_files"
    )

    paths.print_note = "Some output file paths are defined in dolearn::evaluate()."

    # Create output directories
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    # Show overview of the paths
    messenger(paths)

    if callable(transformers):
        transformers, model_dict = transformers(model_dict=model_dict)

    messenger("Start: Training full model on task")
    with timer.time_step(indent=2, message="Running model training took:"):
        # Metric to select hyperparameter values by
        metric = (
            "balanced_accuracy"
            if "classification" in task
            else "neg_mean_squared_error"
        )

        train_out = train_full_model(
            x=prepared_modeling_dict["dataset"],
            y=prepared_modeling_dict["labels"],
            model=prepared_modeling_dict["model"],
            grid=model_dict["grid"],
            positive=prepared_modeling_dict["new_positive_label"],
            y_labels=prepared_modeling_dict["new_label_idx_to_new_label"],
            k=k,
            split=prepared_modeling_dict["split"],
            eval_by_split=prepared_modeling_dict["split"] is not None,
            aggregate_by_groups=prepared_modeling_dict["aggregate_by_groups"],
            weight_loss_by_groups=prepared_modeling_dict["weight_loss_by_groups"],
            weight_loss_by_class=prepared_modeling_dict["weight_loss_by_class"],
            weight_per_split=prepared_modeling_dict["weight_per_dataset"],
            metric=metric,
            task=task,
            transformers=transformers,
            train_test_transformers=train_test_transformers,
            add_channel_dim=model_dict["requires_channel_dim"],
            add_y_singleton_dim=False,
            num_jobs=num_jobs,
            seed=seed,
            identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
            # NOTE: Outer loop (best_estimator_) fit failings always raise an error
            grid_error_score=np.nan,
            messenger=messenger,
        )

    # Print results
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        # Extract scores
        scores = train_out["Evaluation"]["Scores"].copy()
        cols_to_move = ["Split", "Threshold Version", "AUC"]
        cols_to_delete = [
            "Repetition",
            "Experiment",
            "Task",
            "Model",
            "Seed",
            "Num Classes",
        ]
        col_order = cols_to_move + [x for x in scores.columns if x not in cols_to_move]
        col_order = [
            x for x in col_order if x in scores.columns and x not in cols_to_delete
        ]

        messenger(train_out["Evaluation"]["What"], indent=4)
        messenger("\n", scores[col_order])

        messenger("Optimal hyperparameters:", indent=4)
        for key in model_dict["grid"].keys():
            messenger(key, ": ", train_out["Estimator"].get_params()[key], indent=8)

    messenger("Start: Saving results")
    with timer.time_step(indent=2):
        # Save the estimator
        dump(train_out["Estimator"], paths["model_path"])

        # Save the evaluation scores, confusion matrices, etc.
        messenger("Saving evaluation", indent=2)
        Evaluator.save_evaluations(  # TODO
            combined_evaluations=train_out["Evaluation"],
            warnings=train_out["Warnings"],  # TODO list?
            out_path=paths["out_path"],
            identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
        )

        # Plot ROC curves
        plot_roc_curves(
            roc_curves=train_out["Evaluation"]["ROC"],
            plot_path=paths["out_path"] / "ROC_curves.png",
        )

        # Save the predictions
        if train_out["Predictions"] is not None:
            messenger("Saving predictions", indent=2)
            Evaluator.save_predictions(
                predictions_list=[train_out["Predictions"]],
                targets=train_out["Targets"],
                groups=train_out["Groups"],
                split_indices_list=[train_out["Split"]],
                out_path=paths["out_path"],
                identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
            )


def plot_roc_curves(roc_curves: ROCCurves, plot_path: pathlib.Path) -> None:
    # Don't get DEBUG messages from matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Plotting with seaborn
    plt.figure(figsize=(10, 8))
    colors = mpl.colormaps["Dark2"].colors
    sns.set(style="whitegrid")

    roc_dict = {
        path.replace("Repetition.0.Split.", ""): roc_curves.get(path)
        for path in roc_curves.paths
    }

    # Plot each individual ROC curve
    for color, (key, roc_) in zip(colors, roc_dict.items()):
        plt.plot(
            roc_.fpr,
            roc_.tpr,
            color=color,
            lw=2 if key == "Average" else 1,
            alpha=1.0 if key == "Average" else 0.6,
            label=f"{key} (AUC = {roc_.auc:.2f})",
        )

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=2)

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)
    plt.title("ROC Curves\n(Predicting *Training* Data)", fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)

    # Save the plot to disk
    plt.savefig(plot_path, dpi=300)
    plt.show()
