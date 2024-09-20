import pathlib
import json
from typing import Callable, List, Optional, Union, Dict
from joblib import dump, __version__ as joblib_version
from sklearn import __version__ as sklearn_version
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from utipy import StepTimer, Messenger, check_messenger
from generalize import Evaluator, train_full_model
from generalize.evaluate.roc_curves import ROCCurves
from generalize.evaluate.probability_densities import ProbabilityDensities
from generalize import __version__ as generalize_version

from lionheart.modeling.prepare_modeling import prepare_modeling
from lionheart.modeling.feature_contribution import FeatureContributionAnalyzer
from lionheart import __version__ as lionheart_version

# TODO: Rename labels to targets (Make it clear when these are class indices / strings!)
# TODO: Make this work with regression
# TODO: Test single-dataset works
# TODO: Write docstring for function


def run_full_model_training(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    feature_name_to_feature_group_path: Union[str, pathlib.Path],
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
    refit_fn: Optional[Callable] = None,
    num_jobs: int = 1,
    seed: Optional[int] = 1,
    required_lionheart_version: Optional[str] = None,
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
    refit_fn
        An optional function for finding the best hyperparameter
        combination from `cv_results_` in grid search.

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
    plot_path = out_path / "plots"

    prepared_modeling_dict = prepare_modeling(
        dataset_paths=dataset_paths,
        out_path=out_path,
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
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

    messenger(
        f"Final dataset sample counts:\n{prepared_modeling_dict['dataset_sizes']}",
        add_indent=2,
    )

    # Unpack parts of the prepared modeling objects
    model_dict = prepared_modeling_dict["model_dict"]
    task = prepared_modeling_dict["task"]

    # Add to paths
    paths = prepared_modeling_dict["paths"]
    paths.set_path("plot_path", plot_path, collection="out_dirs")
    paths.set_paths(
        {
            "model_path": out_path / "model.joblib",
            "training_info": out_path / "training_info.json",
            "feature_contrib_path": out_path / "feature_contributions.csv",
            "feature_effects_path": out_path / "feature_effects_on_probability.csv",
            "plot_feature_contrib_total_path": plot_path
            / "feature_contributions_by_category.total.png",
            "plot_feature_contrib_average_path": plot_path
            / "feature_contributions_by_category.average.png",
            "plot_feature_effects_path": plot_path
            / "feature_effects_on_probability.png",
        },
        collection="out_files",
    )

    paths.print_note = "Some output file paths are defined in generalize::evaluate()."

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
            groups=prepared_modeling_dict["groups"],
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
            refit_fn=refit_fn,
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

    # Extract scores
    scores = train_out["Evaluation"]["Scores"].copy()
    cols_to_move = [
        c for c in ["Split", "Threshold Version", "AUC"] if c in scores.columns
    ]
    col_order = cols_to_move + [c for c in scores.columns if c not in cols_to_move]

    # Move the columns first for the saved evaluations
    train_out["Evaluation"]["Scores"] = scores[col_order]

    # Print results
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        # Remove columns from the messaging to log
        cols_to_delete = [
            "Repetition",
            "Experiment",
            "Task",
            "Model",
            "Seed",
            "Num Classes",
        ]
        col_order = [
            c for c in col_order if c in scores.columns and c not in cols_to_delete
        ]

        messenger(train_out["Evaluation"]["What"], indent=4)
        messenger("\n", scores[col_order])

        messenger("Optimal hyperparameters:", indent=4)
        for key in model_dict["grid"].keys():
            messenger(key, ": ", train_out["Estimator"].get_params()[key], indent=8)

    messenger("Gathering training info:", add_indent=4)
    training_info = {
        "Task": "Cancer Detection"
        if prepared_modeling_dict["task"] == "binary_classification"
        else "Cancer Subtyping",
        "Modeling Task": prepared_modeling_dict["task"],
        "Package Versions": {
            "lionheart": lionheart_version,
            "generalize": generalize_version,
            "joblib": joblib_version,
            "sklearn": sklearn_version,
            "Min. Required lionheart": required_lionheart_version
            if required_lionheart_version is not None
            else "N/A",
        },
        "Labels": {
            "Labels to Use": labels_to_use,
            "Positive Label": prepared_modeling_dict["new_positive_label"],
            "New Label Index to New Label": prepared_modeling_dict[
                "new_label_idx_to_new_label"
            ],
            "New Label to New Label Index": prepared_modeling_dict[
                "new_label_to_new_label_idx"
            ],
        },
        "Data": {
            "Shape": prepared_modeling_dict["dataset"].shape,
            "Target counts": prepared_modeling_dict["label_counts"],
        },
    }
    if isinstance(dataset_paths, dict):
        training_info["Data"]["Datasets"] = {
            "Names": list(dataset_paths.keys()),
            "Number of Samples": prepared_modeling_dict["dataset_sizes"],
        }
    messenger(json.dumps(convert_numpy_types(training_info), indent=4), indent=0)

    messenger("Start: Saving results")
    with timer.time_step(indent=2):
        # Avoid DEBUG messages from matplotlib
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # Save the estimator
        dump(train_out["Estimator"], paths["model_path"])

        # Save training info
        with open(paths["training_info"], "w") as f:
            json.dump(convert_numpy_types(training_info), f)

        # Save the evaluation scores, confusion matrices, etc.
        messenger("Saving evaluation", indent=2)
        Evaluator.save_evaluations(  # TODO
            combined_evaluations=train_out["Evaluation"],
            warnings=train_out["Warnings"],  # TODO list?
            out_path=paths["out_path"],
            identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
        )

        # Save CV optimization results
        cv_results = pd.DataFrame(train_out["CV Results"]).loc[
            :,
            ["rank_test_score", "mean_test_score"]
            + [f"param_{hparam}" for hparam in model_dict["grid"].keys()]
            + ["mean_fit_time", "std_fit_time"],
        ]

        # Note: When no dim wrappers are used, the column will
        # already just be `param_pca__target_variance`
        if "param_pca__kwargs" in cv_results.columns:
            # Get target variance as it's own column
            cv_results["param_pca__target_variance"] = cv_results[
                "param_pca__kwargs"
            ].apply(lambda x: x["target_variance"])
            del cv_results["param_pca__kwargs"]
        cv_results.to_csv(paths["out_path"] / "grid_search_results.csv", index=False)

        # Plot hyperparameters
        if "param_model__C" in cv_results.columns:
            plot_hparams(
                cv_results,
                hparam_col="param_model__C",
                hparam_name="Lasso 'C'",
                plot_path=paths["plot_path"] / "hyperparameter.Lasso.C.png",
            )
        if "param_pca__target_variance" in cv_results.columns:
            plot_hparams(
                cv_results,
                hparam_col="param_pca__target_variance",
                hparam_name="PCA Target Variance",
                plot_path=paths["plot_path"] / "hyperparameter.PCA.target_variance.png",
            )

        # Plot ROC curves
        if "ROC" in train_out["Evaluation"]:
            plot_roc_curves(
                roc_curves=train_out["Evaluation"]["ROC"],
                plot_path=paths["plot_path"] / "ROC_curves.png",
            )

        # Save the predictions
        if train_out["Predictions"] is not None:
            messenger("Saving predictions", indent=2)

            class_idx_to_label_map = None
            positive_label = None
            if "classification" in task:
                class_idx_to_label_map = training_info["Labels"][
                    "New Label Index to New Label"
                ]
                if prepared_modeling_dict["new_positive_label"] is not None:
                    positive_label = class_idx_to_label_map[
                        prepared_modeling_dict["new_positive_label"]
                    ]

            # Combine the predictions to get the
            # data frame to save
            combined_predictions = Evaluator.combine_predictions(
                predictions_list=[train_out["Predictions"]],
                targets=train_out["Targets"],
                groups=train_out["Groups"],
                split_indices_list=[train_out["Split"]],
                target_idx_to_target_label_map=class_idx_to_label_map,
                positive_class=positive_label,
                identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
            )

            # Save the predictions to disk
            Evaluator.save_combined_predictions(
                combined_predictions=combined_predictions, out_path=paths["out_path"]
            )

            if task == "binary_classification":
                messenger("Calculating probability densities", indent=2)
                prob_densities = ProbabilityDensities().calculate_densities(
                    combined_predictions,
                    probability_col="P(Cancer)",
                    target_col="Target Label",
                    group_cols="Split"
                    if "Split" in combined_predictions.columns
                    else None,
                )
                messenger("Saving probability densities", indent=2)
                prob_densities.save(
                    path=paths["out_path"] / "probability_densities.csv"
                )

        if task == "binary_classification":
            messenger("Calculating feature contributions", indent=2)
            with timer.time_step(indent=4):
                feature_contrib_analyser = FeatureContributionAnalyzer(
                    X=prepared_modeling_dict["dataset"],
                    pipeline=train_out["Estimator"],
                    feature_names=prepared_modeling_dict["feature_names"],
                    groups=prepared_modeling_dict["feature_group_names"],
                )
            messenger("Saving feature contributions", indent=2)
            feature_contrib_analyser.save_contributions(
                path=paths["feature_contrib_path"]
            )
            feature_contrib_analyser.save_effects(path=paths["feature_effects_path"])
            feature_contrib_analyser.plot_contributions(
                save_path=paths["plot_feature_contrib_average_path"],
                group_summarizer="mean",
                fig_size=(10, 15),
            )
            feature_contrib_analyser.plot_contributions(
                save_path=paths["plot_feature_contrib_total_path"],
                group_summarizer="sum",
                fig_size=(7, 15),
            )
            feature_contrib_analyser.plot_effects(
                save_path=paths["plot_feature_effects_path"]
            )


def plot_roc_curves(roc_curves: ROCCurves, plot_path: pathlib.Path) -> None:
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
            lw=2 if key in ["Average", "Overall"] else 1,
            alpha=1.0 if key in ["Average", "Overall"] else 0.6,
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


def plot_hparams(
    cv_results: pd.DataFrame, hparam_col: str, hparam_name: str, plot_path: pathlib.Path
) -> None:
    # Plot with seaborn.stripplot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=hparam_col, y="mean_test_score", data=cv_results, jitter=False)
    plt.xlabel(hparam_name)
    plt.ylabel("Balanced Accuracy")
    plt.grid(True)
    plt.savefig(plot_path, dpi=300)
    plt.show()


# Function to convert numpy types to native Python types for json
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {
            convert_numpy_types(key): convert_numpy_types(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
