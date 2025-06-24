import pathlib
from typing import Callable, List, Optional, Tuple, Union, Dict
import warnings
import random
import contextlib
import pandas as pd
import numpy as np
from utipy import StepTimer, Messenger, check_messenger, random_alphanumeric
from generalize import Evaluator, nested_cross_validate

from lionheart.modeling.prepare_modeling_command import parse_merge_datasets
from lionheart.plotting.plot_inner_scores import plot_inner_scores
from lionheart.modeling.prepare_modeling import prepare_modeling

# TODO: Rename labels to targets (Make it clear when these are class indices / strings!)
# TODO: Make this work with regression
# TODO: Implement non-nested cross-validation.
# TODO: Add requirements for dataset shape in dataset_paths arg

def run_nested_cross_validation(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    feature_name_to_feature_group_path: Union[str, pathlib.Path],
    task: str,
    # Containing partial model function, is_skorch, grid (hparams), etc.
    model_dict: dict,
    labels_to_use: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,  # None for 2D
    feature_indices: Optional[List[Union[Tuple[int, int], int]]] = None,
    train_only_datasets: Optional[List[str]] = None,
    train_only_labels: Optional[List[str]] = None,
    merge_datasets: Optional[Dict[str, List[str]]] = None,
    k_outer: int = 10,
    k_inner: Optional[int] = 10,
    reps: int = 1,
    transformers: Optional[Union[List[tuple], Callable]] = None,
    train_test_transformers: List[str] = [],
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_per_dataset: bool = False,
    expected_shape: Optional[Dict[int, int]] = None,
    inner_metric: Optional[str] = None,
    refit: Union[bool, str, Callable] = True,
    num_jobs: int = 1,
    seed: Optional[int] = 1,
    exp_name: str = "",
    timer: StepTimer = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> None:
    """
    Run cross-validation on one or more datasets.

    Parameters
    ----------
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more datasets. When multiple datasets are specified (and they are
        not merged to 1 dataset), leave-one-dataset-out cross-validation is performed.
        When multiple datasets, pass as a dict mapping dataset name -> dataset path.
    out_path: Union[str, pathlib.Path]
        Path to the directory where the results will be saved.
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more meta data files (one per dataset path, see `dataset_paths`).
        When multiple datasets, pass as a dict mapping dataset name -> meta data path.
        The meta data should be a .csv file containing 2 or 3 columns:
        {sample id, target, group (optional)}.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression',
            'leave_one_class_out_binary_classification'}.
        'leave_one_class_out_binary_classification':
            Cross-validate binary classification on multiclass data where each positive multiclass label
            is held out with a proportionate number of negative samples. So the model is binary
            but multiclasses are the outer splits. All datasets are pooled to one dataset.
            Sampled negative samples for the splits are thus from the global pool of negative samples,
            not from the specific dataset.
    model_dict: dict
        A dictionary containing the partial model function, boolean is_skorch (if using Skorch),
        the grid of hyperparameters, and any additional parameters for the model.
        TODO: Please check this is true and improve this arg doc.
    labels_to_use: Optional[List[str]], default=None
        The labels to use in classification. When specified, at least two labels/groups
        should be specified (separated by a whitespace). When more than two labels are specified,
        multiclass classification is used. When no labels are specified, all labels are used.
        Combine multiple labels to a single label/group (e.g., cancer <- colon,rectal,prostate)
        by giving a name and the parenthesis-wrapped, comma-separated labels. E.g.
        'cancer(colon,rectal,prostate)'.
    feature_sets: Optional[List[int]], default=None
        List of feature sets to use (only for 3D datasets). Default is to use all available feature sets.
    feature_indices: Optional[List[Union[Tuple[int, int], int]]], default=None
        List of feature indices to use. If 2D dataset, indices should be integers.
        If 3D dataset, tuples with indices of the feature set and the index of the feature.
    train_only_datasets: Optional[List[str]], default=None
        List of dataset names to use for training only.
        Only relevant when `dataset_paths` has >1 paths.
        Note: For datasets mentioned in `merge_datasets`, all datasets
        in a group should have the same `train_only` status. I.e. either
        all be listed or not listed in `train_only_datasets`.
    train_only_labels: Optional[List[str]], default=None
        List of cancer types to use for training only.
        NOTE: Only used when `task=="leave_one_class_out_binary_classification"`.
    merge_datasets:  Optional[Dict[str, List[str]]], default=None
        Dict mapping collapsed dataset name to a list with names of the dataset members.
        List of named dataset groups that should be merged to a single dataset.
        E.g., `["BestDataset(D1,D2,D3)", "WorstDataset(D4,D5)"]`.
    k_outer: int, default=10
        The number of outer folds for nested cross-validation or folds in regular cross-validation.
        Ignored when `dataset_paths` is a dict with multiple dataset paths.
        In that case, cross-dataset-validation (aka. leave-one-dataset-out) is performed instead
        and each dataset becomes an outer fold (except those listed in `train_only_datasets`).
    k_inner: int or None, default=10
        The number of inner folds of the nested cross-validation for hyperparameter tuning.
        When `None` and `dataset_paths` is a dict with multiple dataset paths, the inner
        folds are the datasets present in the training data.
    reps: int, default=1
        The number of repetitions for the cross-validation. In nested cross-validation,
        when the model is *deterministic* only the inner loop sampling differs between
        the repetitions.
    transformers: default=None
        Either a list, or a function to make the list, of tuples with the transformers to be applied to the dataset, in the order they should be applied.
        Each tuple should contain a) the transformer name and b) the initialized transformer instance.
        The transformers are added to the final model pipeline.
        Function:
            For when the transformers rely on information about the loaded dataset.
            When passing a function, it should take the following arguments (in this order):
                class_to_idx_map: Dict[str, int]
                dataset: np.ndarray
                confounders: Optional[np.ndarray]
                TODO: Update this part!
            It should return the list of transformer tuples.
            Note that the dataset and confounders are copies.
    train_test_transformers: str
        Names of transformers with the `training_mode: bool` argument
        to disable during testing.
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
    weight_per_dataset : bool
        Whether to weight training loss (by class and/or group) separately per dataset.
        E.g., when each dataset has bias that shouldn't be associated with the majority class.
        *Ignored* when `dataset_paths` only has 1 path.
    num_jobs: int, default=1
        The number of jobs to use for parallel processing.
        If set to -1, use all available CPUs.
    seed: Optional[int], default=1
        Random state.
        E.g., used for splitting data into folds and for deterministic model initialization.
        Each cross-validation repetition will use `seed`+repetition as seed.
        When the model is a neural network, the seed is not used during model initialization,
        as that would cause all models to have the same initialization (per repetition at least).
        Hence, neural networks are not determistic.
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
    messenger("Preparing to run nested cross-validation")

    if task == "leave_one_class_out_binary_classification":
        if k_inner is None or k_inner < 1:
            raise ValueError(
                f"`--loco`: `k_inner` cannot be `None` and must be positive: Got {k_inner}"
            )
        if merge_datasets is not None:
            messenger(
                "Overwriting `merge_datasets` for leave-one-class-out cross-validation",
                add_msg_fn=warnings.warn,
            )
        merge_datasets = parse_merge_datasets(
            merge_datasets=(
                ["All(" + ",".join(dataset_paths.keys()) + ")"]
                if isinstance(dataset_paths, dict)
                else None
            )
        )

    elif train_only_labels is not None:
        raise ValueError(
            "`train_only_labels` can only be specified when "
            "`task == 'leave_one_class_out_binary_classification'`."
        )

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
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task=task,
        model_dict=model_dict,
        labels_to_use=labels_to_use,
        feature_sets=feature_sets,
        feature_indices=feature_indices,
        train_only_datasets=train_only_datasets,
        train_only_labels=train_only_labels,
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
        name="inner_results_path",
        path=out_path / "inner_results.csv",
        collection="out_files",
    )
    paths.set_path(
        name="best_coefficients_path",
        path=out_path / "best_coefficients.csv",
        collection="out_files",
    )
    paths.set_path(
        name="tmp_path",
        path=out_path / f"tmp_{random_alphanumeric(size=10)}",
        collection="tmp_dirs",
    )
    paths.print_note = "Some output file paths are defined later."

    # Create output directories
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)
    paths.mk_output_dirs(collection="tmp_dirs", messenger=messenger)

    if callable(transformers):
        transformers, model_dict = transformers(model_dict=model_dict)

    labels = prepared_modeling_dict["labels"]
    outer_splits = prepared_modeling_dict["split"]
    y_labels = prepared_modeling_dict["new_label_idx_to_new_label"]
    positive_label = prepared_modeling_dict["new_positive_label"]
    weight_per_split = prepared_modeling_dict["weight_per_dataset"]
    if task == "leave_one_class_out_binary_classification":
        # Update labels to binary labels
        # and create outer split by the multiclass labels
        # with same proportions of negative samples
        # (relative to all positive or negative samples respectively)
        # NOTE: `outer_splits` is a dict of splits
        labels, outer_splits, y_labels = _update_labels_for_leave_one_class_out(
            multiclass_labels=prepared_modeling_dict["labels"],
            label_idx_to_label=prepared_modeling_dict["new_label_idx_to_new_label"],
            negative_label="Control",
            positive_label_name="Cancer",
            reps=reps,
            train_only_labels=None,
        )
        positive_label = 1
        weight_per_split = False
        task = "binary_classification"
        k_outer = None

        # Show fold sizes
        messenger("Fold sizes (different sampling of controls per repetition):\n")
        for split_name, split in outer_splits.items():
            messenger(
                f"{split_name}: {dict(zip(*np.unique(split, return_counts=True)))}"
            )

    # As we need to remove the temporary directories again
    # even when the code fails or is interrupted
    # we put everything in an try/except
    try:
        # Show overview of the paths
        messenger(paths)

        messenger("Start: Cross-validating model on task")
        with timer.time_step(indent=2, message="Running nested cross-validation took:"):
            if inner_metric is None:
                inner_metric = (
                    "balanced_accuracy"
                    if "classification" in task
                    else "neg_mean_squared_error"
                )

            cv_out = nested_cross_validate(
                x=prepared_modeling_dict["dataset"],
                y=labels,
                model=prepared_modeling_dict["model"],
                grid=model_dict["grid"],
                groups=prepared_modeling_dict["groups"],
                positive=positive_label,
                y_labels=y_labels,
                k_outer=k_outer,
                k_inner=k_inner,
                outer_split=outer_splits,
                eval_by_split=outer_splits is not None,
                aggregate_by_groups=prepared_modeling_dict["aggregate_by_groups"],
                weight_loss_by_groups=prepared_modeling_dict["weight_loss_by_groups"],
                weight_loss_by_class=prepared_modeling_dict["weight_loss_by_class"],
                weight_per_split=weight_per_split,
                tmp_path=paths["tmp_path"],
                inner_metric=inner_metric,
                refit=refit,
                task=task,
                transformers=transformers,
                train_test_transformers=train_test_transformers,
                add_channel_dim=model_dict["requires_channel_dim"],
                add_y_singleton_dim=False,
                reps=reps,
                num_jobs=num_jobs,
                seed=seed,
                identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
                eval_idx_colname="Repetition",
                # NOTE: Outer loop (best_estimator_) fit failings always raise an error
                # (use cv_error_score='raise' to not ignore it, I think)
                grid_error_score=np.nan,
                cv_error_score="raise",
                messenger=messenger,
            )

        messenger("Start: Saving results")
        with timer.time_step(indent=2):
            # Save the evaluation scores, confusion matrices, etc.
            messenger("Saving evaluations", indent=2)
            Evaluator.save_evaluations(
                combined_evaluations=cv_out["Evaluation"]["Evaluations"],
                warnings=cv_out["Warnings"],
                out_path=paths["out_path"],
            )

            # Save the summary of the results
            if cv_out["Evaluation"]["Summary"] is not None:
                messenger("Saving evaluation summary", indent=2)
                Evaluator.save_evaluation_summary(
                    evaluation_summary=cv_out["Evaluation"]["Summary"],
                    out_path=paths["out_path"],
                    identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
                )

            # Save the predictions
            if cv_out["Outer Predictions"] is not None:
                test_sample_ids = prepared_modeling_dict["sample_ids"][
                    cv_out["Outer Indices"][0]
                ]
                messenger("Saving outer predictions", indent=2)
                Evaluator.save_predictions(
                    predictions_list=cv_out["Outer Predictions"],
                    targets_list=cv_out["Outer Targets"],
                    groups_list=cv_out["Outer Groups"],
                    sample_ids=test_sample_ids,
                    split_indices_list=cv_out["Outer Splits"],
                    out_path=paths["out_path"],
                    identifier_cols_dict=prepared_modeling_dict["identifier_cols_dict"],
                )

            # Save the inner results
            # These were calculated by scikit-learn
            if cv_out["Inner Results"] is not None:
                messenger("Saving inner results", indent=2)

                # Combine all data frames
                inner_results = pd.concat(
                    cv_out["Inner Results"],
                    keys=range(len(cv_out["Inner Results"])),
                    ignore_index=False,
                ).reset_index(0)

                # Rename first column
                inner_results.rename(
                    {list(inner_results.columns)[0]: "Repetition"},
                    inplace=True,
                    axis=1,  # Columns
                )
                inner_results.to_csv(paths["inner_results_path"], index=False)

            # Save the inner results
            # These were calculated by scikit-learn
            if (
                cv_out["Best Coefficients"] is not None
                and cv_out["Best Coefficients"][0] is not None
            ):
                messenger("Saving best coefficients", indent=2)

                # Combine all data frames
                best_coefficients = pd.concat(
                    cv_out["Best Coefficients"],
                    keys=range(len(cv_out["Best Coefficients"])),
                    ignore_index=False,
                ).reset_index(0)

                # Rename first column
                best_coefficients.rename(
                    {list(inner_results.columns)[0]: "Repetition"},
                    inplace=True,
                    axis=1,  # Columns
                )
                best_coefficients.to_csv(paths["best_coefficients_path"], index=False)

        # Print results
        pd.set_option("display.max_columns", None)
        if cv_out["Evaluation"]["Summary"] is not None:
            messenger(cv_out["Evaluation"]["Summary"]["Scores"])
        else:
            messenger(cv_out["Evaluation"]["Evaluations"]["Scores"])

        messenger("Start: Plotting inner cross-validation scores")
        plot_inner_scores(
            inner_results=inner_results,
            messenger=messenger,
            metric_name=inner_metric,
            save_dir=paths["out_path"],
        )

    except Exception as e:
        # Delete created temporary folders
        paths.rm_tmp_dirs(raise_on_fail=False, messenger=messenger)
        messenger(f"Error: {e}")
        raise

    # Remove temporary directories
    paths.rm_tmp_dirs(raise_on_fail=False, messenger=messenger)


# TODO: Add unit tests
def _update_labels_for_leave_one_class_out(
    multiclass_labels: np.ndarray,
    label_idx_to_label: Dict[int, str],
    negative_label: str,
    positive_label_name: str,
    reps: int,
    train_only_labels: Optional[List[str]],
):
    #### Update labels to binary labels ####
    multiclass_label_to_label_idx = {
        lab: idx for idx, lab in label_idx_to_label.items()
    }
    binary_labels = np.array(
        [
            int(lab != multiclass_label_to_label_idx[negative_label])
            for lab in multiclass_labels
        ]
    )
    binary_new_label_idx_to_new_label = {0: negative_label, 1: positive_label_name}

    #### Create label splits ####
    label_to_label_counts = dict(zip(*np.unique(multiclass_labels, return_counts=True)))
    positive_label_counts = {
        lab: counts
        for lab, counts in label_to_label_counts.items()
        if lab != multiclass_label_to_label_idx[negative_label]
        and (
            train_only_labels is None
            or label_idx_to_label[lab] not in train_only_labels
        )
    }
    num_positives = sum(positive_label_counts.values())
    assert num_positives > 0, "Found no samples with the positive label"

    positive_label_proportions = {
        lab: count / num_positives for lab, count in positive_label_counts.items()
    }
    negative_indices = _get_indices_of(
        multiclass_labels, multiclass_label_to_label_idx[negative_label]
    )
    assert len(negative_indices) > 0, "Found no samples with the negative label"

    outer_split = {}
    for rep in range(reps):
        # Split indices of negative samples by the proportions of the positive labels
        negative_sample_split = _split_indices(
            indices=negative_indices,  # TODO: Currently includes `train_only_datasets` indices
            proportions=positive_label_proportions,
            seed=rep,  # TODO: It would be better to run each repetition with a separate split seed
            copy=True,
        )
        assert len(negative_sample_split.keys()) == len(
            positive_label_proportions.keys()
        )

        label_to_indices = {
            "split__" + label_idx_to_label[lab]: indices
            + list(_get_indices_of(multiclass_labels, lab))
            for lab, indices in negative_sample_split.items()
        }

        if train_only_labels is not None:
            # Add indices for train-only labels
            label_to_indices.update(
                {
                    "train_only(" + "split__" + lab + ")": list(
                        _get_indices_of(
                            multiclass_labels, multiclass_label_to_label_idx[lab]
                        )
                    )
                    for lab in train_only_labels
                }
            )

        # Invert mapping of labels and indices
        index_to_label = {
            idx: lab for lab, indices in label_to_indices.items() for idx in indices
        }
        outer_split[f"sampling_{rep}"] = [
            index_to_label[i] for i in range(len(binary_labels))
        ]

    return binary_labels, outer_split, binary_new_label_idx_to_new_label


def _get_indices_of(labels, lab):
    return np.nonzero(np.asarray(labels) == lab)[0]


def _split_indices(
    indices, proportions: Dict[str, float], seed: int = 1, copy=True
) -> Dict[str, List[int]]:
    """
    For splitting indices of samples from the negative class
    by a dict of proportions.

    *Naïve* greedy splitter of indices into groups based on specified proportions.

    NOTE: Given *enough indices*, the following could be a simpler implementation:
    (Note: it may not draw from all classes with small datasets though!)
    ```
    np.random.choice(
        list(proportions.keys()),
        size=len(indices),
        replace=True,
        p=list(proportions.values())
    )
    ```
    """
    if copy:
        indices = indices.copy()
        proportions = proportions.copy()

    # Shuffle indices
    with _temp_numpy_seed(seed):
        np.random.shuffle(indices)

    num_indices = len(indices)

    # Randomize order of proportions dict
    proportions_tuple = list(proportions.items())
    rng = random.Random(seed)
    rng.shuffle(proportions_tuple)

    # Split indices somewhat greedily
    splits = {}
    for key, proportion in proportions_tuple:
        elements_left = len(indices)
        current_num = int(np.floor(num_indices * proportion))
        if current_num > elements_left:
            current_num = elements_left
        splits[key], indices = list(indices[:current_num]), indices[current_num:]
    if len(indices) > 0:
        splits[key] = splits[key] + list(indices)

    return splits


@contextlib.contextmanager
def _temp_numpy_seed(seed: Optional[int]):
    """
    Set temporary `numpy` seed or (when `seed=None`)
    use it to reset random state when exiting context.
    """
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
