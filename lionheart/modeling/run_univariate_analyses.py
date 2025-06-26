import pathlib
from typing import Callable, List, Optional, Union, Dict

from utipy import StepTimer, Messenger, check_messenger, move_column_inplace
from generalize.model import evaluate_univariate_models

from lionheart.modeling.prepare_modeling import prepare_modeling


# TODO: Rename labels to targets (Make it clear when these are class indices / strings!)
# TODO: Make this work with regression
# TODO: Implement non-nested cross-validation.


def run_univariate_analyses(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    task: str,
    feature_name_to_feature_group_path: Union[str, pathlib.Path],
    labels_to_use: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,  # None for 2D
    train_only_datasets: Optional[List[str]] = None,
    merge_datasets: Optional[List[str]] = None,
    weight_loss_by_groups: bool = False,
    weight_per_dataset: bool = False,
    expected_shape: Optional[Dict[int, int]] = None,
    aggregate_by_groups: bool = False,
    k: int = 10,
    standardize_cols: bool = True,
    standardize_rows: bool = False,
    standardize_rows_feature_groups: Optional[List[List[int]]] = None,
    bonferroni_correct: bool = True,
    alpha: float = 0.05,
    feature_set_prefix: str = "Feature Set",
    num_jobs: int = 1,
    seed: Optional[int] = 1,
    exp_name: str = "",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> None:
    """
    Run univariate analyses of features in one or more datasets.

    Used in scripts/univariate_analyses.py.


    TODO: Update docs to univariate analysis (from nested cv)

    Parameters
    ----------
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more datasets. When multiple datasets are specfied,
        leave-one-dataset-out cross-dataset-validation is performed.
        When multiple datasets, pass as a dict mapping dataset name -> dataset path.
        TODO: Add requirements for dataset shape.
    out_path: Union[str, pathlib.Path]
        Path to the directory where the results will be saved.
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path]
        Paths to one or more meta data files (one per dataset path, see `dataset_paths`).
        When multiple datasets, pass as a dict mapping dataset name -> meta data path.
        The meta data should be a .csv file containing 2 columns: {sample id, target}.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression'}.
    labels_to_use: Optional[List[str]], default=None
        The labels to use in classification. When specified, at least two labels/groups
        should be specified (separated by a whitespace). When more than two labels are specified,
        multiclass classification is used. When no labels are specified, all labels are used.
        Combine multiple labels to a single label/group (e.g., cancer <- colon,rectal,prostate)
        by giving a name and the parenthesis-wrapped, comma-separated labels. E.g.
        'cancer(colon,rectal,prostate)'.
    feature_sets: Optional[List[int]], default=None
        List of feature sets to use (only for 3D datasets). Default is to use all available feature sets.
    train_only_datasets: Optional[List[str]], default=None
        List of dataset names to use for training only.
        Only relevant when `dataset_paths` has >1 paths.
        Note: For datasets mentioned in `merge_datasets`, all datasets
        in a group should have the same `train_only` status. I.e. either
        all be listed or not listed in `train_only_datasets`.
    merge_datasets: Optional[List[str]], default=None
        List of named dataset groups that should be merged to a single dataset.
        E.g., `["BestDataset(D1,D2,D3)", "WorstDataset(D4,D5)"]`.
    k: int, default=10
        The number of folds for cross-validation.
        Ignored when `dataset_paths` is a dict with multiple dataset paths.
        In that case, cross-dataset-validation (aka. leave-one-dataset-out) is performed instead
        and each dataset becomes an fold (except those listed in `train_only_datasets`).
    standardize_cols: bool, default=True
        Whether to standardize features before the analyses (and within the cross-validation).
        Features are standardized separately per feature set.
    standardize_rows: bool, default=True
        Whether to standardize rows prior to the analyses
        (and prior to any column standardization).
    standardize_rows_feature_groups
        List of lists with integers per feature group to standardize rows by.
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
    messenger("Preparing to run univariate modeling analysis")

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
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task=task,
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
    task = prepared_modeling_dict["task"]

    # Add to paths
    paths = prepared_modeling_dict["paths"]
    paths.set_path(
        name="univariate_evaluations",
        path=out_path / "univariate_evaluations.csv",
        collection="out_files",
    )
    paths.set_path(
        name="readme",
        path=out_path / "univariate_evaluations.README.txt",
        collection="out_files",
    )

    # Create output directories
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    # Show overview of the paths
    messenger(paths)

    alternative_names = [
        f"{cat}___{seqt}"
        for cat, seqt in zip(
            prepared_modeling_dict["feature_group_names"],
            prepared_modeling_dict["feature_seq"],
        )
    ]

    messenger("Start: Evaluating univariate models")
    with timer.time_step(indent=2, message="Running univariate modeling took:"):
        evaluation, readme_string = evaluate_univariate_models(
            x=prepared_modeling_dict["dataset"],
            y=prepared_modeling_dict["labels"],
            groups=prepared_modeling_dict["groups"],
            task="classification" if "classification" in task.lower() else "regression",
            names=prepared_modeling_dict["feature_names"],
            alternative_names=alternative_names,
            feature_sets=None,  # Already selected by `prepare_modeling()`
            feature_set_prefix=feature_set_prefix,
            alpha=alpha,
            bonferroni_correct=bonferroni_correct,
            standardize_cols=standardize_cols,
            standardize_rows=standardize_rows,
            standardize_rows_feature_groups=standardize_rows_feature_groups,
            aggregate_by_groups=prepared_modeling_dict["aggregate_by_groups"],
            weight_loss_by_groups=prepared_modeling_dict["weight_loss_by_groups"],
            weight_loss_by_class="classification"
            in task.lower(),  # Never part of the scikit-learn model!
            weight_per_split=prepared_modeling_dict["weight_per_dataset"],
            k=k,
            split=prepared_modeling_dict["split"],
            eval_by_split=prepared_modeling_dict["split"] is not None,
            positive_label=prepared_modeling_dict["new_positive_label"],
            y_labels=prepared_modeling_dict["new_label_idx_to_new_label"],
            name_cols=["Cell Type", "Cell Group"],
            num_jobs=num_jobs,
            seed=seed,
            messenger=messenger,
        )

    # Split cell group and seq type
    (cell_groups, seq_types) = zip(
        *[cg.split("___") for cg in evaluation["Cell Group"]]
    )
    # (Over)write columns
    evaluation["Cell Group"] = cell_groups
    evaluation["Seq Type"] = seq_types
    move_column_inplace(evaluation, col="Cell Group", pos=1)
    move_column_inplace(evaluation, col="Seq Type", pos=2)

    # Remove feature set column
    # It's always the LIONHEART scores we evaluate
    if "Feature Set" in evaluation.columns:
        del evaluation["Feature Set"]

    messenger("Start: Saving results")
    with timer.time_step(indent=2):
        # Save the evaluation scores, confusion matrices, etc.
        messenger("Saving evaluations", indent=2)
        evaluation.to_csv(paths["univariate_evaluations"], index=False)
        messenger("Saving README", indent=2)
        with open(paths["readme"], "w") as f:
            f.write(readme_string)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
