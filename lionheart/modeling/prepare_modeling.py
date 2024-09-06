import pathlib
from typing import Callable, List, Optional, Tuple, Union, Dict
import numpy as np
from collections import OrderedDict
import pandas as pd
from utipy import StepTimer, IOPaths, Messenger, check_messenger
from generalize.dataset import (
    load_dataset,
    select_samples,
    parse_labels_to_use,
)
from lionheart.modeling.read_meta_data import read_meta_data

# TODO: Rename labels to targets (Make it clear when these are class indices / strings!)
# TODO: Make this work with regression


def prepare_modeling(
    dataset_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    out_path: Union[str, pathlib.Path],
    meta_data_paths: Union[Dict[str, Union[str, pathlib.Path]], str, pathlib.Path],
    feature_name_to_feature_group_path: Union[str, pathlib.Path],
    task: str,
    # Containing partial model function, grid (hparams), etc.
    model_dict: Optional[dict] = None,
    labels_to_use: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,  # None for 2D
    feature_indices: Optional[List[Union[Tuple[int, int], int]]] = None,
    flatten_feature_sets: Union[bool, str] = "auto",
    train_only_labels: Optional[List[str]] = None,
    train_only_datasets: Optional[List[str]] = None,
    merge_datasets: Optional[Dict[str, List[str]]] = None,
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_per_dataset: bool = False,
    expected_shape: Optional[Dict[int, int]] = None,
    mk_plots_dir: bool = True,
    seed: Optional[int] = 1,
    exp_name: str = "",
    timer: Optional[StepTimer] = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """

    Parameters
    ----------
    ...
    train_only_labels
        List of label names to only use during training.
        Samples with these labels get `train_only(group_id)` wrapped around
        their group ID.
        NOTE: This could lead to dataset leakage when the other labels of
        a dataset are tested on.


    expected_shape
        An optional dict mapping a dimension index to an expected shape.
        A `ValueError` is thrown when the dimension expectations are not met.

    Returns
    -------
    ...

    """
    if "classification" in task and labels_to_use is None:
        # Better to be explicit (`labels_to_use` also specifies positive class)
        raise ValueError("`labels_to_use` must be specified in classification tasks.")
    assert labels_to_use is None or (
        isinstance(labels_to_use, list) and isinstance(labels_to_use[0], str)
    )

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)
    messenger("Preparing modeling process")

    # Create paths container with checks
    out_path = pathlib.Path(out_path)

    if not isinstance(dataset_paths, dict):
        raise TypeError(
            "`dataset_paths` should be a dict mapping `dataset_name->dataset_path`. "
            f"Got {type(dataset_paths)}"
        )
    if not isinstance(meta_data_paths, dict):
        raise TypeError(
            "`meta_data_paths` should be a dict mapping `dataset_name->meta_data_path`. "
            f"Got {type(meta_data_paths)}"
        )
    if sorted(dataset_paths.keys()) != sorted(meta_data_paths.keys()):
        raise ValueError(
            "When `dataset_paths` and `meta_data_paths` are dicts, "
            "they must have the same keys."
        )

    # Extract names of datasets
    dataset_names = list(dataset_paths.keys())

    # Check `merge_datasets` is valid
    if merge_datasets is not None:
        all_merge_datasets = [
            str(item) for li in merge_datasets.values() for item in li
        ]
        if len(set(all_merge_datasets)) != len(all_merge_datasets):
            raise ValueError("`merge_datasets` contained duplicate dataset names.")
        if set(merge_datasets.keys()).intersection(set(dataset_names)):
            raise ValueError(
                "Keys in `merge_datasets` cannot be an existing dataset name."
            )
        if set(all_merge_datasets).difference(set(dataset_names)):
            raise ValueError(
                "`merge_datasets` contained unknown datasets to merge: "
                f"{', '.join(list(set(all_merge_datasets).difference(set(dataset_names))))}."
            )

    # Check `train_only_datasets` is valid
    if train_only_datasets is not None:
        if set(train_only_datasets).difference(set(dataset_names)):
            raise ValueError(
                "`train_only_datasets` contained one or more unknown datasets: "
                f"{', '.join(list(set(train_only_datasets).difference(set(dataset_names))))}."
            )
        if len(train_only_datasets) >= len(dataset_paths):
            raise ValueError("At least one dataset should not be train-only.")
        if len(train_only_datasets) == 0:
            train_only_datasets = None

        # Update `train_only_datasets` with merge groups
        if (
            merge_datasets is not None
            and merge_datasets
            and train_only_datasets is not None
        ):
            rm_from_train_only_datasets = []
            for merge_group_name, merge_group_members in merge_datasets.items():
                if any([d in train_only_datasets for d in merge_group_members]):
                    if not all([d in train_only_datasets for d in merge_group_members]):
                        raise ValueError(
                            "When one merge group member is in `train_only_datasets`, all members must be."
                        )
                    rm_from_train_only_datasets += merge_group_members
                    train_only_datasets.append(merge_group_name)
            train_only_datasets = [
                d for d in train_only_datasets if d not in rm_from_train_only_datasets
            ]

    # Convert to path dicts for the path collection
    dataset_paths = {
        f"{name}_dataset_path": path for name, path in dataset_paths.items()
    }
    meta_data_paths = {
        f"{name}_meta_data_path": path for name, path in meta_data_paths.items()
    }

    # As the insertion order of `dataset_paths` and `meta_data_paths`
    # can differ, it is safest to keep a `name -> path names` mapping
    # for the datasets
    dataset_name_to_info = OrderedDict(
        [
            (
                name,
                {
                    "dataset_path_name": f"{name}_dataset_path",
                    "meta_data_path_name": f"{name}_meta_data_path",
                },
            )
            for name in dataset_names
        ]
    )

    # Create paths collection
    paths = IOPaths(
        in_files={
            **dataset_paths,
            **meta_data_paths,
            "feature_name_to_feature_group_path": feature_name_to_feature_group_path,
        },
        out_dirs={
            "out_path": out_path,
        },
    )

    if mk_plots_dir:
        paths.set_path(
            name="plotting_path",
            path=out_path / "plots",
            collection="out_dirs",
        )

    # Init timestamp handler
    # Note: Does not handle nested timing!
    # When using the messenger as msg_fn, messages are indented properly
    if timer is None:
        timer = StepTimer(msg_fn=messenger, verbose=messenger.verbose)

    # Start timer for runtime
    timer.stamp(name="prepare_modeling() start")

    # Incoming model dict
    if model_dict is not None:
        messenger(f"Incoming `model_dict`:\n{model_dict}")

    # Load dataset
    messenger(
        f"Start: Loading {len(dataset_paths)} "
        f"dataset{'s' if len(dataset_paths) > 1 else ''}"
    )
    with timer.time_step(indent=2):
        # Make index tuples
        if (
            feature_sets is not None
            and len(feature_sets)
            and feature_indices is not None
            and isinstance(feature_sets[0], int)  # Not a tuple
        ):
            feature_indices = [
                (fs, fi) for fs in feature_sets for fi in feature_indices
            ]

        # Either use user specified setting or figure it out ('auto')
        assert isinstance(flatten_feature_sets, (bool, str))
        flatten_feature_sets = (
            flatten_feature_sets
            if isinstance(flatten_feature_sets, bool)
            else (model_dict is None or model_dict["expected_ndim"] <= 2)
        )

        # Load all datasets
        datasets = [
            load_dataset(
                path=paths[dataset_info["dataset_path_name"]],
                indices=feature_indices,
                feature_sets=feature_sets if feature_indices is None else None,
                flatten_feature_sets=flatten_feature_sets,
                name=dataset_name if dataset_name != "unnamed" else None,
                expected_shape=expected_shape,
                messenger=messenger,
            )
            for dataset_name, dataset_info in dataset_name_to_info.items()
        ]

        # Create indices for each dataset for after concatenation
        prev_end_index = 0
        for dset, name in zip(datasets, dataset_name_to_info.keys()):
            dataset_name_to_info[name]["indices"] = np.arange(
                start=prev_end_index,  # Inclusive
                stop=prev_end_index + dset.shape[0],  # Exclusive
                dtype=np.int64,
            )
            prev_end_index += dset.shape[0]
            dataset_name_to_info[name]["length"] = dset.shape[0]
        dataset = np.concatenate(datasets, axis=0)

    # Load meta data
    messenger("Start: Reading meta data")
    with timer.time_step(indent=2, message="Reading meta data took:"):
        labels = []
        sample_ids = []
        groups_tuples = []
        for dataset_name, dataset_info in dataset_name_to_info.items():
            # Read meta data
            # TODO consider renaming labels to targets so it
            # also makes sense in regression
            # NOTE: Multiple datasets can have the same
            # sample ID for different subjects
            (
                _sample_ids,
                _labels,
                _groups,  # May be None
                _sample_id_to_label,
                _label_to_sample_ids,
            ) = read_meta_data(
                paths[dataset_info["meta_data_path_name"]],
                task=task,
                targets_as_str="classification" in task,
                name=dataset_name if dataset_name != "unnamed" else None,
                messenger=messenger,
            )
            if "classification" in task:
                # Make labels lowercase
                _labels = [lab.lower() for lab in _labels]
            dataset_info["labels"] = _labels
            labels += _labels
            sample_ids += _sample_ids
            groups_tuples.append((dataset_name, len(_labels), _groups))

            if dataset_info["length"] != len(_labels):
                raise ValueError(
                    f"{dataset_name + ': ' if dataset_name != 'unnamed' else ''}"
                    f"Found {len(_labels)} labels but dataset "
                    f"has {dataset_info['length']} samples."
                )

        # If any of the datasets have specified groups (or `train_only_labels` is specified)
        # We need all of them to have groups
        if (
            any([gs is not None for *_, gs in groups_tuples])
            or train_only_labels is not None
        ):
            groups = []
            for dset, exp_len, gs in groups_tuples:
                if gs is None:
                    groups += [f"{dset}_{i}" for i in range(exp_len)]
                else:
                    groups += gs
        else:
            groups = None

        # Check labels *pre-collapse*
        num_labels = len(np.unique(labels))
        # Save to enable check for collapsings
        num_labels_pre_collapse = num_labels
        with messenger.indentation(add_indent=2):
            collapse_string = (
                " (without collapsing)" if labels_to_use is not None else ""
            )
            messenger(
                f"Number of total labels{collapse_string}: "
                f"{num_labels_pre_collapse}"
            )

    # Create a "dataset"-like array with the names of the datasets
    # for each sample, so we know this information after sample-selection
    dataset_ids = [
        (dataset_name, idx)
        for dataset_name, dataset_info in dataset_name_to_info.items()
        for idx in dataset_info["indices"]
    ]

    # Ensure it is sorted by its index in the dataset
    dataset_ids = sorted(dataset_ids, key=lambda x: x[1])

    # Convert to an array
    # By setting the dtype to `object`, we don't need to worry
    # about the string size being truncated when setting some
    # labels to the train only string
    dataset_ids = np.array([x[0] for x in dataset_ids], dtype=object)

    # Replace dataset IDs for merged groups
    if merge_datasets is not None:
        messenger("Start: Merging datasets")
        for merge_group_name, merge_group_members in merge_datasets.items():
            for member in merge_group_members:
                dataset_ids[dataset_ids == member] = merge_group_name

    # `select_samples()` requires at least 2D array
    dataset_ids = np.expand_dims(dataset_ids, 1)
    if groups is not None:
        groups = np.expand_dims(groups, 1)
    sample_ids = np.expand_dims(sample_ids, 1)

    # Select the samples to use in the analysis
    # This may include collapsing multiple labels to a single label
    # TODO: The below is not tested with regression or when `labels_to_use` is None!!!
    new_positive_label = None
    new_label_idx_to_new_label = None
    if "classification" in task:
        if labels_to_use is not None:
            messenger("Start: Extracting samples for given labels")
            with timer.time_step(indent=2, message="Extracting samples took:"):
                # Extract the actual labels to use and
                # a map of label collapsings
                # as well as the positive label (class index) after collapsings
                all_labels_to_use, collapse_map, positive_label = parse_labels_to_use(
                    labels_to_use=labels_to_use, unique_labels=list(np.unique(labels))
                )

                # Select the samples of interest
                (
                    [dataset, dataset_ids, groups, sample_ids],
                    labels,  # now 0,1,2,...
                    new_label_idx_to_new_label,
                    new_positive_label,
                ) = select_samples(
                    datasets=[dataset, dataset_ids, groups, sample_ids],
                    labels=labels,
                    labels_to_use=all_labels_to_use,
                    collapse_map=collapse_map,
                    positive_label=positive_label,
                    downsample=False,
                    shuffle=False,
                    rm_prefixed_index=True,
                    seed=seed,
                    messenger=messenger,
                )

                # Check labels *post-collapse*
                num_labels = len(np.unique(labels))

                if num_labels != num_labels_pre_collapse:
                    with messenger.indentation(add_indent=2):
                        messenger(
                            "Number of unique labels (with collapsing): ", num_labels
                        )

        # Check labels are valid in case of binary classification
        if task == "binary_classification" and num_labels != 2:
            ltu_string = ""
            if labels_to_use is not None and num_labels < 2:
                ltu_string = " Perhaps you misspelled the labels in `labels_to_use`?"
            raise ValueError(
                f"Binary classification requires exactly 2 target labels, "
                f"found {num_labels}.{ltu_string}"
            )

    elif "regression" in task:
        # TODO Allow regression to have multiple targets?
        num_labels = 1

    # For train_only datasets, we wrap the
    # dataset name with "train_only()",
    # as nested_cross_validate will recognize this
    if train_only_datasets is not None:
        dataset_ids[np.isin(dataset_ids, train_only_datasets)] = [
            f"train_only({i})"
            for i in dataset_ids[np.isin(dataset_ids, train_only_datasets)]
        ]

    # The train-only labels, we wrap the
    # dataset name with "train_only()",
    # as nested_cross_validate will recognize this
    if train_only_labels is not None:
        groups[np.isin(labels, train_only_labels)] = [
            f"train_only({g})" for g in groups[np.isin(labels, train_only_labels)]
        ]

    # Extract dataset IDs to list
    dataset_ids = dataset_ids.squeeze().tolist()
    if groups is not None:
        groups = groups.squeeze().tolist()

    # Name->Identifier for single-value columns
    identifier_cols_dict = {
        "Model": model_dict["name"] if model_dict is not None else "N/A",
        "Task": task.replace("_", " ").title(),
        "Experiment": exp_name,
        "Seed": seed,
    }

    # Ensure data types work with model
    dataset = dataset.astype(np.float32)
    if "classification" not in task:
        # Regression require 32bit float
        labels = np.asarray(labels, np.float32)

    # Update whether to weight loss per dataset
    weight_per_dataset = bool(weight_per_dataset and len(set(dataset_ids)) > 1)

    # Initialize model
    model = None
    if model_dict is not None:
        messenger("Start: Initializing model")
        model = _init_model(
            model_dict=model_dict,
            # Better to balance classes in model when not doing it per dataset
            balance_classes="classification" in task and not weight_per_dataset,
            seed=seed,
        )
        messenger(f"Updated `model_dict`: \n{model_dict}")
        messenger(f"Model: \n{model}")

    # Specify the (outer) `split`s
    # Either `None` for regular single-dataset cross-validation
    # or the `dataset_ids` for cross-dataset-validation
    split = None
    # If we have more than 1 unique (non-train-only) dataset
    # we perform leave-one-dataset-out cross-validation
    if len(set([d for d in dataset_ids if "train_only" not in d])) > 1:
        split = dataset_ids

    # Feature names and groups
    # for plotting cell type contributions
    feature_name_to_feature_group = pd.read_csv(
        paths["feature_name_to_feature_group_path"]
    )
    feature_names = feature_name_to_feature_group.iloc[:, 1].astype("string")
    feature_group_names = feature_name_to_feature_group.iloc[:, 3].astype("string")

    # Record end timestamp
    timer.stamp(name="prepare_modeling() end")

    return {
        "paths": paths,
        "dataset": dataset,
        "groups": groups,
        "labels": labels,
        "sample_ids": sample_ids,
        "feature_names": feature_names,
        "feature_group_names": feature_group_names,
        "model": model,
        "model_dict": model_dict,
        "new_positive_label": new_positive_label,
        "new_label_idx_to_new_label": new_label_idx_to_new_label,
        "new_label_to_new_label_idx": {
            lab: idx for idx, lab in new_label_idx_to_new_label.items()
        },
        "split": split,
        "dataset_sizes": {
            v: c for v, c in zip(*np.unique(dataset_ids, return_counts=True))
        },
        "label_counts": {v: c for v, c in zip(*np.unique(labels, return_counts=True))},
        "task": task,
        "aggregate_by_groups": bool(groups is not None and aggregate_by_groups),
        "weight_loss_by_groups": bool(groups is not None and weight_loss_by_groups),
        "weight_loss_by_class": bool(
            "classification" in task and weight_per_dataset
        ),  # Otherwise done by model object
        "weight_per_dataset": weight_per_dataset,
        "identifier_cols_dict": identifier_cols_dict,
    }


def _init_model(model_dict, balance_classes, seed):
    expected_keys = set(["expected_ndim", "settings", "model"])
    missing_keys = expected_keys.difference(set(model_dict.keys()))
    if missing_keys:
        raise ValueError(
            f"`model_dict` lacked the following keys: {', '.join(list(missing_keys))}"
        )

    # SKLEARN models
    model_args = {}

    # Add arguments from model_dict
    model_args.update(model_dict["settings"])

    # Replace defaults with user-accessible args
    if seed is not None:
        model_args["random_state"] = seed
    if model_dict.get("max_iter", None) is not None:
        model_args["max_iter"] = model_dict["max_iter"]
    if balance_classes:
        model_args["class_weight"] = "balanced"

    # Initialize model
    model_dict["model"] = model_dict["model"](**model_args)

    return model_dict["model"]
