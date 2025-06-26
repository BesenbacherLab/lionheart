import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utipy import Messenger, IOPaths
from typing import List, Optional, Dict

from lionheart.modeling.transformers import (
    prepare_benchmark_transformers_fn,
    prepare_transformers_fn,
)
from lionheart.modeling.model_dict import create_model_dict


def prepare_modeling_command(
    args,
    paths: IOPaths,
    messenger: Messenger,
    init_model: bool = True,
    prep_transformers: bool = True,
):
    if not hasattr(args, "subtype"):
        args.subtype = False

    if not hasattr(args, "feature_categories"):
        args.feature_categories = []

    if len(args.meta_data_paths) != len(args.dataset_paths):
        raise ValueError(
            f"`--meta_data_paths` ({len(args.meta_data_paths)}) and "
            f"`--dataset_paths` ({len(args.dataset_paths)}) did not "
            "have the same number of paths."
        )
    if len(args.dataset_paths) == 0 and not args.use_included_features:
        raise ValueError(
            "When `--use_included_features` is not enabled, "
            "at least 1 dataset needs to be specified."
        )
    if args.dataset_names is not None and len(args.dataset_names) != len(
        args.dataset_paths
    ):
        raise ValueError(
            "When specifying `--dataset_names`, it must have one name per dataset "
            "(i.e. same length as `--dataset_paths`)."
        )
    if (
        len(args.dataset_paths) <= 1
        and args.merge_datasets is not None
        and not args.use_included_features
    ):
        raise ValueError(
            "`--merge_datasets` was specified even though only one dataset path was specified."
        )

    dataset_paths = {}
    meta_data_paths = {}
    for path_idx, dataset_path in enumerate(args.dataset_paths):
        nm = f"new_dataset_{path_idx}"
        if args.dataset_names is not None:
            nm = args.dataset_names[path_idx]
        dataset_paths[nm] = dataset_path
        meta_data_paths[nm] = args.meta_data_paths[path_idx]

    merge_datasets = None
    if args.merge_datasets is not None:
        # Get mapping of the dataset mergings
        merge_datasets: Dict[str, List[str]] = parse_merge_datasets(args.merge_datasets)

    messenger(f"Got paths to {len(dataset_paths)} external datasets")

    train_only = []
    if args.train_only:
        if (
            len(args.train_only) == len(args.meta_data_paths)
            and not args.use_included_features
        ):
            raise ValueError(
                "At least one dataset cannot be mentioned in `train_only`."
            )
        if len(args.train_only) > len(args.meta_data_paths):
            raise ValueError(
                "At least one dataset cannot be mentioned in `train_only`."
            )
        for idx in args.train_only:
            if idx > len(dataset_paths):
                raise ValueError(
                    "A dataset index in `--train_only` was greater "
                    f"than the number of specified datasets: {idx}"
                )
        if args.dataset_names is not None:
            train_only = [
                args.dataset_names[train_only_idx] for train_only_idx in args.train_only
            ]
        else:
            train_only = [
                f"new_dataset_{train_only_idx}" for train_only_idx in args.train_only
            ]

    # Add included features
    if args.use_included_features:
        if args.feature_type != "LIONHEART":
            raise ValueError(
                "When `--feature_type` is specified as a benchmark, "
                "`--use_included_features` cannot be enabled."
            )
        shared_features_dir = paths["resources_dir"] / "shared_features"
        shared_features_paths = pd.read_csv(shared_features_dir / "dataset_paths.csv")

        # Remove validation datasets
        shared_features_paths = shared_features_paths.loc[
            ~shared_features_paths.Validation
        ]

        messenger(f"Using {len(shared_features_paths)} included datasets")

        # Extract dataset paths
        shared_features_dataset_paths = {
            nm: shared_features_dir / rel_path
            for nm, rel_path in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths["Dataset Path"],
            )
        }

        # Extract meta data paths
        shared_features_meta_data_paths = {
            nm: shared_features_dir / rel_path
            for nm, rel_path in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths["Meta Data Path"],
            )
        }

        # Extract train-only status
        shared_features_train_only_flag = {
            nm: t_o
            for nm, t_o in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths[
                    f"Train Only {'Subtype' if args.subtype else 'Status'}"
                ],
            )
        }

        # Add new paths and settings to user's specificationss
        dataset_paths.update(shared_features_dataset_paths)
        meta_data_paths.update(shared_features_meta_data_paths)
        train_only += [nm for nm, t_o in shared_features_train_only_flag.items() if t_o]

    feature_name_to_feature_group_path = (
        paths["resources_dir"] / "feature_names_and_grouping.csv"
    )

    model_dict = None
    if init_model:
        model_dict = create_model_dict(
            name="Lasso Logistic Regression",
            model_class=LogisticRegression,
            settings={
                "penalty": "l1",
                "solver": "saga",
                "max_iter": args.max_iter,
                "tol": 0.0001,
                "random_state": args.seed,
            },
            grid={"model__C": np.asarray(args.lasso_c)},
        )

    transformers_fn = None
    if prep_transformers:
        if args.feature_type == "LIONHEART":
            include_indices = None
            if args.feature_categories:
                include_indices = get_category_indices(
                    args, feature_name_to_feature_group_path
                )
            transformers_fn = prepare_transformers_fn(
                pca_target_variance=args.pca_target_variance,
                min_var_thresh=[0.0] if not args.feature_categories else [],
                scale_rows=["mean", "std"],
                standardize=True,
                post_scale_feature_indices=include_indices,
            )
        else:
            if args.feature_categories:
                raise ValueError(
                    "`--feature_categories` selection only works with `--feature_type LIONHEART`"
                )
            transformers_fn = prepare_benchmark_transformers_fn(
                feature_type=args.feature_type,
                pca_target_variance=args.pca_target_variance,
                min_var_thresh=[0.0],
                standardize=True,
            )

    return (
        model_dict,
        transformers_fn,
        dataset_paths,
        train_only,
        merge_datasets,
        meta_data_paths,
        feature_name_to_feature_group_path,
    )


def get_category_indices(args, feature_name_to_feature_group_path) -> List[int]:
    """
    Get indices of categories to include in the analysis.
    """
    category_signs = [cat[0] == "-" for cat in args.feature_categories]
    if not all(x == category_signs[0] for x in category_signs):
        raise ValueError(
            "`--feature_categories`: All listed categories must have "
            "the same sign ('-' or no '-' prefix)."
        )
    exclude = category_signs[0]

    # Load feature to category mapping
    feature_idx_name_category = pd.read_csv(
        feature_name_to_feature_group_path, sep="\t"
    ).iloc[:, :3]
    feature_idx_name_category.columns = ["idx", "cell_type", "category"]
    feature_idx_name_category["category"] = [
        str(cat).lower() for cat in feature_idx_name_category["category"]
    ]
    user_categories = [
        cat[1:].lower() if exclude else cat.lower() for cat in args.feature_categories
    ]
    unique_categories = feature_idx_name_category.category.unique()
    unknown_categories = set(user_categories).difference(set(unique_categories))
    if unknown_categories:
        raise ValueError(
            "One or more specified `--feature_categories` was not recognized: "
            f"{', '.join(list(unknown_categories))}"
        )
    if exclude:
        # Return indices where the category should not be excluded
        return feature_idx_name_category.loc[
            ~feature_idx_name_category.category.isin(user_categories)
        ].idx.to_list()

    # Return indices in the specified categories
    return feature_idx_name_category.loc[
        feature_idx_name_category.category.isin(user_categories)
    ].idx.to_list()


def prepare_validation_command(
    args,
    paths: IOPaths,
    messenger: Messenger,
):
    if not hasattr(args, "subtype"):
        args.subtype = False
    if not hasattr(args, "dataset_names"):
        args.dataset_names = None

    if len(args.meta_data_paths) != len(args.dataset_paths):
        raise ValueError(
            "`--meta_data_paths` and `--dataset_paths` did not "
            "have the same number of paths."
        )

    if args.dataset_names is not None and len(args.dataset_names) != len(
        args.dataset_paths
    ):
        raise ValueError(
            "When specifying `--dataset_names`, it must have one name per dataset "
            "(i.e. same length as `--dataset_paths`)."
        )

    dataset_paths = {}
    meta_data_paths = {}
    for path_idx, dataset_path in enumerate(args.dataset_paths):
        nm = f"new_dataset_{path_idx}"
        if args.dataset_names is not None:
            nm = args.dataset_names[path_idx]
        dataset_paths[nm] = dataset_path
        meta_data_paths[nm] = args.meta_data_paths[path_idx]

    messenger(f"Got paths to {len(dataset_paths)} external datasets")

    # Add included features
    for attr in ["use_included_validation", "use_included_features"]:
        if (
            len(args.dataset_paths) == 0
            and hasattr(args, attr)
            and not getattr(args, attr)
        ):
            raise ValueError(
                f"When `--{attr}` is not enabled, a dataset needs to be specified."
            )

        if hasattr(args, attr) and getattr(args, attr):
            shared_features_dir = paths["resources_dir"] / "shared_features"
            shared_features_paths = pd.read_csv(
                shared_features_dir / "dataset_paths.csv"
            )

            # Get validation datasets
            if attr == "use_included_validation":
                shared_features_paths = shared_features_paths.loc[
                    shared_features_paths.Validation
                ]
            elif attr == "use_included_features":
                shared_features_paths = shared_features_paths.loc[
                    ~shared_features_paths.Validation
                ]

            messenger(f"Using {len(shared_features_paths)} included datasets")

            # Extract dataset paths
            shared_features_dataset_paths = {
                nm: shared_features_dir / rel_path
                for nm, rel_path in zip(
                    shared_features_paths["Dataset Name"],
                    shared_features_paths["Dataset Path"],
                )
            }

            # Extract meta data paths
            shared_features_meta_data_paths = {
                nm: shared_features_dir / rel_path
                for nm, rel_path in zip(
                    shared_features_paths["Dataset Name"],
                    shared_features_paths["Meta Data Path"],
                )
            }

            # Add new paths and settings to user's specificationss
            dataset_paths.update(shared_features_dataset_paths)
            meta_data_paths.update(shared_features_meta_data_paths)

    return (
        dataset_paths,
        meta_data_paths,
    )


def parse_merge_datasets(
    merge_datasets: Optional[List[str]],
) -> Dict[str, List[str]]:
    """
    Parse list of dataset collapsings and create a dict
    mapping new dataset to list of member dataset names.

    Parameters
    ----------
    merge_datasets
        List of strings with dataset collapsings.
        Given as 'new_dataset(dataset1,dataset2,dataset3)'.
        That is, a new name and the parenthesis-wrapped, comma-separated labels.

    Returns
    -------
    dict
        A collapse_map mapping `name->datasets`.
    """
    if merge_datasets is None:
        return None

    # Remove empty strings (not sure this can happen but sanity check)
    merge_datasets = [collapsing for collapsing in merge_datasets if collapsing]

    collapse_map = {}

    # Find collapsings and post-collapse labels
    for group in merge_datasets:
        if "(" not in group or group[-1] != ")":
            raise ValueError(
                "Dataset merge groups must be passed as `name(dataset1,dataset2)`. "
                f"The following was wrongly formatted: {group}."
            )
        else:
            group_name, group = group.split("(")
            if "," in group_name:
                raise ValueError(
                    "Dataset merge groups must be passed as `name(dataset1,dataset2)`. "
                    "Merge group string had a comma prior to '('. "
                )
            group = group[:-1]
            datasets_in_group = group.split(",")
            # Trim leading and trailing whitespaces
            datasets_in_group = [
                dataset_name.strip() for dataset_name in datasets_in_group
            ]
            assert len(datasets_in_group) > 0, (
                f"Found no comma-separated dataset names within the parentheses: {group}."
            )
            collapse_map[group_name] = datasets_in_group

    return collapse_map
