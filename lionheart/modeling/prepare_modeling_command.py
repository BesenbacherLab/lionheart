import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utipy import Messenger, IOPaths

from lionheart.modeling.transformers import prepare_transformers_fn
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

    if len(args.meta_data_paths) != len(args.dataset_paths):
        raise ValueError(
            "`--meta_data_paths` and `--dataset_paths` did not "
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

    dataset_paths = {}
    meta_data_paths = {}
    for path_idx, dataset_path in enumerate(args.dataset_paths):
        nm = f"new_dataset_{path_idx}"
        if args.dataset_names is not None:
            nm = args.dataset_names[path_idx]
        dataset_paths[nm] = dataset_path
        meta_data_paths[nm] = args.meta_data_paths[path_idx]

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
        train_only = [
            f"new_dataset_{train_only_idx}" for train_only_idx in args.train_only
        ]

    # Add included features
    if args.use_included_features:
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
        transformers_fn = prepare_transformers_fn(
            pca_target_variance=args.pca_target_variance,
            min_var_thresh=[0.0],
            scale_rows=["mean", "std"],
            standardize=True,
        )

    return (
        model_dict,
        transformers_fn,
        dataset_paths,
        train_only,
        meta_data_paths,
        feature_name_to_feature_group_path,
    )


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
