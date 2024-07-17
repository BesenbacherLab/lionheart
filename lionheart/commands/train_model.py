"""
Command for training a new model on specified features.

"""

import logging
import pathlib
import numpy as np
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths
from sklearn.linear_model import LogisticRegression

from lionheart.modeling.transformers import prepare_transformers_fn
from lionheart.modeling.run_full_modeling import run_full_model_training
from lionheart.modeling.model_dict import create_model_dict
from lionheart.utils.dual_log import setup_logging

"""
Todos

- The "included" features must have meta data for labels and cohort
- The specified "new" features must have meta data for labels and (optionally) cohort
    - Probably should allow specifying multiple cohorts from different files
- Parameters should be fixed, to reproduce paper? Or be settable to allow optimizing? (The latter but don't clutter the API!)
- Describe that when --use_included_features is NOT specified and only one --dataset_paths is specified, within-dataset cv is used for hparams optim
- Figure out train_only edge cases
- Allow calculating thresholds from a validation dataset? Perhaps that is a separate script? 
    Then in predict() we can have an optional arg for setting custom path to a roc curve object?
"""


def setup_parser(parser):
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="*",
        help="Path(s) to `feature_dataset.npy` file(s) containing the collected features. "
        "\nExpects shape <i>(?, 10, 489)</i> (i.e., <i># samples, # feature sets, # features</i>). "
        "\nOnly the first feature set is used.",
    )
    parser.add_argument(
        "--meta_data_paths",
        type=str,
        nargs="*",
        help="Path(s) to csv file(s) where the:"
        "\n  1) the first column contains the <b>sample IDs</b>"
        "\n  2) the second column contains the <b>label</b> (one of {<i>'control', 'cancer', 'exclude'</i>})"
        "\n  3) the (optional) third column contains <b>subject ID</b> "
        "(for when subjects have more than one sample)"
        "\nWhen --dataset_paths has multiple paths, there must be "
        "one meta data path per dataset, in the same order."
        "\nSamples with the <i>'exclude'</i> label are excluded from the training.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the collected features at. "
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--use_included_features",
        action="store_true",
        help="Whether to use the included features in the model training."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        help="Path to directory with framework resources, such as the included features. "
        "\nRequired when --use_included_features is specified.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of folds in <i>within-dataset</i> cross-validation for tuning hyperparameters via grid search."
        "\n<u><b>Ignored</b></u> when multiple test datasets are specified, as leave-one-dataset-out cross-validation is used instead.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30000,
        help="Maximum number of iterations used to train the model.",
    )
    parser.add_argument(
        "--train_only",
        type=str,
        nargs="*",
        help="Indices of specified datasets that should only be used for training "
        "during cross-validation for hyperparameter tuning.\n0-indexed so in the range 0->(num_datasets-1)."
        # TODO: Figure out what to do with one test dataset and n train-only datasets?
        "\nWhen --use_included_features is NOT specified, at least one dataset cannot be train-only."
        # TODO: Should we allow setting included features to train-only?
        "\nWHEN TO USE: If you have a dataset with only one of the classes (controls or cancer) "
        "\nwe cannot test on the dataset during cross-validation. It may still be a great addition"
        "\nto the training data, so flag it as 'train-only'.",
    )
    parser.add_argument(
        "--pca_target_variance",
        type=float,
        default=[0.994, 0.995, 0.996, 0.997, 0.998],
        nargs="*",
        help="Target(s) for the explained variance of selected principal components. Used to select the most-explaining components."
        "\nWhen multiple targets are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--lasso_c",
        type=float,
        default=np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]),
        nargs="*",
        help="Inverse Lasso regularization strength value(s) for `sklearn.linear_model.LogisticRegression`."
        "\nWhen multiple values are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate <i>predictions</i> per subject before evaluations. "
        "The predicted probabilities averaged per group."
        "\nOnly the evaluations are affected by this. "
        "\n<u><b>Ignored</b></u> when no subjects are present in the meta data.",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="Number of available CPU cores to use in parallelization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random state supplied to `sklearn.linear_model.LogisticRegression`.",
    )
    parser.set_defaults(func=main)


EPILOG = (
    """<h1>Examples:</h1>

Simple example using defaults:

"""
    + """<b>$ %(prog)s</b>
--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory
""".replace("\n", " ")
    + """

Train a model on a single dataset. This uses within-dataset cross-validation for hyperparameter optimization:

"""
    + """<b>$ %(prog)s</b>
--dataset_paths path/to/dataset/feature_dataset.npy
--meta_data_paths path/to/dataset/meta_data.csv
--out_dir path/to/output/directory
""".replace("\n", " ")
)


def main(args):
    out_path = pathlib.Path(args.out_dir)

    # Create output directory
    paths = IOPaths(out_dirs={"out_path": out_path})
    paths.mk_output_dirs(collection="out_dirs")

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="collect_samples-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running training of model")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    if len(args.meta_data_paths) != len(args.dataset_paths):
        raise ValueError(
            "`--meta_data_paths` and `--dataset_paths` did not "
            "have the same number of paths."
        )

    dataset_paths = {}
    meta_data_paths = {}
    for path_idx, dataset_path in enumerate(args.dataset_paths):
        nm = f"new_dataset_{path_idx}"
        dataset_paths[nm] = dataset_path
        meta_data_paths[nm] = args.meta_data_paths[path_idx]

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
        if args.resources_dir is None:
            raise ValueError(
                "When `--use_included_features` is specified, "
                "`--resources_dir` must be specified as well."
            )

        shared_features_dir = pathlib.Path(args.resources_dir) / "shared_features"
        shared_features_paths = pd.read_csv(shared_features_dir / "dataset_paths.csv")

        # Extract dataset paths
        shared_features_dataset_paths = {
            nm: path
            for nm, path in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths["Dataset Path"],
            )
        }

        # Extract meta data paths
        shared_features_meta_data_paths = {
            nm: path
            for nm, path in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths["Meta Data Path"],
            )
        }

        # Extract train-only status
        shared_features_train_only_flag = {
            nm: t_o
            for nm, t_o in zip(
                shared_features_paths["Dataset Name"],
                shared_features_paths["Train Only"],
            )
        }

        # Add new paths and settings to user's specificationss
        dataset_paths.update(shared_features_dataset_paths)
        meta_data_paths.update(shared_features_meta_data_paths)
        train_only += [nm for nm, t_o in shared_features_train_only_flag.items() if t_o]

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
    )

    transformers_fn = prepare_transformers_fn(
        pca_target_variance=args.pca_target_variance,
        min_var_thresh=0.0,
        scale_rows=["mean", "std"],
        standardize=True,
    )

    run_full_model_training(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        task="binary_classification",
        model_dict=model_dict,
        feature_sets=[0],
        train_only_datasets=train_only,
        k=args.k,
        transformers=transformers_fn,
        aggregate_by_groups=args.aggregate_by_subjects,
        weight_by_groups=True,
        weight_per_dataset=True,
        expected_shape={1: 10, 2: 489},  # 10 feature sets, 489 cell types
        num_jobs=args.num_jobs,
        seed=args.seed,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
