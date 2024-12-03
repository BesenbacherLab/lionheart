"""
Script that cross-validates with specified features / cohorts..

"""

import logging
import pathlib
import numpy as np
from utipy import Messenger, StepTimer, IOPaths

from lionheart.modeling.prepare_modeling_command import prepare_modeling_command
from lionheart.modeling.run_cross_validate import run_nested_cross_validation
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples

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
- Ensure Control is the negative label and Cancer is the positive label!
"""


def setup_parser(parser):
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="*",
        help="Path(s) to `feature_dataset.npy` file(s) containing the collected features. ",
    )
    parser.add_argument(
        "--meta_data_paths",
        type=str,
        nargs="*",
        help="Path(s) to csv file(s) where 1) the first column contains the sample IDs, "
        "and 2) the second contains their label, and 3) the (optional) third column contains subject ID "
        "(for when subjects have more than one sample). "
        "When `dataset_paths` has multiple paths, there must be "
        "one meta data path per dataset, in the same order.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the collected features at. "
            "A `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        help="Path to directory with framework resources such as the included features.",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="*",
        help="Names of datasets. <i>Optional</i> but helps interpretability of secondary outputs."
        "\nUse quotes (e.g. 'name of dataset 1') in case of whitespace."
        "\nWhen passed, one name per specified dataset in the same order as --dataset_paths.",
    )
    parser.add_argument(
        "--use_included_features",
        action="store_true",
        help="Whether to use the included features in the model training. "
        "When specified, the `--resources_dir` must also be specified. "
        "When NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(  # TODO Fix help
        "--k_outer",
        type=int,
        default=10,
        help="Number of outer folds in **within-dataset** cross-validation for tuning hyperparameters via grid search. "
        "**Ignored** when multiple test datasets are specified, as leave-one-dataset-out cross-validation is used instead.",
    )
    parser.add_argument(  # TODO Fix help
        "--k_inner",
        type=int,
        default=10,
        help="Number of inner folds in **within-dataset** cross-validation for tuning hyperparameters via grid search. "
        "**Ignored** when 4 or more test datasets are specified, as leave-one-dataset-out cross-validation is used instead.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Number of iterations/epochs to train the model. A good default is around 5000. "
        "Not necessarily used in all models.",
    )
    parser.add_argument(
        "--train_only",
        type=str,
        nargs="*",
        help="Indices of specified datasets that should only be used for training "
        "during CV for hyperparameter tuning. 0-index so in the range 0->(num_datasets-1). "
        # TODO: Figure out what to do with one test dataset and n train-only datasets?
        "When `--use_included_features` is NOT specified, at least one dataset cannot be train-only. "
        # TODO: Should we allow setting included features to train-only?
        "",
    )
    parser.add_argument(
        "--pca_target_variance",
        type=float,
        default=[0.994, 0.995, 0.996, 0.997, 0.998],
        nargs="*",
        help="Target(s) for explained variance of principal components. Selects number of components by this. "
        "When multiple targets are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--lasso_c",
        type=float,
        default=np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]),
        nargs="*",
        help="Inverse Lasso regularization strength value(s) for `sklearn.linear_model.LogisticRegression`. "
        "When multiple values are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate *predictions* per subject before evaluations. "
        "The predicted probabilities averaged per group."
        "Only the evaluations are affected by this. "
        "**Ignored** when no subjects are present in the meta data.",
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


examples = Examples(
    introduction="While the examples don't use parallelization, it is recommended to use `--num_jobs 10` for a big speedup."
)
# TODO: Make into CV example
examples.add_example(
    description="Simple example using defaults:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
EPILOG = examples.construct()


def main(args):
    out_path = pathlib.Path(args.out_dir)
    resources_dir = pathlib.Path(args.resources_dir)

    # Create output directory
    paths = IOPaths(
        in_dirs={
            "resources_dir": resources_dir,
        },
        out_dirs={
            "out_path": out_path,
        },
    )
    paths.mk_output_dirs(collection="out_dirs")

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="cross-validate-model-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running cross-validation of model")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    (
        model_dict,
        transformers_fn,
        dataset_paths,
        train_only,
        meta_data_paths,
        feature_name_to_feature_group_path,
    ) = prepare_modeling_command(
        args=args,
        paths=paths,
        messenger=messenger,
    )

    run_nested_cross_validation(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task="binary_classification",
        model_dict=model_dict,
        labels_to_use=["0_Control(control)", "1_Cancer(cancer)"],
        feature_sets=[0],
        train_only_datasets=train_only,
        k_outer=args.k_outer,
        k_inner=args.k_inner,
        transformers=transformers_fn,
        aggregate_by_groups=args.aggregate_by_subjects,
        weight_loss_by_groups=True,
        weight_per_dataset=True,
        expected_shape={1: 10, 2: 489},  # 10 feature sets, 489 cell types
        inner_metric="balanced_accuracy",
        refit=True,
        num_jobs=args.num_jobs,
        seed=args.seed,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
