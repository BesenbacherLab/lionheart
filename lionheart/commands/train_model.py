"""
Command for training a new model on specified features.

"""

import logging
import pathlib
import joblib
import numpy as np
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths
from packaging import version
from sklearn.linear_model import LogisticRegression
from generalize.model.cross_validate import make_simplest_model_refit_strategy

from lionheart.modeling.transformers import prepare_transformers_fn
from lionheart.modeling.run_full_modeling import run_full_model_training
from lionheart.modeling.model_dict import create_model_dict
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.global_vars import JOBLIB_VERSION, ENABLE_SUBTYPING
from lionheart.utils.cli_utils import Examples
from lionheart import __version__ as lionheart_version


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
        default=[],
        help="Path(s) to `feature_dataset.npy` file(s) containing the collected features. "
        "\nExpects shape <i>(?, 10, 489)</i> (i.e., <i># samples, # feature sets, # features</i>). "
        "\nOnly the first feature set is used.",
    )
    parser.add_argument(
        "--meta_data_paths",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to csv file(s) where:"
        "\n  1) the first column contains the <b>sample IDs</b>"
        "\n  2) the second column contains the <b>cancer status</b>\n      One of: {<i>'control', 'cancer', 'exclude'</i>}"
        "\n  3) the third column contains the <b>cancer type</b> "
        + (
            (
                "for subtyping (see --subtype)"
                "\n     Either one of:"
                "\n       {<i>'control', 'colorectal cancer', 'bladder cancer', 'prostate cancer',"
                "\n       'lung cancer', 'breast cancer', 'pancreatic cancer', 'ovarian cancer',"
                "\n       'gastric cancer', 'bile duct cancer', 'hepatocellular carcinoma',"
                "\n       'head and neck squamous cell carcinoma', 'nasopharyngeal carcinoma',"
                "\n       'exclude'</i>} (Must match exactly (case-insensitive) when using included features!) "
                "\n     or a custom cancer type."
                "\n     <b>NOTE</b>: When not running subtyping, any character value is fine."
            )
            if ENABLE_SUBTYPING
            else "[NOTE: Not currently used so can be any string value!]."
        )
        + "\n  4) the (optional) fourth column contains the <b>subject ID</b> "
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
            "Path to directory to store the trained model in."
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        required=True,
        help="Path to directory with framework resources such as the included features. ",
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
        help="Whether to use the included features in the model training."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    if ENABLE_SUBTYPING:
        parser.add_argument(
            "--subtype",
            action="store_true",
            help="Whether to train a multiclass classification model for predicting the cancer type."
            "\nSpecify the cancer types to include in the model via --subtypes_to_use."
            "\nBy default, only the cases are included (no controls)."
            "\nTypically, this model is run on the samples that the cancer detector predicts as cancer."
            "\nSubtyping models select hyperparameters via classical cross-validation (not on"
            "\ncross-dataset generalization) and are thus more likely to overfit. To reduce overfitting,"
            "\nwe select the model with lowest values of --lasso_c and --pca_target_variance"
            "\nthat score within a standard deviation of the best combination.",
        )
        parser.add_argument(
            "--subtypes_to_use",
            type=str,
            nargs="*",
            default=[
                "colorectal cancer",
                "bladder cancer",
                "prostate cancer",
                "lung cancer",
                "breast cancer",
                "pancreatic cancer",
                "ovarian cancer",
                "gastric cancer",
                "bile duct cancer",
                "hepatocellular carcinoma",
            ],
            help="The cancer types to include in the model when --subtype is specified."
            "\nBy default, only cancer types with >10 samples in the included features are used.\n"
            "\nUse quotes (e.g. 'colorectal cancer') in case of whitespace."
            "\nControls can be included with 'control' although this is untested territory.",
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
        help="Indices of specified datasets that should only be used for training"
        "during cross-validation\nfor hyperparameter tuning.\n0-indexed so in the range 0->(num_datasets-1)."
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
        default=[0.994, 0.995, 0.996, 0.997, 0.998, 0.999],
        nargs="*",
        help="Target(s) for the explained variance of selected principal components."
        "\nUsed to select the most-explaining components."
        "\nWhen multiple targets are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--lasso_c",
        type=float,
        default=np.array(
            [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
        ),
        nargs="*",
        help="Inverse LASSO regularization strength value(s) for `sklearn.linear_model.LogisticRegression`."
        "\nWhen multiple values are provided, they are used in grid search.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate <i>predictions</i> per subject before evaluations. "
        "\nThe predicted probabilities are averaged per group."
        "\nOnly the evaluations are affected by this. "
        "\n<u><b>Ignored</b></u> when no subject IDs are present in the meta data.",
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
    parser.add_argument(
        "--required_lionheart_version",
        type=str,
        help="Optionally set a minimally required LIONHEART version for this model instance.\n"
        "`lionheart predict_sample` will check for this version and fail if the LIONHEART installation is outdated.",
    )
    parser.set_defaults(func=main)


examples = Examples(
    introduction="While the examples don't use parallelization, it is recommended to use `--num_jobs 10` for a big speedup."
)
examples.add_example(
    description="Simple example using defaults:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
# TODO Implement this:
examples.add_example(
    description="Train a model on a single dataset. This uses within-dataset cross-validation for hyperparameter optimization:",
    example="""--dataset_paths path/to/dataset/feature_dataset.npy
--meta_data_paths path/to/dataset/meta_data.csv
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory""",
)
if ENABLE_SUBTYPING:
    examples.add_example(
        description="Subtyping example using defaults:",
        example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
    --meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
    --out_dir path/to/output/directory
    --use_included_features
    --resources_dir path/to/resource/directory
    --subtype""",
    )
    examples.add_example(
        description="Subtyping example with all cancer types (normally only include those with `n>10`).\nFor custom cancer types, add them to --subtypes_to_use.",
        example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
    --meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
    --out_dir path/to/output/directory
    --use_included_features
    --resources_dir path/to/resource/directory
    --subtype
    --subtypes_to_use 'colorectal cancer' 'bladder cancer' 'prostate cancer' 'lung cancer' 'breast cancer' 'pancreatic cancer' 'ovarian cancer' 'gastric cancer' 'bile duct cancer' 'hepatocellular carcinoma' 'head and neck squamous cell carcinoma' 'nasopharyngeal carcinoma'""",
    )
EPILOG = examples.construct()


def main(args):
    if not ENABLE_SUBTYPING:
        args.subtype = False

    # Start by checking version of joblib
    if joblib.__version__ != JOBLIB_VERSION:
        raise RuntimeError(
            f"Currently, `joblib` must be version {JOBLIB_VERSION}, got: {joblib.__version__}. "
            "Did you activate the correct conda environment?"
        )
    if args.required_lionheart_version is not None and version.parse(
        args.required_lionheart_version
    ) > version.parse(lionheart_version):
        raise RuntimeError(
            "`--required_lionheart_version` was never than "
            "the currently installed version of LIONHEART."
        )

    out_path = pathlib.Path(args.out_dir)
    resources_dir = pathlib.Path(args.resources_dir)

    # Create output directory
    paths = IOPaths(out_dirs={"out_path": out_path})
    paths.mk_output_dirs(collection="out_dirs")

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="train_model-")
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
        shared_features_dir = resources_dir / "shared_features"
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
        resources_dir / "feature_names_and_grouping.csv"
    )

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
        grid={"model__C": args.lasso_c},
    )

    transformers_fn = prepare_transformers_fn(
        pca_target_variance=args.pca_target_variance,
        min_var_thresh=[0.0],
        scale_rows=["mean", "std"],
        standardize=True,
    )

    run_full_model_training(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task="binary_classification"
        if not args.subtype
        else "multiclass_classification",
        model_dict=model_dict,
        labels_to_use=["0_Control(control)", "1_Cancer(cancer)"]
        if not args.subtype
        else [
            f"{i}_{c.title().replace(' ', '_')}({c.lower()})"
            for i, c in enumerate(args.subtypes_to_use)
        ],
        feature_sets=[0],
        train_only_datasets=train_only,
        merge_datasets={"Combined Data": list(dataset_paths.keys())}
        if args.subtype
        else None,
        k=args.k,
        transformers=transformers_fn,
        aggregate_by_groups=args.aggregate_by_subjects,
        weight_loss_by_groups=True,
        weight_per_dataset=True,
        expected_shape={1: 10, 2: 489},  # 10 feature sets, 489 cell types
        refit_fn=make_simplest_model_refit_strategy(
            main_var=("model__C", "minimize"),
            score_name="balanced_accuracy",
            other_vars=[("pca__target_variance", "minimize")],
            messenger=messenger,
        )
        if args.subtype
        else None,
        num_jobs=args.num_jobs,
        seed=args.seed,
        required_lionheart_version=args.required_lionheart_version,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
