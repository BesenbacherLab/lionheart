"""
Command for training a new model on specified features.

"""

import logging
import pathlib
import joblib
from utipy import Messenger, StepTimer, IOPaths
from packaging import version
from generalize.model.cross_validate import make_simplest_model_refit_strategy

from lionheart.modeling.prepare_modeling_command import prepare_modeling_command
from lionheart.modeling.run_full_modeling import run_full_model_training
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.global_vars import (
    JOBLIB_VERSION,
    ENABLE_SUBTYPING,
    LABELS_TO_USE,
    LASSO_C_OPTIONS,
    LASSO_C_OPTIONS_STRING,
    PCA_TARGET_VARIANCE_FM_OPTIONS,
    PCA_TARGET_VARIANCE_FM_OPTIONS_STRING,
)
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
        "\nExpects shape <i>(?, 10, 898)</i> (i.e., <i># samples, # feature sets, # features</i>). "
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
        "\nUse quotes (e.g., 'name of dataset 1') in case of whitespace."
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
            "\nUse quotes (e.g., 'colorectal cancer') in case of whitespace."
            "\nControls can be included with 'control' although this is untested territory.",
        )
    # TODO: For help, check if k is used when <4 datasets are specified
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
        type=int,
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
        "--merge_datasets",
        type=str,
        nargs="*",
        help="List of dataset groups that should be merged into a single dataset "
        "during cross-validation in hyperparameter tuning. "
        "Given as `NewName(D1,D2,D3)`. "
        "\nOnly relevant when `dataset_paths` has >1 paths. "
        "\nNames must match those in `dataset_names` which must also be specified. \n\n"
        "Example: `--merge_datasets BestDataset(D1,D2) WorstDataset(D3,D4,D5)` "
        "would create 2 datasets where D1 and D2 make up the first, and D3-5 make up the second. "
        "\nDatasets not mentioned are not affected. \n\n"
        "Note: Be careful about spaces in the dataset names or make sure to quote each string. ",
    )
    parser.add_argument(
        "--pca_target_variance",
        type=float,
        default=PCA_TARGET_VARIANCE_FM_OPTIONS,
        nargs="*",
        help="Target(s) for the explained variance of selected principal components."
        "\nUsed to select the most-explaining components."
        "\nWhen multiple targets are provided, they are used in grid search."
        "\nDefaults to: " + PCA_TARGET_VARIANCE_FM_OPTIONS_STRING,
    )
    parser.add_argument(
        "--lasso_c",
        type=float,
        default=LASSO_C_OPTIONS,
        nargs="*",
        help="Inverse LASSO regularization strength value(s) for `sklearn.linear_model.LogisticRegression`."
        "\nWhen multiple values are provided, they are used in grid search."
        "\nDefaults to: " + LASSO_C_OPTIONS_STRING,
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
    # Declare defaults for cv-only args to allow sharing preparation function
    parser.set_defaults(
        feature_type="LIONHEART",
        feature_categories=[],
        loco=False,
        loco_train_only_classes=False,
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

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="train_model-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running training of model")
    messenger.now()

    # Create output directory
    paths = IOPaths(
        in_dirs={
            "resources_dir": resources_dir,
        },
        out_dirs={
            "out_path": out_path,
        },
    )
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

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
        merge_datasets,
        meta_data_paths,
        feature_name_to_feature_group_path,
    ) = prepare_modeling_command(
        args=args,
        paths=paths,
        messenger=messenger,
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
        labels_to_use=LABELS_TO_USE
        if not args.subtype
        else [
            f"{i}_{c.title().replace(' ', '_')}({c.lower()})"
            for i, c in enumerate(args.subtypes_to_use)
        ],
        feature_sets=[0],
        train_only_datasets=train_only,
        merge_datasets={"Combined Data": list(dataset_paths.keys())}
        if args.subtype
        else merge_datasets,
        k=args.k,
        transformers=transformers_fn,
        aggregate_by_groups=args.aggregate_by_subjects,
        weight_loss_by_groups=True,
        weight_per_dataset=True,
        expected_shape={1: 10, 2: 898},  # 10 feature sets, 898 cell type features
        refit_fn=make_simplest_model_refit_strategy(
            main_var=("model__C", "minimize"),
            score_name="balanced_accuracy",
            other_vars=[("pca__target_variance", "minimize")],
            messenger=messenger,
        )
        # TODO: Take merge_datasets into account here?
        if args.subtype or (len(dataset_paths) - len(train_only)) < 2
        else None,
        num_jobs=args.num_jobs,
        seed=args.seed,
        required_lionheart_version=args.required_lionheart_version,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
