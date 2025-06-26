"""
Script that cross-validates with specified features / cohorts..

"""

import logging
import pathlib
import warnings
from utipy import Messenger, StepTimer, IOPaths
from generalize.model.cross_validate import make_simplest_model_refit_strategy

from lionheart.modeling.prepare_modeling_command import prepare_modeling_command
from lionheart.modeling.run_cross_validate import run_nested_cross_validation
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples, Guide
from lionheart.utils.global_vars import (
    LABELS_TO_USE,
    LASSO_C_OPTIONS,
    LASSO_C_OPTIONS_STRING,
    PCA_TARGET_VARIANCE_OPTIONS,
    PCA_TARGET_VARIANCE_OPTIONS_STRING,
)

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

# Disable font manager debugging messages
logging.getLogger("matplotlib.font_manager").disabled = True


def setup_parser(parser, show_advanced: bool):
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
        "\n  3) the third column contains the <b>cancer type</b>. "
        + (
            (
                "Only used for leave-one-cancer-type-out (`--loco`) cross-validation. "
                "\n      Any strings other than 'control' and 'exclude' is considered a cancer type (i.e., the `cancer status` column is ignored in `--loco`). "
                "\n      Use the same name for a given cancer type across all datasets. "
            )
            if show_advanced
            else "This is only used in advanced modes.\n      For the binary cancer vs. control classification, any string will do (not used)."
        )
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
                "\n     <b>NOTE</b>: When not running subtyping or `--loco`, any character value is fine."
            )
            if False  # ENABLE_SUBTYPING
            else ""
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
            "Path to directory to store the cross-validation results at. "
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        required=True,
        help="Path to directory with framework resources such as the included features.",
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
        help="Whether to use the included features in the cross-validation."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--k_outer",
        type=int,
        default=10,
        help="Number of outer folds in <i>within-dataset</i> cross-validation. "
        "\n<u><b>Ignored</b></u> when multiple test datasets are specified, "
        "as leave-one-dataset-out cross-validation is used instead.",
    )
    parser.add_argument(
        "--k_inner",
        type=int,
        default=10,
        help="Number of inner folds in cross-validation for tuning hyperparameters via grid search. "
        "\n<u><b>Ignored</b></u> when 4 or more <i>test</i> datasets (incl. included features) are specified, "
        "as leave-one-dataset-out cross-validation is used instead.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30000,
        help="Number of iterations/epochs to train the model.",
    )
    parser.add_argument(
        "--train_only",
        type=int,
        nargs="*",
        help="Indices of specified datasets that should only be used for training."
        "\n0-indexed so in the range 0->(num_datasets-1)."
        # TODO: Figure out what to do with one test dataset and n train-only datasets?
        "\nWhen --use_included_features is NOT specified, at least one dataset cannot be train-only."
        "\nWHEN TO USE: If you have a dataset with only one of the classes (controls or cancer) "
        "\nwe cannot test on the dataset. It may still be a great addition"
        "\nto the training data, so flag it as 'train-only'.",
    )
    parser.add_argument(
        "--merge_datasets",
        type=str,
        nargs="*",
        help="List of dataset groups that should be merged into a single dataset. "
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
        default=PCA_TARGET_VARIANCE_OPTIONS,
        nargs="*",
        help="Target(s) for the explained variance of selected principal components."
        "\nUsed to select the most-explaining components."
        "\nWhen multiple targets are provided, they are used in grid search. "
        "\nDefaults to: " + PCA_TARGET_VARIANCE_OPTIONS_STRING,
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
        "\nThe predicted probabilities are averaged per subject."
        "\nOnly the evaluations are affected by this. "
        "\n<u><b>Ignored</b></u> when no subject IDs are present in the meta data.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions.",
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
    if show_advanced:
        adv = parser.add_argument_group("Advanced options")
        adv.add_argument(
            "--feature_categories",
            type=str,
            nargs="*",
            help="Cell type category to use / exclude. See the categories in "
            "`<resources_dir>/feature_names_and_grouping.csv`. "
            "\nSpecify either a set of categories to use (e.g. `--feature_categories=Blood/Immune`) "
            "or a set of categories to exclude (e.g. `--feature_categories=-Blood/Immune`). "
            "When excluding, be sure to use `=-` so the value is not interpreted as an argument.",
        )
        adv.add_argument(
            "--feature_type",
            type=str,
            default="LIONHEART",
            choices=["LIONHEART", "bin_depths", "lengths", "length_ratios"],
            help="The feature type (for benchmarking). "
            "One of {'LIONHEART', 'bin_depths', 'lengths', 'length_ratios'}. "
            "\nNote that many options only work with LIONHEART scores.",
        )
        adv.add_argument(
            "--loco",
            action="store_true",
            help="Whether to run leave-one-class-out cross-validation. "
            "\nAll datasets will be merged and each class (cancer type) "
            "becomes a fold along with a proportional number of sampled controls. "
            "\nThe model still predicts 'cancer vs. control'."
            "\nNote: This does NOT represent cross-dataset generalization!",
        )
        adv.add_argument(
            "--loco_train_only_classes",
            type=str,
            nargs="*",
            help="Names of cancer types that should only be used for training (`--loco` only).",
        )
    else:
        # Declare defaults for advanced options so the args
        # can be used without existence checks
        parser.set_defaults(
            feature_type="LIONHEART",
            feature_categories=[],
            loco=False,
            loco_train_only_classes=False,
        )

    parser.set_defaults(func=main)


# TODO: Allow specifying the thresholds as in other commands?
# TODO: Also, rename Threshold Version to Threshold Name as in the other commands

epilog_guide = Guide()
epilog_guide.add_title("OUTPUT:")
epilog_guide.add_description(
    """evaluation_summary.csv : data frame
    Overall summarized evaluation metrics per threshold.
    To get just the average AUC for the Max. Youden's J threshold, as reported in the paper, use:
        `$ awk 'NR==1 || /Average/ && /J Threshold/' {out_dir}/evaluation_summary.csv`

    Columns:
        Measure: The summary statistic that the row represents.
        ...
        Threshold: The actual probability cutoff used to determine the predicted class.
        Threshold Version: The name of the threshold (i.e. probability cutoff) used for decision making.
        Model: Name of the applied model <i>architecture</i>.
        Seed: The random state used. For reproducibility.
    
splits_summary.csv : data frame
    Summarized evaluation metrics per <i>dataset</i> from the leave-one-<i>dataset</i>-out cross-validation.
    That is, how well training on all the other dataset and predicting on the listed dataset works.
    To get just the average AUC for the Max. Youden's J threshold, as reported in the paper, use:
        `$ awk 'NR==1 || /Average/ && /J Threshold/' {out_dir}/splits_summary.csv`

evaluation_scores.csv : data frame
    This data frame contains the evaluation scores from each train/test split in the outer cross-validation.
    
    Columns:
        ...
        Threshold: The actual probability cutoff used to determine the predicted class.
        Positive Class: The name of the positive class used to calculate the metrics.
        Num Classes: The number of classes.
        Fold: The name of the outer <i>test</i> fold (i.e., <i>dataset</i> tested on when using leave-one-dataset-out cross-validation).
        Model: Name of the applied model <i>architecture</i>.
        Threshold Version: The name of the threshold (i.e. probability cutoff) used for decision making.
        Num Warnings: Number of warnings caught during the cross-validation. If any, see them in `warnings.csv`.
        
predictions.csv : data frame
    This data frame contains the predicted probabilities per sample.
    
    Columns:
        Prediction: The probability of the sample being from a cancer patient.
        Target: The actual cancer status of the sample.
        Group: The unique subject identifier (when specified in the meta data).
        Sample ID: The unique sample identifier.
        Split: The name of the outer <i>test</i> fold (i.e., <i>dataset</i> tested on when using leave-one-dataset-out cross-validation).
        Model: Name of the applied model <i>architecture</i>.
        Seed: The random state used. For reproducibility.

best_coefficients.csv : data frame
    The coefficient values for the best hyperparameter combinations. 
    Zero-padded column-wise, since different numbers of features can be present after PCA and LASSO. Remove all zeroes from the "right" to remove padding.
    The final column ("outer_split") identifies the outer loop fold, although it cannot be mapped back to the datasets.

inner_results.csv : data frame
    Evaluation scores from the inner cross-validation for each hyperparameter combination.
    The final column ("outer_split") identifies the outer loop fold, although it cannot be mapped back to the datasets.
    Used to plot the `inner_cv_*.png` files.

ROC_curves.json : dict
    The ROC curves from each train/test split in the outer cross-validation.
    Can be loaded with `ROCCurves.load()` from `generalize` or just as a json file.

confusion_matrices.json : dict
    The confusion matrices from each train/test split in the outer cross-validation.
    Can be loaded with `ConfusionMatrices.load()` from `generalize` or just as a json file.
    To get the total (sum) confusion matrix, see `total_confusion_matrices.json`.

"""
)
epilog_guide.add_vertical_space(1)

examples = Examples(
    introduction="While the examples don't use parallelization, it is recommended to use `--num_jobs 10` for a big speedup."
)

examples.add_example(
    description="Cross-validate with only the shared features:",
    example="""--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
examples.add_example(
    description="Cross-validating with two custom datasets and the included datasets:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
--dataset_names 'dataset_1' 'dataset_2'
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
examples.add_example(
    description="Cross-validating on a single dataset. This uses classic nested K-fold cross-validation:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory""",
)

EPILOG = epilog_guide.construct_guide() + examples.construct()


def main(args):
    out_path = pathlib.Path(args.out_dir)
    resources_dir = pathlib.Path(args.resources_dir)

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="cross-validate-model-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running cross-validation of model")
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

    # TODO: Take merge_datasets into account here?
    if args.k_inner < 0 or len(dataset_paths) - len(train_only) >= 4 and not args.loco:
        args.k_inner = None
        messenger(
            "Overriding --k_inner: Inner loop will use leave-one-dataset-out cross-validation "
            "to optimize hyperparameters for cross-dataset generalization. "
        )

    refit = (
        make_simplest_model_refit_strategy(
            main_var=("model__C", "minimize"),
            score_name="balanced_accuracy",
            other_vars=[("pca__target_variance", "minimize")],
            messenger=messenger,
        )
        if args.k_inner is not None or args.loco
        else True
    )

    labels_to_use = LABELS_TO_USE
    # --loco checks
    if args.loco:
        loco_checks(args, messenger)
        labels_to_use = None

    expected_shapes = {
        "LIONHEART": {1: 10, 2: 898},  # 10 feature sets, 898 cell type features
        "length_ratios": {1: 2689},
        "bin_depths": {1: 2689},
        "lengths": {1: 321},
    }
    feature_sets = {
        "LIONHEART": [0],
        "length_ratios": None,
        "bin_depths": None,
        "lengths": None,
    }

    run_nested_cross_validation(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task="binary_classification"
        if not args.loco
        else "leave_one_class_out_binary_classification",
        model_dict=model_dict,
        labels_to_use=labels_to_use,
        feature_sets=feature_sets[args.feature_type],
        train_only_datasets=train_only,
        merge_datasets=merge_datasets,
        k_outer=args.k_outer,
        k_inner=args.k_inner,
        reps=args.reps,
        transformers=transformers_fn,
        aggregate_by_groups=args.aggregate_by_subjects,
        weight_loss_by_groups=True,
        weight_per_dataset=True,
        expected_shape=expected_shapes[args.feature_type],
        inner_metric="balanced_accuracy",
        refit=refit,
        num_jobs=args.num_jobs,
        seed=args.seed,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def loco_checks(args, messenger):
    if args.train_only:
        raise NotImplementedError(
            "`--train_only` datasets is not currently supported in leave-one-class-out (`--loco`) cross-validation."
        )
    if args.merge_datasets:
        messenger(
            "`--merge_datasets` is ignored in leave-one-class-out cross-validation, "
            "as all datasets are merged to one.",
            add_msg_fn=warnings.warn,
        )
