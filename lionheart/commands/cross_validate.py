"""
Script that cross-validates with specified features / cohorts..

"""

import logging
import pathlib
from utipy import Messenger, StepTimer, IOPaths

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
            if False  # ENABLE_SUBTYPING
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
        type=str,
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

    if args.k_inner < 0 or len(dataset_paths) - len(train_only) >= 4:
        args.k_inner = None
        messenger(
            "Overriding --k_inner: Inner loop will use leave-one-dataset-out cross-validation "
            "to optimize hyperparameters for cross-dataset generalization. "
        )

    run_nested_cross_validation(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=feature_name_to_feature_group_path,
        task="binary_classification",
        model_dict=model_dict,
        labels_to_use=LABELS_TO_USE,
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
