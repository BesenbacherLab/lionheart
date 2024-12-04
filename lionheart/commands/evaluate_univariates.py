"""
Script that cross-validates with specified features / cohorts..

"""

import logging
import pathlib
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths

from lionheart.modeling.prepare_modeling_command import prepare_modeling_command
from lionheart.modeling.run_univariate_analyses import run_univariate_analyses
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples


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
        "--k",
        type=int,
        default=10,
        help="Number of folds in <i>within-dataset</i> cross-validation. "
        "\n<u><b>Ignored</b></u> when multiple test datasets are specified, "
        "as leave-one-dataset-out cross-validation is used instead.",
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
    parser.set_defaults(func=main)


examples = Examples(
    introduction="While the examples don't use parallelization, it is recommended to use `--num_jobs 10` for a big speedup."
)

examples.add_example(
    description="Evaluate univariates with only the shared features:",
    example="""--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
examples.add_example(
    description="Evaluate univariates with two custom datasets and the included datasets:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy path/to/dataset_2/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv path/to/dataset_2/meta_data.csv
--dataset_names 'dataset_1' 'dataset_2'
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory""",
)
examples.add_example(
    description="Evaluate univariates on a single dataset. This uses classic nested K-fold cross-validation:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy
--meta_data_paths path/to/dataset_1/meta_data.csv
--out_dir path/to/output/directory
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
    setup_logging(dir=str(out_path / "logs"), fname_prefix="evaluate-univariates-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running univariate analysis of model")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    (
        _,
        _,
        dataset_paths,
        train_only,
        meta_data_paths,
        feature_name_to_feature_group_path,
    ) = prepare_modeling_command(
        args=args,
        paths=paths,
        messenger=messenger,
    )

    # Feature names and groups
    feature_name_to_feature_group = pd.read_csv(feature_name_to_feature_group_path)
    feature_names = feature_name_to_feature_group.iloc[:, 1].astype("string")
    feature_group_names = feature_name_to_feature_group.iloc[:, 3].astype("string")

    run_univariate_analyses(
        dataset_paths=dataset_paths,
        out_path=paths["out_path"],
        meta_data_paths=meta_data_paths,
        task="binary_classification",
        names=feature_names,
        cell_groups=feature_group_names,
        labels_to_use=["0_Control(control)", "1_Cancer(cancer)"],
        feature_sets=[0],
        train_only_datasets=train_only,
        k=args.k,
        standardize_cols=True,
        standardize_rows=True,
        weight_loss_by_groups=True,
        weight_per_dataset=True,
        expected_shape={1: 10, 2: 489},  # 10 feature sets, 489 cell types
        aggregate_by_groups=args.aggregate_by_subjects,
        bonferroni_correct=True,
        num_jobs=args.num_jobs,
        seed=args.seed,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")