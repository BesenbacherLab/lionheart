"""
Script that validates a model on one or more specified validation datasets.

"""

import logging
import pathlib
from utipy import Messenger, StepTimer, IOPaths
from lionheart.modeling.run_customize_thresholds import run_customize_thresholds
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples
from lionheart.utils.global_vars import INCLUDED_MODELS, LABELS_TO_USE
from lionheart.modeling.prepare_modeling_command import prepare_validation_command

# TODO Not implemented
# - Add the figure of sens/spec thresholds in train and test ROCs


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
            "Path to directory to store the validation outputs in."
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--use_included_features",
        action="store_true",
        help="Whether to use the included features."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        help="Path to directory with framework resources.",
    )
    parser.add_argument(
        "--model_name",
        choices=INCLUDED_MODELS,
        type=str,
        help="Name of the included model to validate."
        "\nNOTE: only one of `--model_name` and `--custom_model_dir` can be specified.",
    )
    parser.add_argument(
        "--custom_model_dir",
        type=str,
        help="Path to a directory with a custom model to use. "
        "\nThe directory must include the files `model.joblib` and `training_info.json`.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate <i>predictions</i> per subject before evaluations. "
        "\nThe predicted probabilities are averaged per subject."
        "\n<u><b>Ignored</b></u> when no subject IDs are present in the meta data.",
    )
    parser.set_defaults(func=main)


# TODO: Add more examples
examples = Examples()
examples.add_example(
    description="Extract ROC curve for a custom model on a custom dataset:",
    example="""--dataset_paths path/to/dataset_1/feature_dataset.npy 
--meta_data_paths path/to/dataset_1/meta_data.csv 
--out_dir path/to/model_validation
--resources_dir path/to/resource/directory
--custom_model_dir path/to/new_model
""",
)
examples.add_example(
    description="Extract ROC curve for an included model on a custom dataset:",
    example=f"""--dataset_paths path/to/dataset_1/feature_dataset.npy 
--meta_data_paths path/to/dataset_1/meta_data.csv 
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory
--model_name {INCLUDED_MODELS[0]}
""",
)
examples.add_example(
    description="Extract ROC curve for included model on included datasets (its training-data):",
    example=f"""--out_dir path/to/output/directory
--resources_dir path/to/resource/directory
--model_name {INCLUDED_MODELS[0]}
--use_included_features
""",
)
EPILOG = examples.construct()


def main(args):
    out_path = pathlib.Path(args.out_dir)

    if sum([args.model_name is not None, args.custom_model_dir is not None]) != 1:
        raise ValueError(
            "Exactly one of {`--model_name`, `--custom_model_dir`} "
            "should be specified at a time."
        )
    if args.model_name is not None:
        if args.resources_dir is None:
            raise ValueError(
                "When `--model_name` is specified, "
                "`--resources_dir` must also be specified."
            )

        resources_dir = pathlib.Path(args.resources_dir)
        model_dir = resources_dir / "models" / args.model_name

    else:
        resources_dir = None
        model_dir = pathlib.Path(args.custom_model_dir)

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="customize-thresholds-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running threshold customization")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    paths = IOPaths(out_dirs={"out_dir": out_path})
    if resources_dir is not None:
        paths.set_path("resources_dir", resources_dir, "in_dirs")

    dataset_paths, meta_data_paths = prepare_validation_command(
        args=args,
        paths=paths,
        messenger=messenger,
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    # Show overview of the paths
    messenger(paths)

    run_customize_thresholds(
        dataset_paths=dataset_paths,
        out_path=out_path,
        meta_data_paths=meta_data_paths,
        model_dir=model_dir,
        labels_to_use=LABELS_TO_USE,
        feature_sets=[0],
        aggregate_by_groups=args.aggregate_by_subjects,
        expected_shape={1: 10, 2: 898},  # 10 feature sets, 898 cell types
        timer=timer,
        messenger=messenger,
    )

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
