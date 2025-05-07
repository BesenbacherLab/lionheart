import argparse
import os
from lionheart.commands import (
    collect_samples,
    customize_thresholds,
    extract_features,
    predict,
    train_model,
    validate,
    cross_validate,
    evaluate_univariates,
    guides,
)
from lionheart.utils.cli_utils import (
    LION_ASCII,
    LIONHEART_ASCII,
    LIONHEART_STRING,
    README_STRING,
    CustomRichHelpFormatter,
    wrap_command_description,
)

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def main():
    # 1. Check the env var (you could also default to False if missing)
    show_advanced = os.getenv("LH_ADVANCED", "").lower() in ("1", "true", "yes")

    parser = argparse.ArgumentParser(
        description=f"""\n\n                                                                               
{LION_ASCII}                                        

{LIONHEART_ASCII}

<b>L</b>iquid B<b>i</b>opsy C<b>o</b>rrelati<b>n</b>g C<b>h</b>romatin
Acc<b>e</b>ssibility and cfDN<b>A</b> Cove<b>r</b>age
Across Cell-<b>T</b>ypes
.................

Detect Cancer from whole genome sequenced plasma cell-free DNA.

Start by <b>extracting</b> the features from a BAM file (hg38 only). Then <b>predict</b> whether a sample is from a cancer patient or not.

Easily <b>train</b> a new model on your own data or perform <b>cross-validation</b> to compare against the paper.

{README_STRING}
        """,
        formatter_class=CustomRichHelpFormatter,
    )

    # Add --version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"lionheart {metadata.version('lionheart')}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        # description="",
        help="additional help",
        dest="command",
    )

    # Command 0
    # subparsers.add_parser(
    #     "guide_me",
    #     help=f"Print a guide of the steps and processes in using {LIONHEART_STRING}",
    #     description=wrap_command_description(
    #         f"Run this command to show a guide of the steps and processes in using {LIONHEART_STRING}."
    #     ),
    #     formatter_class=parser.formatter_class,
    # )

    # Command 1
    parser_ef = subparsers.add_parser(
        "extract_features",
        help="Extract features from a BAM file",
        description=wrap_command_description(extract_features.DESCRIPTION),
        formatter_class=parser.formatter_class,
        epilog=extract_features.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    extract_features.setup_parser(parser_ef)

    # Command 2
    parser_ps = subparsers.add_parser(
        "predict_sample",
        help="Predict cancer status of a sample",
        description=wrap_command_description("PREDICT the cancer status of a sample."),
        formatter_class=parser.formatter_class,
        epilog=predict.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    predict.setup_parser(parser_ps)

    # Command 3
    parser_cl = subparsers.add_parser(
        "collect",
        help="Collect predictions and/or features across samples",
        description=wrap_command_description(
            "COLLECT predictions and/or extracted features for multiple samples. "
            "\nCollecting the features creates a 'dataset' that can be used in "
            "other `lionheart` commands."
        ),
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    collect_samples.setup_parser(parser_cl)

    # Command 4
    parser_er = subparsers.add_parser(
        "customize_thresholds",
        help="Extract ROC curve and probability densitities for using custom probability thresholds",
        description=wrap_command_description(
            "Extract the ROC curve and probability densities for a model's predictions on one or more datasets. "
            "\nThis allows using probability thresholds fitted to your own data in "
            "`lionheart predict_sample` and `lionheart validate`."
        ),
        formatter_class=parser.formatter_class,
        epilog=customize_thresholds.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    customize_thresholds.setup_parser(parser_er)

    # # Command 5
    parser_cv = subparsers.add_parser(
        "cross_validate",
        help="Cross-validate the cancer detection model on your own data and/or the included features",
        description=wrap_command_description(
            "CROSS-VALIDATE your features with nested leave-one-dataset-out (or classic) cross-validation. "
            "Use your extracted features and/or the included features. "
            "\nAllows seeing the effect on generalization of adding your own data to the training. "
            "\n\nNote: The settings are optimized for use with the included features and optional "
            "additional datasets. They may not be optimal for more custom designs."
            "\n\nAdvanced options: Enable advanced options by setting the `LH_ADVANCED=1` environment variable "
            "(e.g. `LH_ADVANCED=1 lionheart cross_validate --help´)."
        ),
        formatter_class=parser.formatter_class,
        epilog=cross_validate.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    cross_validate.setup_parser(parser_cv, show_advanced)

    # Command 6
    parser_tm = subparsers.add_parser(
        "train_model",
        help="Train a model on your own data and/or the included features",
        description=wrap_command_description(
            "TRAIN A MODEL on your extracted features and/or the included features. "
            "\n\nNOTE: The included evaluation is of predictions of the training data."
        ),
        formatter_class=parser.formatter_class,
        epilog=train_model.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_tm)

    # # Command 7
    parser_va = subparsers.add_parser(
        "validate",
        help="Validate a model on a validation dataset",
        description=wrap_command_description(
            "VALIDATE your trained model on a validation dataset, such as the included validation dataset."
        ),
        formatter_class=parser.formatter_class,
        epilog=validate.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    validate.setup_parser(parser_va)

    # # Command 8
    parser_eu = subparsers.add_parser(
        "evaluate_univariates",
        help="Evaluate the cancer detection potential of each feature separately on your own data and/or the included features",
        description=wrap_command_description(
            "EVALUATE your features separately on their cancer detection potential. "
            "Use your extracted features and/or the included features. "
        ),
        formatter_class=parser.formatter_class,
        epilog=evaluate_univariates.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    evaluate_univariates.setup_parser(parser_eu)

    args = parser.parse_args()
    if args.command == "guide_me":
        formatter = parser._get_formatter()
        formatter.add_text(guides.get_usage_guide())
        parser._print_message(formatter.format_help())
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
