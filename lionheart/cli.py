import argparse
from lionheart.commands import (
    collect_samples,
    extract_features,
    predict,
    train_model,
    validate,
    cross_validate,
    guides,
)
from lionheart.utils.cli_utils import (
    LION_ASCII,
    LIONHEART_ASCII,
    LIONHEART_STRING,
    CustomRichHelpFormatter,
    wrap_command_description,
)


def main():
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
        """,
        formatter_class=CustomRichHelpFormatter,
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
            "COLLECT predictions and/or extracted features for multiple samples."
        ),
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    collect_samples.setup_parser(parser_cl)

    # Command 4
    parser_tm = subparsers.add_parser(
        "train_model",
        help="Train a model on your own data and/or the included features",
        description=wrap_command_description(
            "TRAIN A MODEL on your extracted features and/or the included features."
        ),
        formatter_class=parser.formatter_class,
        epilog=train_model.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_tm)

    # # Command 5
    # parser_va = subparsers.add_parser(
    #     "validate",
    #     help="Validate a trained model on one or more validation datasets",
    #     description=wrap_command_description(
    #         "VALIDATE your trained model one or more validation datasets, such as the included validation dataset."
    #     ),
    #     formatter_class=parser.formatter_class,
    #     epilog=validate.EPILOG,
    # )
    # # Delegate the argument setup to the respective command module
    # validate.setup_parser(parser_va)

    # # Command 6
    # parser_cv = subparsers.add_parser(
    #     "cross_validate",
    #     help="Cross-validate the cancer detection model on your own data and/or the included features",
    #     description=wrap_command_description(
    #         "CROSS-VALIDATE your features with nested leave-one-dataset-out (or classic) cross-validation. "
    #         "Use your extracted features and/or the included features."
    #     ),
    #     formatter_class=parser.formatter_class,
    # )
    # # Delegate the argument setup to the respective command module
    # cross_validate.setup_parser(parser_cv)

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
