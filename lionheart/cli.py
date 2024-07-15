import argparse
from argparse import RawTextHelpFormatter
from lionheart.commands import collect_samples, extract_features, predict, train_model


def main():
    parser = argparse.ArgumentParser(
        description="""**LIONHEART Cancer Detector**\n\nDetect Cancer from whole genome sequenced plasma cell-free DNA.\n\n
Start by *extracting* the features from a BAM file. Then *predict* whether a sample is from a cancer patient or not.\n\n
Easily *train* a new model on your own data or perform **cross-validation** to compare against the paper.
        """,
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        # description="",
        help="additional help",
        dest="command",
    )

    # Command 1
    parser_ef = subparsers.add_parser(
        "extract_features",
        help="Extract features from a BAM file",
        description="Extract features from a BAM file.",
    )
    # Delegate the argument setup to the respective command module
    extract_features.setup_parser(parser_ef)

    # Command 2
    parser_ps = subparsers.add_parser(
        "predict_sample",
        help="Predict cancer status for a sample",
        description="Predict whether a sample is cancer or control.",
    )
    # Delegate the argument setup to the respective command module
    predict.setup_parser(parser_ps)

    # Command 3
    parser_cl = subparsers.add_parser(
        "collect",
        help="Collect predictions and/or features across samples",
        description="Collect predictions and/or extracted features for multiple samples.",
    )
    # Delegate the argument setup to the respective command module
    collect_samples.setup_parser(parser_cl)

    # Command 4
    parser_tm = subparsers.add_parser(
        "train_model",
        help="Train a model on your own data and/or the included features",
        description="Train a model on your extracted features and/or the included features.",
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_tm)

    # Command 5
    parser_cv = subparsers.add_parser(
        "cross_validate",
        help="Cross-validate the cancer detection model on your own data and/or the included features",
        description="Perform nested leave-one-dataset-out (or classic) cross-validation "
        "with your extracted features and/or the included features.",
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_cv)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
