import argparse
from argparse import RawTextHelpFormatter
from lionheart.commands import extract_features, predict


def main():
    parser = argparse.ArgumentParser(
        description="""**LIONHEART Cancer Detector**\\nDetect Cancer from whole genome sequenced cell-free DNA.\n\n
Start by *extracting* the features from a BAM file. Then *predict* whether a sample is from a cancer patient or not.
        """,
        formatter_class=RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        # description="",
        help="additional help",
        dest="command",
    )

    # Command1
    parser_c1 = subparsers.add_parser(
        "extract_features",
        help="Extract features",
        description="Extract features from a BAM file.",
    )
    # Delegate the argument setup to the respective command module
    extract_features.setup_parser(parser_c1)

    # Command2
    parser_c2 = subparsers.add_parser(
        "predict_sample",
        help="Predict sample",
        description="Predict whether a sample is cancer or control.",
    )
    # Delegate the argument setup to the respective command module
    predict.setup_parser(parser_c2)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
