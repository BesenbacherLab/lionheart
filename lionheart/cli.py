import re
import argparse
from rich_argparse import RawTextRichHelpFormatter
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from lionheart.commands import collect_samples, extract_features, predict, train_model


# Add styles for bold and italic text
RawTextRichHelpFormatter.styles["argparse.bold"] = "bold"
RawTextRichHelpFormatter.styles["argparse.italic"] = "italic"
RawTextRichHelpFormatter.styles["argparse.underline"] = "underline"
RawTextRichHelpFormatter.styles["argparse.group_header"] = (
    RawTextRichHelpFormatter.styles["argparse.groups"]
)


# Define highlight regex patterns for bold and italic, ensuring the markers are not included in the captured groups
RawTextRichHelpFormatter.highlights.append(r"\<b\>(?P<bold>.+?)\</b\>")
RawTextRichHelpFormatter.highlights.append(r"\<i\>(?P<italic>.+?)\</i\>")
RawTextRichHelpFormatter.highlights.append(r"\<u\>(?P<underline>.+?)\</u\>")
RawTextRichHelpFormatter.highlights.append(r"\<h1\>(?P<group_header>.+?)\</h1\>")


# Custom formatter class to remove markers after formatting
class CustomRichHelpFormatter(RawTextRichHelpFormatter):
    def _strip_html_tags(self, text):
        # Remove bold markers
        text = re.sub(r"\</?b\>", "", text)
        text = re.sub(r"\</?i\>", "", text)
        text = re.sub(r"\</?u\>", "", text)
        text = re.sub(r"\</?h1\>", "", text)
        return text

    def format_help(self):
        help_text = super().format_help()
        return self._strip_html_tags(help_text)

    def _format_action(self, action):
        action_help = super()._format_action(action)
        return self._strip_html_tags(action_help)

    def _format_text(self, text):
        formatted_text = super()._format_text(text)
        return self._strip_html_tags(formatted_text)


def main():
    parser = argparse.ArgumentParser(
        description="""<b>LIONHEART Cancer Detector</b>\n\nDetect Cancer from whole genome sequenced plasma cell-free DNA.\n\n
Start by <i>extracting</i> the features from a BAM file. Then <i>predict</i> whether a sample is from a cancer patient or not.\n\n
Easily <i>train</i> a new model on your own data or perform <i>cross-validation</i> to compare against the paper.
        """,
        formatter_class=CustomRichHelpFormatter,
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
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    extract_features.setup_parser(parser_ef)

    # Command 2
    parser_ps = subparsers.add_parser(
        "predict_sample",
        help="Predict cancer status of a sample",
        description="Predict whether a sample is cancer or control.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    predict.setup_parser(parser_ps)

    # Command 3
    parser_cl = subparsers.add_parser(
        "collect",
        help="Collect predictions and/or features across samples",
        description="Collect predictions and/or extracted features for multiple samples.",
        formatter_class=parser.formatter_class,
    )
    # Delegate the argument setup to the respective command module
    collect_samples.setup_parser(parser_cl)

    # Command 4
    parser_tm = subparsers.add_parser(
        "train_model",
        help="Train a model on your own data and/or the included features",
        description="Train a model on your extracted features and/or the included features.",
        formatter_class=parser.formatter_class,
        epilog=train_model.EPILOG,
    )
    # Delegate the argument setup to the respective command module
    train_model.setup_parser(parser_tm)

    # Command 5
    parser_cv = subparsers.add_parser(
        "cross_validate",
        help="Cross-validate the cancer detection model on your own data and/or the included features",
        description="Perform nested leave-one-dataset-out (or classic) cross-validation "
        "with your extracted features and/or the included features.",
        formatter_class=parser.formatter_class,
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
