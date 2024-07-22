"""
Script that validates model on with specified features / cohorts..

"""

import logging
import pathlib
import numpy as np
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths

from lionheart.utils.dual_log import setup_logging


def setup_parser(parser):
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="*",
        help="Path(s) to `feature_dataset.npy` file(s) containing the collected features. ",
    )
    parser.add_argument(
        "--meta_data_paths",
        type=str,
        nargs="*",
        help="Path(s) to csv file(s) where 1) the first column contains the sample IDs, "
        "and 2) the second contains their label, and 3) the (optional) third column contains subject ID "
        "(for when subjects have more than one sample). "
        "When `dataset_paths` has multiple paths, there must be "
        "one meta data path per dataset, in the same order.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the collected features at. "
            "A `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        help="Path to directory with framework resources, such as the included features.",
    )
    parser.add_argument(
        "--use_included_validation",
        action="store_true",
        help="Whether to validate on the included validation dataset. "
        "When specified, the `--resources_dir` must also be specified. "
        "When NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate *predictions* per subject before evaluations. "
        "The predicted probabilities averaged per group."
        "Only the evaluations are affected by this. "
        "**Ignored** when no subjects are present in the meta data.",
    )
    parser.set_defaults(func=main)


def main(args):
    out_path = pathlib.Path(args.out_dir)

    # Create output directory
    paths = IOPaths(
        out_dirs={
            "out_path": out_path,
        }
    )
    paths.mk_output_dirs(collection="out_dirs")

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="validate_model-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running validation of model")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # TODO: training code

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
