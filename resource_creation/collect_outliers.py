"""
Script for finding bins with outlier-level coverages across samples (e.g. controls).
"""

import argparse
import logging
import pathlib
import numpy as np
import pandas as pd

from utipy import Messenger, StepTimer, IOPaths
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.bed_ops import (
    read_bed_as_df,
)


def load_candidate_df(path, messenger):
    # candidates.txt: row index, count, and ZIP probability for rows with count > count_threshold.
    df = read_bed_as_df(
        path=path,
        col_names=["index", "coverage", "poisson_prob"],
        messenger=messenger,
    )
    return df


def load_zero_cov_indices(path):
    """
    Load text file with indices of zero-coverage bins
    """
    # zeros.txt: 0-indexed row indices (from the filtered rows) with count == 0.
    with open(path, "r") as file:
        indices = [int(line.strip()) for line in file.readlines()]
    return np.array(indices)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--candidate_dirs",
        required=True,
        type=str,
        nargs="*",
        help=(
            "Paths to directories with outlier candidates. "
            "Should contain files with names given by "
            "`--candidates_filename`, `--zero_cov_filename`."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Directory path to store the outliers at. "
            "A `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        default=[10e-5],
        nargs="*",
        help=(
            "Probability threshold(s)."
            "Bins with probabilities lower than or equal to a threshold "
            "in more than its respective `--out_ofs` percent of the samples "
            "are considered outliers. Only bins with coverages greather than "
            "the chromosome's mean coverage are considered."
        ),
    )
    parser.add_argument(
        "--out_ofs",
        type=float,
        default=[0.25],
        nargs="*",
        help=(
            "Percentage(s) of samples that a bin must fall under the respective threshold "
            "(see `--thresholds`) to be considered an outlier bin. "
            "Usually, the percentage of out-ofs would be lower, the lower the threshold is."
        ),
    )
    parser.add_argument("--candidates_filename", type=str, default="candidates.txt")
    parser.add_argument("--zero_cov_filename", type=str, default="zeros.txt")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)

    # Prepare logging messenger
    setup_logging(dir=str(out_dir / "logs"), fname_prefix="collect_outlier_bins-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running collection of outliers")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    def create_file_group_paths(fgroup, filename):
        return {
            f"candidate_{i}_{fgroup}": pathlib.Path(path) / filename
            for i, path in enumerate(args.candidate_dirs)
        }

    candidate_paths = create_file_group_paths("candidates", args.candidates_filename)
    zero_cov_paths = create_file_group_paths("zero_bins", args.zero_cov_filename)

    # Create paths container with checks
    paths = IOPaths(
        in_files={
            **candidate_paths,
            **zero_cov_paths,
        },
        in_dirs={
            f"candidate_dir_{i}": path for i, path in enumerate(args.candidate_dirs)
        },
        out_dirs={
            "out_dir": out_dir,
        },
        out_files={
            "outlier_indices": out_dir / "outlier_indices.npy",
            "zero_coverage_bins_indices": out_dir / "zero_coverage_indices.npy",
        },
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Preparing thresholds and out-of percentages")

    if not len(args.thresholds):
        raise ValueError(
            "At least 1 threshold must be specified. `--thresholds` was empty."
        )
    if len(args.thresholds) != len(args.out_ofs):
        raise ValueError(
            f"`--thresholds` ({len(args.thresholds)}) and `--out_ofs` ({len(args.out_ofs)}) "
            "did not have the same number of elements."
        )

    # Check `out_ofs` are valid
    for i, out_of in enumerate(args.out_ofs):
        if not (0.0 <= out_of <= 1.0):
            raise ValueError(
                f"`--out_ofs` values must be between 0 and 1. Value {i} was {out_of}."
            )

    # Check `thresholds` are valid
    for i, thresh in enumerate(args.thresholds):
        if not (0.0 <= thresh <= 1.0):
            raise ValueError(
                "`--thresholds` values must be between 0 and 1. "
                f"Value {i} was {thresh}."
            )

    # Sort thresholds and out-ofs together
    thresh_order = np.argsort(args.thresholds)
    thresholds = np.asarray(args.thresholds)[thresh_order]
    out_ofs = np.asarray(args.out_ofs)[thresh_order]

    messenger(f"Thresholds: {thresholds}", indent=2)
    messenger(f"Out of percentages: {out_ofs}", indent=2)

    #####################################
    #### Extract outlier coordinates ####
    #####################################

    # Outliers
    messenger("Start: Loading candidates")
    messenger(f"Will load {len(candidate_paths)} candidate paths in total", indent=2)

    # Load all candidate data frames
    # and concatenate them
    with timer.time_step(indent=4, name_prefix="load_candidates"):
        candidates_df = pd.concat(
            [load_candidate_df(path, messenger) for path in candidate_paths.values()],
            ignore_index=True,
        )

    messenger("Start: Extracting outliers for each threshold")

    with timer.time_step(indent=4):
        outlier_indices = []

        messenger(f"Total non-unique candidates: {len(candidates_df)}", indent=2)

        for thresh, out_of in zip(thresholds, out_ofs):
            # Get bin indices where the bins have probabilities
            # below or equal to the threshold
            candidate_indices = (
                candidates_df[candidates_df["poisson_prob"] <= thresh]["index"]
                .to_numpy()
                .flatten()
            )

            # Count the number of samples an index is
            # an outlier candidate in
            candidate_indices, idx_counts = np.unique(
                candidate_indices, return_counts=True
            )

            # Get indices where they are outlier candidates in
            # `out_of` percentage or more of the samples
            indices_for_thresh = list(
                candidate_indices[idx_counts / len(candidate_paths) >= out_of]
            )
            outlier_indices += indices_for_thresh
            messenger(
                f"Threshold {thresh}: {len(indices_for_thresh)} outliers",
                indent=2,
            )

        # Remove duplicate indices
        outlier_indices = np.unique(outlier_indices)

        messenger(f"Unique outliers: {len(outlier_indices)}", indent=2)

    messenger("Start: Saving outlier indices")
    np.save(paths["outlier_indices"], outlier_indices)

    ##############################
    #### All-zero coordinates ####
    ##############################

    # All-zero coordinates
    messenger("Start: Finding all-zero coverage bins")

    # NOTE: This likely takes a lot of memory?
    all_zero_indices = [load_zero_cov_indices(path) for path in zero_cov_paths.values()]

    messenger(
        "Non-unique zero-coverage bins: "
        f"{len(np.concatenate(all_zero_indices).flatten())} ",
        indent=2,
    )

    with timer.time_step(indent=2, name_prefix="zeroes"):
        # Combine all zero-coverage indices across samples
        all_zero_indices = np.concatenate(all_zero_indices).flatten()

        # Count how many times each index is present
        unique_indices, index_counts = np.unique(all_zero_indices, return_counts=True)

        # Find the indices that have zero-coverage in all bins
        always_zero_indices = unique_indices[
            np.argwhere(index_counts == len(all_zero_indices)).flatten()
        ]

        messenger(f"Found {len(always_zero_indices)} all-zero bins", indent=2)

    messenger("Start: Saving all-zero coverage bins")
    np.save(paths["zero_coverage_bins_indices"], always_zero_indices)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
