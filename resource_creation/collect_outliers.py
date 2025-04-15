"""
Script for finding bins with outlier-level coverages across samples (e.g. controls).
"""

import argparse
import gc
import logging
import pathlib
from typing import Dict, List, Set, Union
import numpy as np
import pandas as pd

from utipy import Messenger, StepTimer, IOPaths
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.bed_ops import (
    read_bed_as_df,
)


def load_candidate_df(path, messenger) -> pd.DataFrame:
    """
    Load outlier candidates.

    Parameters
    ----------
    path
        Path to text file (candidates.txt) with rows containing:
           chromosome, original index in chromosome, coverage count, and
           ZIP tail (cdf) probability for bins with count > count_threshold
    """

    df = read_bed_as_df(
        path=path,
        col_names=["chromosome", "index", "coverage", "poisson_prob"],
        messenger=messenger,
    )
    df["key"] = df["chromosome"] + "__" + df["index"].astype(str)
    return df


def load_zero_cov_indices_to_structured_array(path, messenger):
    df = read_bed_as_df(
        path=path,
        col_names=["chromosome", "index"],
        messenger=messenger,
    )

    # Convert chromosome number to integer
    df["chromosome"] = df["chromosome"].str.replace("chr", "").astype(np.int32)
    df["index"] = df["index"].astype(np.int32)

    # Convert to a list of tuples and then to a structured array
    tuples = list(df.itertuples(index=False, name=None))
    dtype = np.dtype([("chr", np.int32), ("index", np.int32)])
    arr = np.array(tuples, dtype=dtype)

    # Sort the array by the tuple keys
    return arr


def parse_chrom_index_strings(s: Union[List[str], Set[str]]) -> Dict[str, np.ndarray]:
    """
    Parse the "<chrom>__<chrom_index>" strings from a list/set
    and produce a dictionary mapping each chromosome to its
    indices.

    Parameters
    ----------
    s
        Set of strings formatted as "<chrom>__<chrom_index>".

    Returns
    -------
    Dict
        Mapping of chromosomes to integer numpy arrays with chromosome-wise indices.
    """
    # Split into columns in data frame
    chrom_to_index_df = pd.DataFrame(
        [tuple(key.split("__")) for key in s],
        columns=["chromosome", "index"],
    )

    # Make index integer
    chrom_to_index_df["index"] = chrom_to_index_df["index"].astype(int)

    # Convert to dict to allow saving in npz file
    chrom_to_indices = {}
    for chrom, group in chrom_to_index_df.groupby(["chromosome"], sort=False):
        chrom_to_indices[chrom] = np.sort(group["index"].to_numpy())

    return chrom_to_indices


def structured_array_to_dict(x):
    """
    Convert a sorted structured array with fields 'chr' and 'index'
    into a dictionary mapping each chromosome to its sorted numpy array of indices.

    Parameters
    ----------
    x : np.ndarray
        Structured numpy array with dtype [('chr', np.int32), ('index', np.int32)].

    Returns
    -------
    dict
        Dictionary where keys are chromosomes and values are numpy arrays of indices.
    """
    result = {}
    if x.size == 0:
        return result

    # Find where the chromosome changes; assumes common is sorted by 'chr'
    unique_chroms, start_idx = np.unique(x["chr"], return_index=True)
    # Iterate over each unique chromosome
    for i, chrom in enumerate(unique_chroms):
        start = start_idx[i]
        end = start_idx[i + 1] if i + 1 < len(start_idx) else x.size
        # You can return as int arrays
        result[f"chr{chrom}"] = np.asarray(x["index"][start:end], dtype=np.int64)
    return result


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
            "outlier_indices": out_dir / "outlier_indices.npz",
            "zero_coverage_bins_indices": out_dir / "zero_coverage_indices.npz",
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
        outlier_keys = []

        messenger(f"Total non-unique candidates: {len(candidates_df)}", indent=2)

        for thresh, out_of in zip(thresholds, out_ofs):
            # Get bin indices where the bins have probabilities
            # below or equal to the threshold
            candidate_keys = candidates_df[candidates_df["poisson_prob"] <= thresh][
                "key"
            ]

            # Count the number of samples an index is
            # an outlier candidate in
            candidate_indices, idx_counts = np.unique(
                candidate_keys, return_counts=True
            )

            # Get indices where they are outlier candidates in
            # `out_of` percentage or more of the samples
            keys_for_thresh = list(
                candidate_indices[idx_counts / len(candidate_paths) >= out_of]
            )
            outlier_keys += keys_for_thresh
            messenger(
                f"Threshold {thresh}: {len(keys_for_thresh)} outliers",
                indent=2,
            )

        # Remove duplicate indices
        outlier_keys = set(outlier_keys)

        messenger(f"Unique outliers: {len(outlier_keys)}", indent=2)

    # Convert to `chrom -> indices` mapping
    outliers_chrom_to_indices = parse_chrom_index_strings(outlier_keys)

    messenger("Start: Saving outlier indices")
    np.savez(paths["outlier_indices"], **outliers_chrom_to_indices)

    del (
        outliers_chrom_to_indices,
        outlier_keys,
        keys_for_thresh,
        candidates_df,
        candidate_keys,
    )
    gc.collect()

    ##############################
    #### All-zero coordinates ####
    ##############################

    # All-zero coordinates
    messenger("Start: Finding all-zero coverage bins")

    num_paths = len(zero_cov_paths.values())
    for i, path in enumerate(zero_cov_paths.values()):
        zeros_chrom_index_arr = load_zero_cov_indices_to_structured_array(
            path, messenger=messenger
        )
        if i == 0:
            always_zero_chrom_index_arr = zeros_chrom_index_arr
        else:
            always_zero_chrom_index_arr = np.intersect1d(
                always_zero_chrom_index_arr,
                zeros_chrom_index_arr,
                assume_unique=True,
            )

        del zeros_chrom_index_arr
        if i % 5 == 0:
            messenger(
                f"{i}/{num_paths}: {len(always_zero_chrom_index_arr)} uniques",
                indent=2,
            )
            gc.collect()

    messenger(f"Found {len(always_zero_chrom_index_arr)} all-zero bins", indent=2)

    # Convert to `chrom -> indices` mapping
    zeros_chrom_to_indices = structured_array_to_dict(always_zero_chrom_index_arr)

    messenger("Start: Saving all-zero coverage bins")
    np.savez(paths["outlier_indices"], **zeros_chrom_to_indices)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
