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

from concurrent.futures import ProcessPoolExecutor, as_completed
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


def load_zero_cov_indices(path) -> Set[str]:
    """
    Load text file with chromosome-wise indices of zero-coverage bins.

    Parameters
    ----------
    path
        Path to text file (zeros.txt) with rows containing:
            chromosome, original index in chromosome
            for bins with count == 0.

    Returns
    -------
    set
        Set of strings formatted as "<chrom>__<chrom_index>".
    """
    df = read_bed_as_df(
        path=path,
        col_names=["chromosome", "index"],
        messenger=None,
    )
    df["key"] = df["chromosome"] + "__" + df["index"].astype(str)
    return set(df["key"])


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
    for chrom, group in chrom_to_index_df.groupby("chromosome", sort=False):
        chrom_to_indices[chrom] = np.sort(group["index"].to_numpy())

    return chrom_to_indices


def intersect_zero_paths(chunk):
    # Compute the intersection for a given chunk of file paths
    # Note: We pass the global 'messenger' so that each file read
    # reports progress. If that's not desired or causes issues,
    # you can remove it from here
    result = None
    for path in chunk:
        zeros_set = load_zero_cov_indices(path)
        if result is None:
            result = zeros_set
        else:
            result = result.intersection(zeros_set)
    return result


def chunk_zero_paths(paths, n_chunks: int):
    # Compute chunk size so the list of paths is split into roughly equal parts
    chunk_size = int(np.ceil(len(paths) / n_chunks))

    # Create the list of chunks
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]


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
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=5,
        help="Number of chunks to run intersections in for zero-coverage indices. "
        "Beware of memory usage with higher settings.",
    )
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

    # Chunk zero-coverage paths to parallelize intersections
    chunks = chunk_zero_paths(
        paths=list(zero_cov_paths.values()),
        n_chunks=args.n_chunks,
    )

    # Process each chunk in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.n_chunks) as executor:
        futures = {
            executor.submit(intersect_zero_paths, chunk): chunk for chunk in chunks
        }
        for future in as_completed(futures):
            chunk_result = future.result()
            messenger(f"Processed a chunk with {len(chunk_result)} uniques", indent=2)
            results.append(chunk_result)

    # Final overall intersection across all chunks
    overall_zero_set = None
    for s in results:
        if overall_zero_set is None:
            overall_zero_set = s
        else:
            overall_zero_set = overall_zero_set.intersection(s)

    messenger(f"Found {len(overall_zero_set)} all-zero bins", indent=2)

    # Convert to `chrom -> indices` mapping
    zeros_chrom_to_indices = parse_chrom_index_strings(overall_zero_set)

    messenger("Start: Saving all-zero coverage bins")
    np.savez(paths["zero_coverage_bins_indices"], **zeros_chrom_to_indices)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
