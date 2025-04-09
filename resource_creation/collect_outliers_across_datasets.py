"""
Script for collecting outliers across datasets.

"""

import argparse
import logging
import pathlib
from typing import List
import numpy as np

from utipy import Messenger, StepTimer, IOPaths
from lionheart.utils.dual_log import setup_logging


def combine_arrays(arrs: List[np.ndarray], method: str) -> np.ndarray:
    assert method in ["union", "intersection"]
    candidate_indices = np.concatenate(arrs).flatten()
    if method == "intersection":
        # Count the number of datasets an index is
        # an outlier candidate in
        unique_indices, idx_counts = np.unique(candidate_indices, return_counts=True)

        # Find the indices that are outliers in all datasets
        return unique_indices[np.argwhere(idx_counts == len(arrs)).flatten()]

    else:
        return np.unique(candidate_indices)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--outlier_dirs",
        required=True,
        type=str,
        nargs="*",
        help=("Paths to directories outliers and zero-coverage bins."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Directory path to store the combined outliers and zero-coverage bins at. "
            "A `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help=("How to combine the outliers across datasets."),
    )
    parser.add_argument(
        "--zero_method",
        type=str,
        default="intersection",
        choices=["union", "intersection"],
        help=("How to combine the zero-coverage bins across datasets."),
    )
    parser.add_argument(
        "--outlier_filename",
        type=str,
        default="outlier_indices.npy",
        help=("Name of the outlier indices file within each outlier directory."),
    )
    parser.add_argument(
        "--zero_filename",
        type=str,
        default="zero_coverage_indices.npy",
        help=("Name of the zero-coverage indices file within each outlier directory."),
    )

    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)

    # Prepare logging messenger
    setup_logging(
        dir=str(out_dir / "logs"), fname_prefix="collect_outliers_across_datasets-"
    )
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running collection of outliers across datasets")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks

    outlier_paths = {
        f"outlier_path_{i}": pathlib.Path(path) / args.outlier_filename
        for i, path in enumerate(args.outlier_dirs)
    }
    zero_paths = {
        f"zero_path_{i}": pathlib.Path(path) / args.zero_filename
        for i, path in enumerate(args.outlier_dirs)
    }
    paths = IOPaths(
        in_files={
            **outlier_paths,
            **zero_paths,
        },
        in_dirs={f"outlier_dir_{i}": path for i, path in enumerate(args.outlier_dirs)},
        out_dirs={
            "out_dir": out_dir,
        },
        out_files={
            "outlier_indices": out_dir / "outlier_indices.npz",
            "zero_coverage_indices": out_dir / "zero_coverage_indices.npz",
        },
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")

    # Show overview of the paths
    messenger(paths)

    ##################
    #### Outliers ####
    ##################

    messenger(
        f"Start: Combining outlier bin indices using method: {args.outlier_method}"
    )

    # Combine outliers across datasets
    outlier_indices = combine_arrays(
        arrs=[np.load(paths[key]) for key in outlier_paths.keys()],
        method=args.outlier_method,
    )
    messenger(f"Final outlier indices: {len(outlier_indices)}", indent=2)

    messenger("Start: Saving outlier indices")
    np.save(paths["outlier_indices"], outlier_indices)

    #######################
    #### Zero-coverage ####
    #######################

    messenger(
        f"Start: Combining zero-coverage bin indices using method: {args.zero_method}"
    )

    # Combine zero-coverage indices across datasets
    zero_indices = combine_arrays(
        arrs=[np.load(paths[key]) for key in zero_paths.keys()],
        method=args.zero_method,
    )
    messenger(f"Final zero-coverage indices: {len(zero_indices)}", indent=2)

    messenger("Start: Saving zero-coverage indices")
    np.save(paths["zero_coverage_indices"], zero_indices)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
