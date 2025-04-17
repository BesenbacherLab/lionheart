"""
Script for collecting outliers across datasets.

"""

import argparse
import logging
import pathlib
from typing import Dict, List
import warnings
import numpy as np

from utipy import Messenger, StepTimer, IOPaths
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.utils import load_chrom_indices


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
        help=("Paths to directories with outliers and zero-coverage bins."),
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
        default="outlier_indices.npz",
        help=("Name of the outlier indices file within each outlier directory."),
    )
    parser.add_argument(
        "--zero_filename",
        type=str,
        default="zero_coverage_indices.npz",
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

    chromosomes = [f"chr{i}" for i in range(1, 23)]

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

    # Load outlier dicts from each dataset
    outlier_index_collections: Dict[str, Dict[str, np.ndarray]] = {
        key: load_chrom_indices(paths[key]) for key in outlier_paths.keys()
    }

    # Check availability of chromosomes per collection
    missing_chromosomes = [
        (key, chrom)
        for key, coll in outlier_index_collections.items()
        for chrom in chromosomes
        if chrom not in coll
    ]
    if missing_chromosomes:
        messenger(
            "These chromosomes were missing from the following "
            f"outlier collections:\n{missing_chromosomes}",
            add_msg_fn=warnings.warn,
        )

    # Combine outliers across datasets per chromosome
    outlier_chrom_to_indices = {
        chrom: combine_arrays(
            arrs=[
                coll.get(chrom, np.array([]))
                for coll in outlier_index_collections.values()
            ],
            method=args.outlier_method,
        )
        for chrom in chromosomes
    }

    messenger("Final outlier counts:", indent=2)
    for chrom in chromosomes:
        messenger(f"{chrom}: {len(outlier_chrom_to_indices[chrom])}", indent=4)

    messenger("Start: Saving outlier indices")
    np.savez(paths["outlier_indices"], **outlier_chrom_to_indices)

    #######################
    #### Zero-coverage ####
    #######################

    messenger(
        f"Start: Combining zero-coverage bin indices using method: {args.zero_method}"
    )

    # Load zero-coverage dicts from each dataset
    always_zero_index_collections: Dict[str, Dict[str, np.ndarray]] = {
        key: load_chrom_indices(paths[key]) for key in zero_paths.keys()
    }

    # Check availability of chromosomes per collection
    missing_chromosomes = [
        (key, chrom)
        for key, coll in outlier_index_collections.items()
        for chrom in chromosomes
        if chrom not in coll
    ]
    if missing_chromosomes:
        messenger(
            "These chromosomes were missing from the following "
            f"zero-coverage collections:\n{missing_chromosomes}",
            add_msg_fn=warnings.warn,
        )

    # Combine zero-coverage indices across datasets per chromosome
    always_zero_chrom_to_indices = {
        chrom: combine_arrays(
            arrs=[
                coll.get(chrom, np.array([]))
                for coll in always_zero_index_collections.values()
            ],
            method=args.zero_method,
        )
        for chrom in chromosomes
    }

    messenger("Final zero-coverage bin counts:", indent=2)
    for chrom in chromosomes:
        messenger(f"{chrom}: {len(always_zero_chrom_to_indices[chrom])}", indent=4)

    messenger("Start: Saving zero-coverage indices")
    np.savez(paths["zero_coverage_indices"], **always_zero_chrom_to_indices)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
