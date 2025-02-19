"""
Extracts chromatin track overlaps for each cell type and the coordinates file.

Steps:

 - ...
"""

import os
import argparse
import logging
import pathlib
from typing import Dict, List, Tuple
import concurrent
import pandas as pd
import numpy as np
import scipy.sparse


from utipy import StepTimer, IOPaths, Messenger, random_alphanumeric

# Requires installation of lionheart
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess
from lionheart.utils.bed_ops import (
    get_file_num_lines,
    read_bed_as_df,
    merge_multifile_intervals,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--bam_file",
        type=str,
        help="Path to a **minimal** hg38 BAM file. "
        "We don't need the actual coverage just the bins, "
        "so take a very low-coverage (>= 1 fragment per autosome) file.",
    )
    parser.add_argument("--out_dir", type=str, help="Directory to store output file.")
    parser.add_argument(
        "--chrom_sizes_file",
        type=str,
        help=(
            "Path to file with chromosome sizes. "
            "Only used when `--bin_size != --gc_bin_size`. "
            "Should contain two columns with 1) the name of the chromosome, and "
            "2) the size of the chromosome. Must be tab-separated and have no header."
        ),
    )
    parser.add_argument(
        "--exclusion_bed_files",
        nargs="+",
        required=True,
        type=str,
        help=(
            "Paths to BED files with exclusion intervals, e.g. due to low mappability. "
            "A new exclusion BED file is created with intervals that overlap the prepared BED file. "
        ),
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        help=("Path to 2bit file with reference genome for looking up GC content. "),
    )
    parser.add_argument(
        "--mosdepth_path",
        required=True,
        type=str,
        help=(
            "Path to `mosdepth` application. "
            "Supply something like `'/home/<username>/mosdepth/mosdepth'`."
        ),
    )
    parser.add_argument(
        "--ld_library_path",
        type=str,
        help=(
            "You may need to specify the `LD_LIBRARY_PATH`."
            "\nThis is the path to the `lib` directory in the directory of your "
            "`conda` environment."
            "\nSupply something like `'/home/<username>/anaconda3/envs/<env_name>/lib/'`."
        ),
    )
    parser.add_argument("--bin_size", type=int, default=10, help=("The size of bins."))
    parser.add_argument(
        "--gc_bin_size",
        type=int,
        default=100,
        help=(
            "The size of bins for extracting GC contents around the center of each bin. "
            "This allows adding context when `--bin_size` is low. "
        ),
    )
    parser.add_argument(
        "--num_gc_bins",
        type=int,
        default=100,
        help=(
            "The number of GC content bins. "
            "When not specified the `--bin_size` or `--gc_bins_edges` is used."
            "Only one of `--num_gc_bins` and `--gc_bins_edges` should be specified."
        ),
    )
    parser.add_argument(
        "--excess_method",
        type=str,
        default="remove_end",
        choices=["remove_end", "remove_start", "extend_end", "extend_start"],
        help=(
            "How to handle excess elements when interval sizes aren't divisible by bin size. "
            "Can either remove the excess elements or extend the interval."
        ),
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help=(
            "The number of available CPU cores. Used to parallelize calling of subprocesses."
        ),
    )
    args = parser.parse_args()

    # Prepare logging messenger
    setup_logging(
        dir=str(pathlib.Path(args.out_dir) / "logs"),
        fname_prefix="bin_chromatin_tracks-",
    )
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running binning of chromatin accessibility cell-type tracks")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks
    out_dir = pathlib.Path(args.out_dir)
    tmp_dir = out_dir / f"tmp_{random_alphanumeric(size=15)}"

    chroms = [f"chr{i}" for i in range(1, 23)]

    chrom_out_files = {
        f"{chrom}_out_file": out_dir / f"{chrom}.tsv.gz" for chrom in chroms
    }

    exclusion_paths = {
        f"exclude_{i}": p for i, p in enumerate(args.exclusion_bed_files)
    }

    paths = IOPaths(
        in_files={
            "bam_file": args.bam_file,
            "chrom_sizes_file": args.chrom_sizes_file,
            "reference_file": args.reference_file,
            **exclusion_paths,
        },
        out_dirs={"out_dir": out_dir},
        out_files={
            **chrom_out_files,
            "coordinates_file": out_dir / "bin_coordinates.tsv.gz",
            "gc_bin_edges_file": out_dir / "gc_contents_bin_edges.npy",
            "iss_bin_edges_file": out_dir / "insert_size_bin_edges.npy",
        },
        tmp_files={
            # NOTE: "binned_file" must match output of mosdepth -
            # this defines the mosdepth output directory though
            "binned_file": tmp_dir / "binning.regions.bed.gz",
        },
        tmp_dirs={
            "tmp_dir": tmp_dir,
        },
    )

    # Show overview of the paths
    messenger(paths)

    # Create output directory
    paths.mk_output_dirs(collection="tmp_dirs")
    paths.mk_output_dirs(collection="out_dirs")

    # Load meta data

    # Sort and merge overlapping intervals per file to flatten intervals
    # Using ThreadPoolExecutor to process files concurrently.
    flatten_in_out_args = [
        {"in_file": "track_1.bed", "out_file": "flatteded_track_1.bed"},
    ]
    run_parallel_tasks(
        task_list=flatten_in_out_args,
        worker=sort_and_flatten_track,  # in_file, out_file
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # Merge files per cell-type
    # NOTE: Remember arg order must be as expected by function!
    cell_type__constant_args = {
        "genome_file": paths["chrom_sizes_file"],
        "min_coverage": 0.3,
    }
    merge_cell_types_args = [
        {
            "in_files": [
                "flatteded_track_1.bed",
                "flatteded_track_2.bed",
                "flatteded_track_3.bed",
            ],
            "out_file": "merged_cell_type_track_1.bed",
            **cell_type__constant_args,
        },
    ]
    run_parallel_tasks(
        task_list=merge_cell_types_args,
        worker=merge_by_cell_type,  # in_files, out_file, genome_file, min_coverage
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # Get consensus sites
    # Basically merge all merged cell type files and get those
    # intervals that are present in > 0.9 %
    extract_consensus_sites(
        in_files=[
            "merged_cell_type_track_1.bed",
            "merged_cell_type_track_2.bed",
            "merged_cell_type_track_3.bed",
        ],
        out_file="consensus_sites.bed",
        genome_file=paths["chrom_sizes_file"],
        min_coverage=0.9,
    )

    # Count overlaps between masks and bins

    cell_type__constant_args = {
        "coordinates_file": paths["coordinates_file"],
        "bin_size": 10,
    }
    count_overlaps_args = [
        {
            "overlapping_file": "merged_cell_type_track_1.bed",
            "tmp_counts_file": "tmp_counts_file.bed??",
            "out_files": {"chrXX": "chrXX_overlaps_cell_type1.npz"},
            **cell_type__constant_args,
        },
    ]
    run_parallel_tasks(
        task_list=count_overlaps_args,
        worker=count_overlaps,  # in_files, out_file, genome_file, min_coverage
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # Remove temporary files
    paths.rm_tmp_dirs(messenger=messenger)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def sort_and_flatten_track(in_file: pathlib.Path, out_file: pathlib.Path):
    """
    Sorts and flattens individual interval track.
    """
    merge_multifile_intervals(
        in_files=[in_file],
        out_file=out_file,
        count_coverage=False,
        keep_zero_coverage=False,
        rm_non_autosomes=True,
        pre_sort=True,
    )


def merge_by_cell_type(
    in_files: List[pathlib.Path],
    out_file: pathlib.Path,
    genome_file: pathlib.Path,
    min_coverage: float = 0.3,
):
    merge_multifile_intervals(
        in_files=in_files,
        out_file=out_file,
        count_coverage=True,
        min_coverage=min_coverage,
        rm_non_autosomes=False,  # Already removed
        keep_zero_coverage=False,
        genome_file=genome_file,
    )


def extract_consensus_sites(
    in_files: List[pathlib.Path],
    out_file: pathlib.Path,
    genome_file: pathlib.Path,
    min_coverage: float = 0.9,
):
    merge_multifile_intervals(
        in_files=in_files,
        out_file=out_file,
        count_coverage=True,
        min_coverage=min_coverage,
        rm_non_autosomes=False,  # Already removed
        keep_zero_coverage=False,
        genome_file=genome_file,
    )


def count_overlaps(
    coordinates_file: pathlib.Path,
    overlapping_file: pathlib.Path,
    tmp_counts_file: pathlib.Path,
    out_files: Dict[str, pathlib.Path],
    bin_size: int = 10,
):
    """
    coordinates_file: tsv file with chrom, start, end, idx
    """
    # Assume we can simplify the below?
    # {general_paths["env_path"]}python {general_paths["scripts_path"]}/count_interval_overlaps.py --bed_file_1 {binned_bedfile} --bed_file_2 {path} --out_file {all_bin_overlap_files[version][origin]} --identifier {origin} --as_percentage
    # {general_paths["env_path"]}python {general_paths["scripts_path"]}/split_intervals_by_chromosome.py --bed_file {all_bin_overlap_files[version][origin]} --out_path {bin_overlaps_by_chrom_origin_dir}
    # {general_paths["env_path"]}python {general_paths["scripts_path"]}/check_split_sizes.py --bed_file {all_bin_overlap_files[version][origin]} --split_files {" ".join(to_strings(all_bin_overlap_by_chrom_files[version][origin]))} --out_path {bin_overlaps_by_chrom_origin_dir} --do remove
    # {general_paths["env_path"]}python {general_paths["scripts_path"]}/convert_masks_to_sparse_arrays.py --cell_paths {" ".join(to_strings(input_dirs))} --out_path {out_path} --num_jobs {num_cores}

    # Count overlapping positions
    # For each interval in `in_file`, find the number of overlapping
    # positions in `overlapping_file`.

    check_paths_for_subprocess([coordinates_file, overlapping_file], tmp_counts_file)

    orig_lines = get_file_num_lines(in_file=coordinates_file)

    overlaps_call = " ".join(
        [
            "cat",
            str(coordinates_file),
            "|",
            # Find number of intersecting bps per interval
            # NOTE: Can have multiple lines per interval in a_file (hence the index)
            "|",
            "bedtools intersect",
            "-a",
            "stdin",
            "-b",
            str(overlapping_file),
            "-wao",  # Return coordinates from both files and the overlap count
            # Select chromosome, index and num overlaps
            "|",
            "awk",
            "-F'\t'",
            "-v",
            "OFS='\t'",
            "'{print $1,$4,$8}'",
            ">",
            str(tmp_counts_file),
        ]
    )
    call_subprocess(overlaps_call, "`awk` or `bedtools::intersect` failed")

    # Read as data frame and calculate number of overlaps per interval index
    overlaps_df = (
        read_bed_as_df(path=tmp_counts_file, col_names=["chromosome", "idx", "overlap"])
        .groupby(["chromosome", "idx"])
        .overlap.sum()
        .reset_index()
        .sort_values(["chromosome", "idx"])
        .reset_index(drop=True)
    )

    # Convert to percentage
    overlaps_df["overlap"] /= bin_size

    if not orig_lines == len(overlaps_df):
        raise ValueError(
            f"Original coordinate file had {orig_lines} rows "
            f"but overlap counts file had {len(overlaps_df)} rows."
        )

    # Remove the tmp counts file
    os.remove(str(tmp_counts_file))

    # For each unique chromosome, write a sparse array with the overlap count

    # Save bin indices and GC contents per chromosome
    for chrom_name, group_df in overlaps_df.groupby("chromosome"):
        out_filename = out_files[chrom_name]

        # Select only index and gc columns
        chrom_overlaps = scipy.sparse.csc_matrix(
            group_df.loc[:, ["overlap"]].to_numpy().astype(np.float64)
        )
        save_sparse_array(arr=chrom_overlaps, path=out_filename)


def save_sparse_array(arr, path: pathlib.Path) -> None:
    assert str(path)[-4:] == ".npz"
    scipy.sparse.save_npz(path, arr)


def run_parallel_tasks(task_list, worker, max_workers, messenger):
    """
    Run tasks in parallel using the provided worker function with keyword arguments.

    Parameters
    ----------
    task_list : list of dict
        A list of dictionaries where each dictionary contains keyword arguments for the worker.
    worker : function
        The worker function to run, which should accept keyword arguments.
    max_workers : int, default 4
        Maximum number of parallel worker threads.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each task using keyword arguments.
        futures = {executor.submit(worker, **task): task for task in task_list}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                future.result()
                messenger(f"Task with arguments {task} completed successfully.")
            except Exception as exc:
                messenger(f"Task with arguments {task} failed with exception: {exc}")


def load_metadata(local_paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_data = pd.read_csv(
        local_paths["meta_data"],
        sep="\t",
        usecols=[
            "Sample ID",
            "Reference Version",
            "Experiment ID",
            "Origin",
            "Biosample Type",
            "Data Source",
        ],
    )

    if local_paths["origin_groups"].exists():
        origin_groups_data = pd.read_csv(
            local_paths["origin_groups"],
            usecols=["Biosample.type", "Biosample.term.name", "blood"],
        ).rename(
            columns={
                "Biosample.type": "Biosample type",
                "Origin": "Biosample term name",
            }
        )
    else:
        print(
            f"Did not find origin grouping at {local_paths['origin_groups']}. Skipping."
        )
        origin_groups_data = None

    # Filter meta data
    meta_data = meta_data[meta_data["Reference Version"] != "hg19"]

    return meta_data, origin_groups_data


def get_data_path(id: str, general_paths: Dict[str, str], mask_type: str) -> str:
    return general_paths["mask_paths"][mask_type]["raw_data"] / f"{id}.bed"


def get_sorted_data_path(id: str, general_paths: Dict[str, str], mask_type: str) -> str:
    return general_paths["mask_paths"][mask_type]["sorted_bed_files"][id]


def get_unique_origins(meta_data: pd.DataFrame) -> List[str]:
    return list(meta_data["Origin"].unique())


def standardize_origin(origin):
    return (
        origin.replace(" ", "_")
        .replace("-", "_")
        .replace(",", "_")
        .replace("'", "")
        .replace("/", "")
        .lower()
    )


def set_raw_bedfile_paths(
    general_paths: Dict[str, str], mask_type: str, meta_data: pd.DataFrame
) -> None:
    general_paths["mask_paths"][mask_type]["raw_bed_files"] = [
        get_data_path(id=id, general_paths=general_paths, mask_type=mask_type)
        for id in meta_data["Sample ID"]
    ]


if __name__ == "__main__":
    main()
