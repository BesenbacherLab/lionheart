"""
Extracts chromatin track overlaps for each cell type and the coordinates file.

Steps:

 - ...
"""

import gc
import os
import argparse
import logging
import pathlib
from typing import Dict, List
import concurrent
import concurrent.futures
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
    subtract_intervals,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--coordinates_file",
        required=True,
        type=str,
        help="Path the coordinates file.",
    )
    parser.add_argument(
        "--tracks_dir",
        required=True,
        type=str,
        help="Directory where chromatin tracks are stored.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Directory to store output file. "
        "Separate sparse arrays are stored per cell-type and chromosome.",
    )
    parser.add_argument(
        "--meta_data_file",
        required=True,
        type=str,
        help="Path to `.tsv` file with meta data for the chromatin tracks. "
        "Must have the column: {'sample_id', 'annotated_biosample_name'}, "
        "where the 'sample_id' matches the file name (`<sample_id>.bed.gz`) "
        "in the `--tracks_dir` directory.",
    )
    parser.add_argument(
        "--chrom_sizes_file",
        required=True,
        type=str,
        help=(
            "Path to file with chromosome sizes. "
            "Should contain two columns with 1) the name of the chromosome, and "
            "2) the size of the chromosome. Must be tab-separated and have no header."
        ),
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=10,
        help="The size of bins.",
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
    track_dir = pathlib.Path(args.tracks_dir)
    tmp_dir = out_dir / f"tmp_{random_alphanumeric(size=15)}"

    paths = IOPaths(
        in_files={
            "coordinates_file": args.coordinates_file,
            "chrom_sizes_file": args.chrom_sizes_file,
            "meta_data_file": args.meta_data_file,
        },
        in_dirs={
            "track_dir": track_dir,
        },
        out_dirs={"out_dir": out_dir},
        out_files={"consensus_intervals_file": out_dir / "consensus_intervals.bed"},
        # tmp_files={
        # },
        tmp_dirs={
            "tmp_dir": tmp_dir,
            "tmp_flattened_dir": tmp_dir / "flattened",
            "tmp_merged_dir": tmp_dir / "merged",
            "tmp_subtracted_dir": tmp_dir / "consensus_subtracted",
            "tmp_overlaps_dir": tmp_dir / "overlaps",
        },
    )

    # Load meta data
    messenger("Start: Reading meta data")
    meta_data = pd.read_csv(paths["meta_data_file"], sep="\t")
    sample_ids = meta_data["sample_id"].tolist()
    cell_types = meta_data["annotated_biosample_name"].tolist()
    cell_type_to_sample_ids = (
        meta_data.groupby("annotated_biosample_name")["sample_id"].apply(list).to_dict()
    )
    del meta_data
    gc.collect()
    messenger(
        f"Got {len(sample_ids)} sample IDs for {len(set(cell_types))} cell types",
        indent=2,
    )
    messenger("Cell type -> Sample IDs:", indent=2)
    messenger(cell_type_to_sample_ids, indent=4)

    messenger("Start: Extracting track file paths")
    track_paths = track_dir.glob("*.bed.gz")
    track_paths = [
        ("track_" + path.name[: -len(".bed.gz")], path)
        for path in track_paths
        if path.name[: -len(".bed.gz")] in sample_ids
    ]
    if len(track_paths) == 0:
        raise RuntimeError(
            "Found no tracks in --tracks_dir that matches sample IDs in --meta_data_file"
        )

    chroms = [f"chr{i}" for i in range(1, 23)]
    cell_chrom_out_files = {
        f"{cell_type}_{chrom}_out_file": out_dir / cell_type / f"{chrom}.npz"
        for cell_type in set(cell_types + ["consensus"])
        for chrom in chroms
    }

    paths.set_paths(paths=dict(track_paths), collection="in_files")
    paths.set_paths(paths=cell_chrom_out_files, collection="out_files")

    # Show overview of the paths
    messenger(paths)

    # Create output directory
    paths.mk_output_dirs(collection="tmp_dirs")
    paths.mk_output_dirs(collection="out_dirs")

    # Load meta data

    def make_flattened_path(paths, sample_id):
        return paths["tmp_flattened_dir"] / (sample_id + ".bed")

    # Sort and merge overlapping intervals per file to flatten intervals
    # Using ThreadPoolExecutor to process files concurrently.
    flatten_in_out_kwargs = [
        {
            "in_file": paths["track_" + sample_id],
            "out_file": make_flattened_path(paths, sample_id),
        }
        for sample_id in sample_ids
    ]
    run_parallel_tasks(
        task_list=flatten_in_out_kwargs,
        worker=sort_and_flatten_track,  # in_file, out_file
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    def make_merged_path(paths, cell_type):
        return paths["tmp_merged_dir"] / (cell_type + ".bed")

    # Merge files per cell-type
    merge_cell_types_kwargs = [
        {
            "in_files": [
                make_flattened_path(paths, sample_id)
                for sample_id in cell_type_sample_ids
            ],
            "out_file": make_merged_path(paths, cell_type),
            "genome_file": paths["chrom_sizes_file"],
            "min_coverage": 0.3,
        }
        for cell_type, cell_type_sample_ids in cell_type_to_sample_ids.items()
    ]
    run_parallel_tasks(
        task_list=merge_cell_types_kwargs,
        worker=merge_by_cell_type,  # in_files, out_file, genome_file, min_coverage
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # Get consensus sites
    # Basically merge all merged cell type files and get those
    # intervals that are present in > 0.9 %
    extract_consensus_sites(
        in_files=[make_merged_path(paths, cell_type) for cell_type in cell_types],
        out_file=paths["consensus_intervals_file"],
        genome_file=paths["chrom_sizes_file"],
        min_coverage=0.9,
    )

    def make_subtracted_path(paths, cell_type):
        return paths["tmp_subtracted_dir"] / (cell_type + ".bed")

    # Subtract consensus sites from all cell types (reduces size of final output files)
    subtract_consensus_kwargs = [
        {
            "in_file": make_merged_path(paths, cell_type),
            "out_file": make_subtracted_path(paths, cell_type),
            "consensus_file": paths["consensus_intervals_file"],
        }
        for cell_type in cell_types
    ]
    run_parallel_tasks(
        task_list=subtract_consensus_kwargs,
        worker=subtract_consensus_from_cell_type,  # in_file, out_file, consensus_file
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # Count overlaps between masks and bins

    # Check number of intervals in original coordinates file
    num_orig_lines = get_file_num_lines(
        in_file=paths["coordinates_file"],
    )

    def make_overlap_counts_path(paths, cell_type):
        return paths["tmp_overlaps_dir"] / (cell_type + ".bed")

    find_overlaps_kwargs = [
        {
            "coordinates_file": paths["coordinates_file"],
            "overlapping_file": make_subtracted_path(paths, cell_type),
            "overlap_counts_file": make_overlap_counts_path(paths, cell_type),
        }
        for cell_type in cell_types
    ] + [
        {
            "coordinates_file": paths["coordinates_file"],
            "overlapping_file": paths["consensus_intervals_file"],
            "overlap_counts_file": make_overlap_counts_path(paths, "consensus"),
        }
    ]
    run_parallel_tasks(
        task_list=find_overlaps_kwargs,
        worker=find_overlaps,
        max_workers=args.num_jobs,
        messenger=messenger,
    )

    # NOTE: Requires loading into RAM so run serially

    for cell_type in cell_types + ["consensus"]:
        chrom_out_files = {
            chrom: paths[f"{cell_type}_{chrom}_out_file"] for chrom in chroms
        }

        sparsify_overlap_percentages(
            coordinates_file=paths["coordinates_file"],
            overlap_counts_file=make_overlap_counts_path(paths, cell_type),
            out_files=chrom_out_files,
            num_orig_lines=num_orig_lines,
            bin_size=args.bin_size,
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


def subtract_consensus_from_cell_type(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    consensus_file: pathlib.Path,
):
    subtract_intervals(
        in_file=in_file,
        out_file=out_file,
        exclude_file=consensus_file,
        rm_full_if_any=False,
    )


def find_overlaps(
    coordinates_file: pathlib.Path,
    overlapping_file: pathlib.Path,
    overlap_counts_file: pathlib.Path,
):
    """
    coordinates_file: tsv file with chrom, start, end, idx

    Output
    ------
    File where each original interval idx will be present once *per overlap*.
    """
    # Count overlapping positions
    # For each interval in `in_file`, find the number of overlapping
    # positions in `overlapping_file`.

    check_paths_for_subprocess(
        [coordinates_file, overlapping_file], overlap_counts_file
    )

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
            str(overlap_counts_file),
        ]
    )
    call_subprocess(overlaps_call, "`awk` or `bedtools::intersect` failed")


def sparsify_overlap_percentages(
    overlap_counts_file: pathlib.Path,
    chrom_out_files: Dict[str, pathlib.Path],
    num_orig_lines: int,
    bin_size: int = 10,
):
    # Read as data frame and calculate number of overlaps per interval index
    overlaps_df = (
        read_bed_as_df(
            path=overlap_counts_file, col_names=["chromosome", "idx", "overlap"]
        )
        .groupby(["chromosome", "idx"])
        .overlap.sum()
        .reset_index()
        .sort_values(["chromosome", "idx"])
        .reset_index(drop=True)
    )

    # Convert to percentage
    overlaps_df["overlap"] /= bin_size

    if not num_orig_lines == len(overlaps_df):
        raise ValueError(
            f"Original coordinate file had {num_orig_lines} rows "
            f"but overlap counts file had {len(overlaps_df)} rows."
        )

    # Remove the tmp counts file
    os.remove(str(overlap_counts_file))

    # For each unique chromosome, write a sparse array with the overlap count

    # Save bin indices and GC contents per chromosome
    for chrom_name, group_df in overlaps_df.groupby("chromosome"):
        out_filename = chrom_out_files[chrom_name]

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


if __name__ == "__main__":
    main()
