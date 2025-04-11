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
import subprocess
from typing import Dict, List, Optional, Tuple
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
    parser.add_argument(
        "--extra_verbose",
        action="store_true",
        help="Whether to log each subprocess task (e.g., `bedtools` calls).",
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
        out_files={
            "consensus_intervals_file": out_dir / "consensus_intervals.bed",
            "chrom_cell_paths": out_dir / "binned_per_chrom_and_cell_type_paths.tsv",
        },
        # tmp_files={
        # },
        tmp_dirs={
            "tmp_dir": tmp_dir,
            "tmp_flattened_dir": tmp_dir / "flattened",
            "tmp_merged_dir": tmp_dir / "merged",
            "tmp_subtracted_dir": tmp_dir / "consensus_subtracted",
            "tmp_overlaps_dir": tmp_dir / "overlaps",
        },
        print_note="Paths to the sparse arrays per cell-type and chromosome are created on the fly.",
    )

    # Load meta data
    messenger("Start: Reading meta data")
    meta_data = pd.read_csv(paths["meta_data_file"], sep="\t")
    sample_ids = meta_data["sample_id"].tolist()
    unique_cell_types = list(set(meta_data["annotated_biosample_name"].tolist()))
    cell_type_to_sample_ids = (
        meta_data.groupby("annotated_biosample_name")["sample_id"].apply(list).to_dict()
    )
    del meta_data
    gc.collect()
    messenger(
        f"Got {len(sample_ids)} sample IDs for {len(unique_cell_types)} cell types",
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
    cell_type_out_dirs = {
        f"{cell_type}_out_dir": out_dir / cell_type
        for cell_type in set(unique_cell_types + ["consensus"])
    }
    cell_chrom_out_files = {
        f"{cell_type}_{chrom}_out_file": out_dir / cell_type / f"{chrom}.npz"
        for cell_type in set(unique_cell_types + ["consensus"])
        for chrom in chroms
    }

    paths.set_paths(paths=dict(track_paths), collection="in_files")
    paths.set_paths(paths=cell_type_out_dirs, collection="out_dirs")
    paths.set_paths(paths=cell_chrom_out_files, collection="out_files")

    # Show overview of the paths
    messenger(paths)

    # Create output directory
    paths.mk_output_dirs(collection="tmp_dirs")
    paths.mk_output_dirs(collection="out_dirs")

    # Load meta data

    def make_flattened_path(paths, sample_id):
        return paths["tmp_flattened_dir"] / (sample_id + ".bed")

    messenger("Start: Flattening individual files")
    with timer.time_step(indent=2):
        # Sort and merge overlapping intervals per file to flatten intervals
        # Using ThreadPoolExecutor to process files concurrently.
        flatten_in_out_kwargs = [
            {
                "in_file": paths["track_" + sample_id],
                "out_file": make_flattened_path(paths, sample_id),
                "use_n_cols": 3,
            }
            for sample_id in sample_ids
        ]
        run_parallel_tasks(
            task_list=flatten_in_out_kwargs,
            worker=sort_and_flatten_track,  # in_file, out_file
            max_workers=args.num_jobs,
            messenger=messenger,
            extra_verbose=args.extra_verbose,
        )

    def make_merged_path(paths, cell_type):
        return paths["tmp_merged_dir"] / (cell_type + ".bed")

    messenger("Start: Merging files per cell type")
    with timer.time_step(indent=2):
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
            extra_verbose=args.extra_verbose,
        )

    messenger("Start: Extracting consensus intervals")
    with timer.time_step(indent=2):
        # Get consensus sites
        # Basically merge all merged cell type files and get those
        # intervals that are present in > 0.9 %
        extract_consensus_sites(
            in_files=[
                make_merged_path(paths, cell_type) for cell_type in unique_cell_types
            ],
            out_file=paths["consensus_intervals_file"],
            genome_file=paths["chrom_sizes_file"],
            min_coverage=0.9,
        )

    def make_subtracted_path(paths, cell_type):
        return paths["tmp_subtracted_dir"] / (cell_type + ".bed")

    messenger("Start: Subtracting consensus intervals from cell type files")
    with timer.time_step(indent=2):
        # Subtract consensus sites from all cell types (reduces size of final output files)
        subtract_consensus_kwargs = [
            {
                "in_file": make_merged_path(paths, cell_type),
                "out_file": make_subtracted_path(paths, cell_type),
                "consensus_file": paths["consensus_intervals_file"],
            }
            for cell_type in unique_cell_types
        ]
        run_parallel_tasks(
            task_list=subtract_consensus_kwargs,
            worker=subtract_consensus_from_cell_type,  # in_file, out_file, consensus_file
            max_workers=args.num_jobs,
            messenger=messenger,
            extra_verbose=args.extra_verbose,
        )

    # Count overlaps between masks and bins

    messenger(f"Start: Counting cell type overlaps per {args.bin_size}bp bin")
    with timer.time_step(indent=2):
        # Check number of intervals in original coordinates file
        num_orig_lines = get_file_num_lines(
            in_file=paths["coordinates_file"],
        )
        coordinates_file_has_header = check_coordinates_file_has_header(
            paths["coordinates_file"]
        )
        if coordinates_file_has_header:
            messenger("`--coordinates_file` has header - adjusting to it", indent=2)
            # Don't count the header
            num_orig_lines -= 1

        def make_overlap_counts_path(paths, cell_type):
            return paths["tmp_overlaps_dir"] / (cell_type + ".bed")

        find_overlaps_kwargs = [
            {
                "coordinates_file": paths["coordinates_file"],
                "overlapping_file": make_subtracted_path(paths, cell_type),
                "overlap_counts_file": make_overlap_counts_path(paths, cell_type),
                "coordinates_file_has_header": coordinates_file_has_header,
            }
            for cell_type in unique_cell_types
        ] + [
            {
                "coordinates_file": paths["coordinates_file"],
                "overlapping_file": paths["consensus_intervals_file"],
                "overlap_counts_file": make_overlap_counts_path(paths, "consensus"),
                "coordinates_file_has_header": coordinates_file_has_header,
            }
        ]
        run_parallel_tasks(
            task_list=find_overlaps_kwargs,
            worker=find_overlaps,
            max_workers=args.num_jobs,
            messenger=messenger,
            extra_verbose=args.extra_verbose,
        )

    # NOTE: Requires loading into RAM so run serially
    messenger("Start: Sparsifying and saving overlap percentages")
    with timer.time_step(indent=2):
        messenger("Reading bin indices", indent=2)
        with timer.time_step(indent=4):
            # Read in bin indices for joining the sparse overlap counts onto
            # so we get the right indexing in the sparse arrays
            bin_indices_df = load_indices_file(
                coordinates_file=paths["coordinates_file"]
            )
        messenger("Splitting bin indices per chromosome", indent=2)
        with timer.time_step(indent=4):
            # Pre-split the bin indices once
            chrom_to_bin_indices = {
                chrom: group
                for chrom, group in bin_indices_df.groupby("chromosome", sort=False)
            }

        num_positive_overlaps = {}

        chrom_cell_out_path_dicts = {}
        for cell_type in unique_cell_types + ["consensus"]:
            chrom_cell_out_path_dicts[cell_type] = {
                chrom: paths[f"{cell_type}_{chrom}_out_file"] for chrom in chroms
            }

        sparsify_kwargs = [
            {
                "overlap_counts_file": make_overlap_counts_path(paths, cell_type),
                "chrom_out_files": chrom_cell_out_path_dicts[cell_type],
                "chrom_to_bin_indices": chrom_to_bin_indices,
                "bin_size": args.bin_size,
                "cell_type": cell_type,
                "num_positive_overlaps": num_positive_overlaps,
                "messenger": messenger,
            }
            for cell_type in unique_cell_types + ["consensus"]
        ]

        run_parallel_tasks(
            task_list=sparsify_kwargs,
            worker=sparsify_overlap_percentages,
            max_workers=args.num_jobs,
            messenger=messenger,
            extra_verbose=args.extra_verbose,
        )

    messenger("Number of overlapping bins per cell-type:")
    overlap_stats_df, num_overlap_message = prepare_overlap_stats(
        num_positive_overlaps=num_positive_overlaps
    )
    messenger(num_overlap_message)
    overlap_stats_df.to_csv(paths["overlap_counts"], index=False)

    # Write paths to tsv file
    paths_df = (
        pd.DataFrame(chrom_cell_out_path_dicts)
        .reset_index()
        .rename(columns={"index": "Chromosome"})
    )
    paths_df.to_csv(paths["chrom_cell_paths"], index=False, sep="\t")

    # Remove temporary files
    paths.rm_tmp_dirs(messenger=messenger)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def prepare_overlap_stats(
    num_positive_overlaps: Dict[str, Tuple[int, int]],
) -> Tuple[pd.DataFrame, str]:
    num_overlap_strings = [
        f"\t{cell_type}:\t{num_bins} bins, {num_bases} bases"
        for cell_type, (num_bins, num_bases) in num_positive_overlaps.items()
    ]
    num_overlap_strings = sorted(num_overlap_strings, key=lambda s: len(s))
    num_overlap_message = "\n".join(num_overlap_strings)

    overlap_stats_df = pd.DataFrame(
        [
            (cell_type, num_bins, num_bases)
            for cell_type, (num_bins, num_bases) in num_positive_overlaps.items()
        ],
        columns=["Cell Type", "Overlapping Bins", "Overlapping Bases"],
    )

    return overlap_stats_df, num_overlap_message


def load_indices_file(coordinates_file: pathlib.Path) -> pd.DataFrame:
    """
    Loads coordinates file and return chromosome and bin index columns.
    """
    if check_coordinates_file_has_header(coordinates_file):
        raise RuntimeError("Coordinates file had header. Not supported.")
    bin_indices_df = (
        read_bed_as_df(
            path=coordinates_file,
            col_names=["chromosome", "start", "end", "idx"],
        )
        .loc[:, ["chromosome", "idx"]]
        .sort_values("idx")
    ).reset_index(drop=True)
    return bin_indices_df


def sort_and_flatten_track(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    use_n_cols: Optional[int] = None,
):
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
        use_n_cols=use_n_cols,
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
    coordinates_file_has_header: bool,
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

    # Skip first line when file has header
    cat_fn = "tail -n +2" if coordinates_file_has_header else "cat"

    overlaps_call = " ".join(
        [
            cat_fn,
            str(coordinates_file),
            # Find number of intersecting bps per interval
            # NOTE: Can have multiple lines per interval in a_file (hence the index)
            "|",
            "bedtools intersect",
            "-a",
            "stdin",
            "-b",
            str(overlapping_file),
            "-wao",  # Return coordinates from both files and the overlap count
            # Select index and num overlaps of non-zero bins
            "|",
            "awk",
            "-F'\t'",
            "-v",
            "OFS='\t'",
            "'$8>0 {print $4,$8}'",
            ">",
            str(overlap_counts_file),
        ]
    )
    call_subprocess(overlaps_call, "`awk` or `bedtools::intersect` failed")


def sparsify_overlap_percentages(
    overlap_counts_file: pathlib.Path,
    chrom_out_files: Dict[str, pathlib.Path],
    chrom_to_bin_indices: Dict[str, pd.DataFrame],
    cell_type: str,
    num_positive_overlaps: dict,
    bin_size: int,
    messenger,
):
    messenger(f"  {cell_type}")

    # Read as data frame and calculate number of overlaps per interval index
    overlaps_df = (
        read_bed_as_df(path=overlap_counts_file, col_names=["idx", "overlap"])
        .groupby(["idx"])
        .overlap.sum()
        .reset_index()
        .reset_index(drop=True)
    )

    num_positive_overlaps[cell_type] = (
        len(overlaps_df),  # Num bins
        int(overlaps_df["overlap"].sum()),  # Num bases
    )

    # Convert to percentage
    overlaps_df["overlap"] /= bin_size

    if len(overlaps_df) < 100:
        raise ValueError(f"`overlaps_df` only contained {len(overlaps_df)} rows.")

    # For each unique chromosome, write a sparse array with the overlap count
    # We use a fast, memory-efficient "left-join" using .map
    # This should avoid copying the data frame and keep the order of bin_indices_df

    # Create the mapping from index to overlap from the right DataFrame.
    mapping = overlaps_df.set_index("idx")["overlap"]

    # Run per chromosome
    for chrom, group in chrom_to_bin_indices.items():
        # Map the 'idx' column of this group to get the overlaps
        # Fill missing values with 0
        overlaps = group["idx"].map(mapping).fillna(0.0).to_numpy(dtype=np.float64)
        # Create the sparse CSC matrix.
        sparse_overlaps = scipy.sparse.csc_matrix(overlaps)

        out_filename = chrom_out_files[chrom]
        save_sparse_array(arr=sparse_overlaps, path=out_filename)

        del overlaps
        gc.collect()

    # Remove the tmp counts file
    os.remove(str(overlap_counts_file))

    del overlaps_df
    gc.collect()


def save_sparse_array(arr, path: pathlib.Path) -> None:
    assert str(path)[-4:] == ".npz"
    scipy.sparse.save_npz(path, arr)


def run_parallel_tasks(task_list, worker, max_workers, messenger, extra_verbose):
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
                if extra_verbose:
                    messenger(f"Task with arguments {task} completed successfully.")
            except Exception as exc:
                messenger(f"Task with arguments {task} failed with exception: {exc}")
                raise


def check_coordinates_file_has_header(filename):
    # Use subprocess to get the first line of the file
    first_line = subprocess.check_output(
        ["head", "-n", "1", filename], universal_newlines=True
    ).strip()
    fields = first_line.split("\t")

    # Try converting the second field to an integer.
    # In a valid bed-like file, the second column (start coordinate) should be numeric.
    try:
        int(fields[1])
        return False  # No header detected
    except (IndexError, ValueError):
        return True  # Likely a header


if __name__ == "__main__":
    main()
