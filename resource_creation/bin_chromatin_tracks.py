"""
Extracts chromatin track overlaps for each cell type and the coordinates file.

Steps:

 - ...
"""

import gc
import argparse
import logging
import pathlib
import subprocess
from typing import Dict, List, Optional, Tuple
import concurrent
import concurrent.futures
import pandas as pd
import scipy.sparse
from utipy import StepTimer, IOPaths, Messenger, random_alphanumeric

# Requires installation of lionheart
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.sparse_ops import convert_nonzero_bins_to_sparse_array
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess
from lionheart.utils.bed_ops import (
    get_file_num_lines,
    read_bed_as_df,
    merge_multifile_intervals,
    subtract_intervals,
)

# Get path to directory that contain this script
# as the sparsify overlaps awk script is in the same directory
script_dir = pathlib.Path(__file__).parent.resolve()


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
        "--chrom_num_bins_file",
        required=True,
        type=str,
        help=(
            "Path to file with number of bins per chromosome pre-exclusion."
            "Should contain two columns with 1) the name of the chromosome, and "
            "2) the number of bins in the chromosome (before any exclusions). "
            "Must be tab-separated and have no header."
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
            "chrom_num_bins_file": args.chrom_num_bins_file,
            "meta_data_file": args.meta_data_file,
        },
        in_dirs={
            "track_dir": track_dir,
        },
        out_dirs={
            "out_dir": out_dir,
            "sparse_overlaps_by_chromosome": out_dir / "sparse_overlaps_by_chromosome",
        },
        out_files={
            "consensus_intervals_file": out_dir / "consensus_intervals.bed",
            "chrom_cell_paths": out_dir / "binned_per_chrom_and_cell_type_paths.tsv",
            "overlap_stats": out_dir / "overlap_counts.tsv",
            "idx_to_cell_type_file": out_dir / "idx_to_cell_type.csv",
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
        f"{cell_type}_out_dir": out_dir / "sparse_overlaps_by_chromosome" / cell_type
        for cell_type in set(unique_cell_types + ["consensus"])
    }
    cell_chrom_out_files = {
        f"{cell_type}_{chrom}_out_file": cell_type_out_dirs[f"{cell_type}_out_dir"]
        / f"{chrom}.npz"
        for cell_type in set(unique_cell_types + ["consensus"])
        for chrom in chroms
    }

    cell_overlap_tmp_dirs = {
        f"tmp_{cell_type}_overlap_out_dir": paths["tmp_overlaps_dir"] / cell_type
        for cell_type in set(unique_cell_types + ["consensus"])
    }

    paths.set_paths(paths=dict(track_paths), collection="in_files")
    paths.set_paths(paths=cell_type_out_dirs, collection="out_dirs")
    paths.set_paths(paths=cell_chrom_out_files, collection="out_files")
    paths.set_paths(paths=cell_overlap_tmp_dirs, collection="tmp_dirs")

    # Show overview of the paths
    messenger(paths)

    # Create output directory
    paths.mk_output_dirs(collection="tmp_dirs")
    paths.mk_output_dirs(collection="out_dirs")

    # Get the number of bins per chromosome prior to any exclusions
    # (So what comes out of mosdepth)
    chrom_to_num_original_bins: Dict[str, int] = (
        pd.read_csv(
            paths["chrom_num_bins_file"],
            header=None,
            sep="\t",
            names=["chromosome", "num_intervals"],
        )
        .set_index("chromosome")["num_intervals"]
        .to_dict()
    )

    # Flatten each track file individually

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

        def make_overlap_dir_path(paths, cell_type):
            return paths[f"tmp_{cell_type}_overlap_out_dir"]

        find_overlaps_kwargs = [
            {
                "coordinates_file": paths["coordinates_file"],
                "overlapping_file": make_subtracted_path(paths, cell_type),
                "initial_sparse_overlaps_dir": make_overlap_dir_path(paths, cell_type),
                "coordinates_file_has_header": coordinates_file_has_header,
            }
            for cell_type in unique_cell_types
        ] + [
            {
                "coordinates_file": paths["coordinates_file"],
                "overlapping_file": paths["consensus_intervals_file"],
                "initial_sparse_overlaps_dir": make_overlap_dir_path(
                    paths, "consensus"
                ),
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

    messenger("Start: Converting overlap percentages to sparse arrays")
    with timer.time_step(indent=2):
        num_positive_overlaps = {}

        chrom_cell_out_path_dicts = {}
        for cell_type in unique_cell_types + ["consensus"]:
            chrom_cell_out_path_dicts[cell_type] = {
                chrom: paths[f"{cell_type}_{chrom}_out_file"] for chrom in chroms
            }

        sparsify_kwargs = [
            {
                "initial_sparse_overlaps_dir": make_overlap_dir_path(paths, cell_type),
                "chrom_out_files": chrom_cell_out_path_dicts[cell_type],
                "chrom_to_num_chrom_bins": chrom_to_num_original_bins,
                "num_positive_overlaps": num_positive_overlaps,
                "bin_size": args.bin_size,
                "cell_type": cell_type,
            }
            for cell_type in unique_cell_types + ["consensus"]
        ]

        messenger("Running conversion to sparse arrays", indent=2)
        with timer.time_step(indent=4):
            run_parallel_tasks(
                task_list=sparsify_kwargs,
                worker=convert_to_sparse_arrays,
                max_workers=args.num_jobs,
                messenger=messenger,
                extra_verbose=args.extra_verbose,
            )

    messenger("Number of overlapping bins per cell-type:")
    overlap_stats_df, num_overlap_message = prepare_overlap_stats(
        num_positive_overlaps=num_positive_overlaps
    )
    messenger(num_overlap_message)
    overlap_stats_df.to_csv(paths["overlap_stats"], index=False)

    # Create map of (index -> cell type)
    messenger("Start: Saving index --> cell type map for array ordering")
    idx_to_cell_type = pd.DataFrame(
        enumerate(sorted(set(unique_cell_types + ["consensus"]))),
        columns=["idx", "cell_type"],
    )
    idx_to_cell_type.to_csv(
        paths["idx_to_cell_type_file"],
        index=False,
        sep=",",
    )

    # Write paths to tsv file
    messenger("Start: Saving paths to output files")
    paths_df = (
        pd.DataFrame(chrom_cell_out_path_dicts)
        .reset_index()
        .rename(columns={"index": "Chromosome"})
    )
    paths_df.to_csv(paths["chrom_cell_paths"], index=False, sep="\t")

    # Remove temporary files
    messenger("Start: Removing temporary directories", indent=2)
    paths.rm_tmp_dirs(messenger=messenger, rm_paths=False)

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
        ).loc[:, ["chromosome", "idx"]]
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
    initial_sparse_overlaps_dir: List[pathlib.Path],
    coordinates_file_has_header: bool,
):
    """
    coordinates_file: tsv file with chrom, start, end, idx

    Output
    ------
    Directory with .txt files with sparse overlaps (original bin index and overlap) per chromosome.
    Allows creating scipy.sparse arrays manually in a later step (assuming we know
    the full number of bins per chromosome).
    """
    # Count overlapping positions
    # For each interval in `in_file`, find the number of overlapping
    # positions in `overlapping_file`

    check_paths_for_subprocess([coordinates_file, overlapping_file])

    # Skip first line when file has header
    cat_fn = "tail -n +2" if coordinates_file_has_header else "cat"

    overlaps_call = " ".join(
        [
            "MAWK=$(command -v mawk >/dev/null 2>&1 && echo mawk || echo awk);",
            "LC_ALL=C",  # Possible speedup for mawk in the end (not sure if it actually has an effect)
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
            "|",
            # Call mawk sparsification script that saves indices and values of non-zero overlaps
            # Note: This handles when original intervals are present >1 time due to multiple overlaps
            f"$MAWK -v outdir='{initial_sparse_overlaps_dir}' -v chr_col=1 -v bin_col=4 -v overlap_col=8 -f {script_dir / 'sparsify_overlaps.awk'}",
        ]
    )
    call_subprocess(overlaps_call, "`mawk` or `bedtools::intersect` failed")


def convert_to_sparse_arrays(
    initial_sparse_overlaps_dir: pathlib.Path,
    chrom_out_files: Dict[str, pathlib.Path],
    chrom_to_num_chrom_bins: Dict[str, int],
    cell_type: str,
    num_positive_overlaps: dict,
    bin_size: int,
):
    chromosomes = list(chrom_out_files.keys())
    initial_sparse_array_paths = {
        chrom: initial_sparse_overlaps_dir / f"{chrom}.sparsed.txt"
        for chrom in chromosomes
    }

    total_nonzeros = 0
    total_sum = 0

    for chrom in chromosomes:
        overlap_nonzeros, overlap_sum = convert_nonzero_bins_to_sparse_array(
            num_bins=chrom_to_num_chrom_bins[chrom],
            scaling_constant=bin_size,
            input_path=initial_sparse_array_paths[chrom],
            output_path=chrom_out_files[chrom],
            array_type="csc",
        )
        total_nonzeros += overlap_nonzeros
        total_sum += overlap_sum

    num_positive_overlaps[cell_type] = (
        int(total_nonzeros),  # Num bins
        int(total_sum),  # Num bases
    )


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
