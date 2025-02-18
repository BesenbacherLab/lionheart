"""
Creates a BED file of the mappable areas of the whole genome with information about the GC context.

Steps:

 - Bin hg38 with mosdepth
 - Find indices of bins to include/exclude using mappability filtering per chromosome
 - Extract GC contents of remaining bins in a 100bp context
 - Create GC bin edges for correction
 - Create average overlapping insert size bin edges for correction
 - Split intervals into chromosome-wise files with indices and GC contents
"""

import gc
import os
import argparse
import logging
import pathlib
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import py2bit


from utipy import StepTimer, IOPaths, Messenger, random_alphanumeric

# Requires installation of lionheart
from lionheart.commands.extract_features import MosdepthPaths
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess
from lionheart.utils.bed_ops import (
    ensure_col_types,
    read_bed_as_df,
    merge_multifile_intervals,
)
from lionheart.utils.gc_content import (
    get_gc_content_all_intervals,
    find_greedy_bin_edges,
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
    parser.add_argument("--out_path", type=str, help="Directory to store output file.")
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
    args = parser.parse_args()

    if args.exclusion_bed_files is not None:
        for bedf in args.exclusion_bed_files:
            assert os.path.isfile(bedf), (
                f"exclusion bed file path was not a file: {bedf}"
            )
    if args.bin_size < 1:
        raise ValueError("`--bin_size` must be a positive number.")
    if args.gc_bin_size < 1:
        raise ValueError("`--gc_bin_size` must be a positive number.")

    # Prepare logging messenger
    setup_logging(
        dir=str(pathlib.Path(args.out_path) / "logs"),
        fname_prefix="bin_genome-",
    )
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running creation of filtered whole genome BED file")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks
    out_path = pathlib.Path(args.out_path)
    tmp_dir = out_path / "tmp"
    mosdepth_tmp_dir = tmp_dir / "tmp_mosdepth"

    chroms = [f"chr{i}" for i in range(1, 23)]

    chrom_out_files = {
        f"{chrom}_out_file": out_path / f"{chrom}.tsv.gz" for chrom in chroms
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
        out_dirs={"out_path": out_path},
        out_files={
            **chrom_out_files,
            "gc_bin_edges_path": out_path / "gc_contents_bin_edges.npy",
            "iss_bin_edges_path": out_path / "insert_size_bin_edges.npy",
        },
        tmp_files={
            # NOTE: "binned_file" must match output of mosdepth -
            # this defines the mosdepth output directory though
            "binned_file": mosdepth_tmp_dir / "binning.regions.bed.gz",
            "merged_exclusion_file": tmp_dir
            / f"tmp_merged_exclusion_file.{random_alphanumeric(size=15)}.bed",
            "filtered_intervals_file": tmp_dir
            / f"tmp_filtered_intervals_file.{random_alphanumeric(size=15)}.bed",
        },
        tmp_dirs={
            "tmp_dir": tmp_dir,
            "mosdepth_dir": mosdepth_tmp_dir,
        },
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")
    paths.mk_output_dirs(collection="tmp_dirs")

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Loading chromosome sizes file")
    with timer.time_step(indent=2):
        chrom_sizes_map = _load_chrom_sizes_file(
            chrom_sizes_file=paths["chrom_sizes_file"]
        )
        messenger("Chromosome sizes:", indent=2)
        messenger(f"{chrom_sizes_map}", indent=4)

    # Merge to a single exclusion file
    messenger("Start: Merging the exclusion BED file(s)")
    with timer.time_step(indent=2):
        merge_multifile_intervals(
            in_files=[paths[key] for key in exclusion_paths.keys()],
            out_file=paths["merged_exclusion_file"],
            sort_numerically=False,
        )

    # Run mosdepth for 10bp bins
    # NOTE: All later BAM files will be binned by mosdepth, so
    # we use mosdepth to ensure same coordinates
    messenger("Start: Calling mosdepth to bin genome")
    with timer.time_step(indent=2):
        _call_mosdepth(
            in_file=paths["bam_file"],
            out_file=paths[
                "binned_file"
            ],  # NOTE: only parent dir is specified via this arg
            bin_size=args.bin_size,
            mosdepth_paths=MosdepthPaths(
                ld_lib_path=args.ld_library_path,
                mosdepth_path=args.mosdepth_path,
            ),
        )

    messenger("Start: Filtering invervals by exclusion files")
    with timer.time_step(indent=2):
        _exclude_bins(
            in_file=paths["binned_file"],
            out_file=paths["filtered_intervals_file"],
            exclude_file=paths["merged_exclusion_file"],
        )

    messenger("Start: Loading BED file into data frame")
    with timer.time_step(indent=2):
        bins_df = read_bed_as_df(
            paths["filtered_intervals_file"],
            col_names=["chromosome", "start", "end", "idx"],
            messenger=messenger,
        )
        messenger(f"Shape: {bins_df.shape}", indent=2)

    messenger("Start: Extracting GC contents")
    with timer.time_step(indent=2, message="GC extraction took: "):
        bins_df, gc_bin_edges = _extract_gc_contents(
            bins_df=bins_df,
            bin_size=args.bin_size,
            gc_bin_size=args.gc_bin_size,
            num_gc_bins=args.num_gc_bins,
            gc_ignore_non_acgt=True,
            chrom_sizes_map=chrom_sizes_map,
            paths=paths,
            timer=timer,
            messenger=messenger,
        )
    messenger(f"Shape: {bins_df.shape}", indent=2)

    # Save bin edges
    messenger("Saving GC bin edges", indent=2)
    np.save(paths["gc_bin_edges_path"], gc_bin_edges)

    messenger("Start: Creating insert size bin edges")
    iss_bin_edges = _create_insert_size_bin_edges(
        start=100,
        stop=220,
        num_bins=40,
    )
    messenger("Saving insert size bin edges", indent=2)
    np.save(paths["iss_bin_edges_path"], iss_bin_edges)

    # For each unique chromosome, write a file with only 'index' and 'gc'
    messenger("Start: Saving bin indices and GC contents per chromosome")
    with timer.time_step(indent=2):
        bins_df = ensure_col_types(bins_df, dtypes={"idx": "int32", "GC": "float32"})
        for chrom_name, group_df in bins_df.groupby("chromosome"):
            out_filename = paths[f"{chrom_name}_out_file"]

            # Select only index and gc columns
            subset = group_df.loc[:, ["idx", "GC"]]

            # Write chromosome bin indices and GC contents to tsv
            subset.to_csv(
                out_filename,
                sep="\t",
                index=False,
                header=True,
                compression="gzip",
            )
            messenger(
                f"Saved file to {out_filename}",
                indent=2,
            )
            messenger(f"With shape: {subset.shape}", indent=4)

    # Remove temporary files
    paths.rm_tmp_dirs(messenger=messenger)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def _load_chrom_sizes_file(chrom_sizes_file: pathlib.Path) -> Dict[str, int]:
    chrom_sizes = pd.read_csv(chrom_sizes_file, sep="\t", header=None)
    chrom_sizes.columns = ["chromosome", "chrom_size"]
    unique_chroms = [f"chr{i}" for i in range(1, 23)]
    chrom_sizes_map = {
        chrom: size
        for chrom, size in zip(chrom_sizes.chromosome, chrom_sizes.chrom_size)
        if chrom in unique_chroms
    }
    return chrom_sizes_map


def _exclude_bins(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    exclude_file: pathlib.Path,
) -> pathlib.Path:
    check_paths_for_subprocess([in_file, exclude_file], out_file)

    # Start by indexing per chromosome, keeping only autosomes
    # Then exclude any bins that overlap with exclude file intervals
    index_and_subtract_cmd = f"""
zcat {in_file} | awk -F'\t' -v OFS='\t' '
BEGIN {{
  prev = "";
  i = 0
}}
$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {{
  if ($1 != prev) {{
    i = 1;
    prev = $1
  }} else {{
    i++;
  }}
  # Output: chrom, start, end, new_index
  print $1, $2, $3, i
}}' | bedtools subtract -a - -b {exclude_file} -A | gzip > {out_file}
"""

    call_subprocess(
        index_and_subtract_cmd,
        "`awk` indexing or `bedtool subtract` failed",
    )

    return out_file


def _call_mosdepth(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    bin_size: int,
    mosdepth_paths: Optional[MosdepthPaths] = None,
) -> pathlib.Path:
    out_dir = out_file.parent
    mosdepth_reference = "mosdepth" if mosdepth_paths is None else str(mosdepth_paths)

    check_paths_for_subprocess(in_file, out_dir)
    mosdepth_call = " ".join(
        [
            "cd",
            str(out_dir),
            ";",
            f"{mosdepth_reference}",
            f"--by {bin_size}",
            "--threads 4",
            "--no-per-base",
            f"{out_dir / 'binning'}",  # Output prefix
            str(in_file),
        ]
    )
    call_subprocess(mosdepth_call, "`mosdepth` failed")

    return out_file


def _extract_gc_contents(
    bins_df: pd.DataFrame,
    bin_size: int,
    gc_bin_size: int,
    num_gc_bins: int,
    gc_ignore_non_acgt: bool,
    chrom_sizes_map: Dict[str, int],
    paths: IOPaths,
    timer: StepTimer,
    messenger,
) -> Tuple[pd.DataFrame, np.ndarray]:
    messenger("Loading 2bit reference file for GC contents extraction", indent=2)
    with timer.time_step(indent=4):
        try:
            tb = py2bit.open(str(paths["reference_file"]))
        except Exception as e:
            messenger("Failed to load reference file:")
            messenger(paths["reference_file"], indent=2)
            raise e

    # Create interval data frame based on the GC bin size
    gc_bins_df = bins_df.copy()
    if gc_bin_size != bin_size:
        messenger("Calculating GC bin coordinates", indent=2)
        with timer.time_step(indent=4):
            # Extend bins to get the GC bin size
            gc_excess = (gc_bin_size - bin_size) / 2
            slop_left = int(np.floor(gc_excess))
            slop_right = int(np.ceil(gc_excess))
            messenger(
                f"Slopping intervals (start -= {slop_left}, end += {slop_right})",
                indent=4,
            )
            gc_bins_df["start"] -= slop_left
            gc_bins_df["end"] += slop_right

            # Truncate to [0, size] where size is for the specific chromosome
            messenger("Truncating coordinates to [0, chromosome size]", indent=4)
            with timer.time_step(indent=6, message="Zero-truncation took:"):
                gc_bins_df.loc[gc_bins_df["start"] < 0, "start"] = 0
                for chrom, chrom_size in chrom_sizes_map.items():
                    gc_bins_df.loc[
                        (gc_bins_df["chromosome"] == chrom)
                        & (gc_bins_df["end"] > chrom_size),
                        "end",
                    ] = chrom_size

    messenger("Extracting GC content for each bin", indent=2)
    with timer.time_step(indent=4):
        bins_df["GC"] = get_gc_content_all_intervals(
            bed_df=gc_bins_df,
            tb=tb,
            ignore_non_acgt=gc_ignore_non_acgt,
            handle_zero_division=np.nan,
        )

    del gc_bins_df
    gc.collect()

    messenger("First 20 GC bins: ", indent=2)
    messenger(bins_df.head(20), indent=4)

    messenger(
        "Removing bins that encountered zero-division in GC calculations",
        indent=2,
    )
    with timer.time_step(indent=4):
        len_before = len(bins_df)
        bins_df.dropna(axis=0, subset=["GC"], inplace=True)
        len_after = len(bins_df)
        messenger(f"Removed {len_before - len_after} bins", indent=2)

    # Find bin edges for GC correction
    messenger("Extracting bin edges for GC correction factors", indent=2)
    with timer.time_step(indent=2):
        gc_bin_edges = find_greedy_bin_edges(
            x=bins_df["GC"].to_numpy(),
            num_bins=num_gc_bins,
            range_=(0.0, 1.0),
        )

    del tb
    gc.collect()

    return bins_df, gc_bin_edges


def _create_insert_size_bin_edges(start: int, stop: int, num_bins: int) -> np.ndarray:
    # Linearly spaced bins
    return np.linspace(start=start, stop=stop, num=num_bins + 1)


if __name__ == "__main__":
    main()
