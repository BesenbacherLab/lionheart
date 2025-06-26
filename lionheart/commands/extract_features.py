"""
Script that extracts features for a single sample.

"""

from typing import Dict, Optional, Callable, Tuple
from collections import OrderedDict
import logging
import pathlib
import json
from dataclasses import dataclass
import warnings
import pandas as pd
import numpy as np
from utipy import Messenger, StepTimer, IOPaths, mk_dir, rm_dir
import concurrent
import concurrent.futures

from lionheart.utils.bam_utils import check_autosomes_in_bam
from lionheart.utils.bed_ops import (
    get_file_num_lines,
    split_nonzeros_by_chromosome,
)
from lionheart.features.create_dataset_inference import (
    create_dataset_for_inference,
    DatasetOutputPaths,
)
from lionheart.utils.sparse_ops import convert_nonzero_bins_to_sparse_array
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples, Guide

# NOTE: Ensure ISS bin edges file fits with this!
FRAGMENT_LENGTH_LIMITS = (100, 220)


@dataclass
class MosdepthPaths:
    mosdepth_path: pathlib.Path
    ld_lib_path: Optional[pathlib.Path]

    def __str__(self) -> str:
        assert isinstance(self.mosdepth_path, pathlib.Path)
        assert isinstance(self.ld_lib_path, pathlib.Path)
        string = ""
        if self.ld_lib_path is not None:
            string += f"LD_LIBRARY_PATH={self.ld_lib_path.resolve()}/ "
        string += str(self.mosdepth_path.resolve())
        return string


def run_mosdepth(
    in_file: pathlib.Path,
    out_dir: pathlib.Path,
    chrom_to_files_out: Dict[str, pathlib.Path],
    insert_size_mode: bool,
    chrom_to_num_chrom_bins: Dict[str, int],
    n_jobs: int,
    length_limits: Tuple[int, int],
    mosdepth_paths: Optional[MosdepthPaths] = None,
    clean_intermediates: bool = True,
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
) -> Dict[str, pathlib.Path]:
    coverage_type = "insert_sizes" if insert_size_mode else "coverage"

    coverage_out_file = pathlib.Path(out_dir) / f"{coverage_type}.regions.bed"

    mosdepth_reference = "mosdepth" if mosdepth_paths is None else str(mosdepth_paths)

    check_paths_for_subprocess(in_file, out_dir)
    mosdepth_call = " ".join(
        [
            "cd",
            str(out_dir),
            ";",
            f"{mosdepth_reference}",
            "--by 10",
            "--threads",
            f"{n_jobs}",
            "--mapq",
            "20",
            f"--min-frag-len {length_limits[0]}",
            f"--max-frag-len {length_limits[1]}"
            + (" --insert-size-mode" if insert_size_mode else ""),
            "--no-per-base",
            f"{out_dir / coverage_type}",  # Output prefix
            str(in_file),
        ]
    )
    messenger(f"{coverage_type}: Calling mosdepth")
    call_subprocess(mosdepth_call, "`mosdepth` failed")

    messenger(f"{coverage_type}: Unzipping output temporarily")
    call_subprocess(
        f"unpigz -p {n_jobs} -f {coverage_out_file}.gz", "`unpigz -f` failed"
    )

    # Get number of lines (bins) in output
    messenger(f"{coverage_type}: Getting number of lines in file")
    coverage_num_lines = get_file_num_lines(in_file=coverage_out_file)

    messenger(f"{coverage_type}: Splitting output by chromosome")
    df_splits_path = out_dir / f"df_{coverage_type}_by_chromosome"
    mk_dir(
        path=df_splits_path,
        arg_name="df_by_chromosome - mosdepth splits",
        raise_on_exists=False,
    )
    split_nonzeros_by_chromosome(
        in_file=coverage_out_file,
        out_dir=df_splits_path,
    )

    # Paths to chromosome-wise nonzero coverage files
    chrom_nonzero_files = {
        f"chr{chrom}": df_splits_path / f"chr{chrom}.txt" for chrom in range(1, 23)
    }

    messenger(f"{coverage_type}: Checking that the splitting did not fail silently")

    split_num_rows = pd.read_csv(
        df_splits_path / "total_rows.txt",
        header=None,
        names=["chromosome", "num_rows"],
        sep="\t",
    )
    if split_num_rows["num_rows"].sum() != coverage_num_lines:
        raise RuntimeError(
            f"{coverage_type}: Splitting {coverage_type} file by chromosome failed. "
            f"Original file had ({coverage_num_lines}) bins, but "
            f"the total number of bins checked during splitting was {split_num_rows['num_rows'].sum()}."
            f"Please try again and report if it keeps happening."
        )

    # Ensure output directory exists
    mk_dir(
        path=chrom_to_files_out["chr1"].parent,
        arg_name=chrom_to_files_out["chr1"].parent.name,
        raise_on_exists=False,
    )

    # Clean up intermediate file
    coverage_out_file.unlink()

    # Save coverage as sparse arrays
    messenger(
        f"{coverage_type}: Converting chromosome-wise nonzero coverages to sparse arrays"
    )
    total_nonzeros, _ = convert_nonzeros_to_sparse_arrays(
        chrom_to_nonzero_files=chrom_nonzero_files,
        chrom_out_files=chrom_to_files_out,
        chrom_to_num_chrom_bins=chrom_to_num_chrom_bins,
    )
    messenger(
        f"{coverage_type}: Got {total_nonzeros} bins with nonzero coverage", indent=2
    )

    # Clean up intermediate files
    if clean_intermediates:
        messenger(f"{coverage_type}: Cleaning up intermediate files")
        rm_dir(
            path=df_splits_path,
            arg_name="df_splits_path",
            raise_missing=True,
            messenger=messenger,
        )


def convert_nonzeros_to_sparse_arrays(
    chrom_to_nonzero_files: Dict[str, pathlib.Path],
    chrom_out_files: Dict[str, pathlib.Path],
    chrom_to_num_chrom_bins: Dict[str, int],
) -> Tuple[int, int]:
    """
    Converts chromosome-wise `.txt` files with nonzero (index, coverage)
    rows to sparse arrays and saves them.
    """
    chromosomes = list(chrom_out_files.keys())
    total_nonzeros = 0
    total_sum = 0

    for chrom in chromosomes:
        coverage_nonzeros, coverage_sum = convert_nonzero_bins_to_sparse_array(
            num_bins=chrom_to_num_chrom_bins[chrom],
            scaling_constant=None,
            input_path=chrom_to_nonzero_files[chrom],
            output_path=chrom_out_files[chrom],
            array_type="csr",
        )
        total_nonzeros += coverage_nonzeros
        total_sum += coverage_sum

    return int(total_nonzeros), int(total_sum)


def standardize_sample(x):
    assert x.ndim == 1
    scaling_factor = np.std(x)
    center = np.mean(x)
    x -= center
    x /= scaling_factor
    return x, center, scaling_factor


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


def setup_parser(parser):
    parser.add_argument(
        "--bam_file",
        required=True,
        type=str,
        help=(
            "Path to a `.bam` file (hg38) for a single sample."
            "\nThe included model is mainly trained on files with 0.3-3x depth but may generalize beyond that."
        ),
    )
    parser.add_argument(
        "--resources_dir",
        required=True,
        type=str,
        help=("Path to directory with framework resources.\nMust be downloaded first."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the output at."
            "\nThis directory should be <b>exclusive to the current sample</b>."
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--mosdepth_path",
        type=str,
        help=(
            "Path to (modified) `mosdepth` application. "
            "\n<b>NOTE</b>: We use a modified version of `mosdepth` - "
            "the original version will not work here. "
            "\n<b>Example</b>: If you have downloaded the forked `mosdepth` "
            "repository to your user directory and"
            "\ncompiled it as specified, "
            "supply something like `'/home/<username>/mosdepth/mosdepth'`."
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
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        help=(
            "Whether to keep all intermediate files."
            "\nOtherwise, we only keep the features, statistics and correction factors."
        ),
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of cores to utilize."
    )
    parser.set_defaults(func=main)


DESCRIPTION = """EXTRACT FEATURES from a BAM file.
"""

# Create Epilog

epilog_guide = Guide()
epilog_guide.add_title("OUTPUT:")
epilog_guide.add_description(
    """feature_dataset.npy : `numpy.ndarray` with shape (10, 898)
    This array contains the main features. 
    There are 10 feature sets of which we only use the first (index=0).
    We included the other feature sets to allow experimentation.
    
    Feature sets:
        <b>0) LIONHEART score</b>
           (Sample-standardized Pearson correlation coefficient <i>(r)</i>)
        1) and its p-value
        2) The normalized dot product
        3) Cosine Similarity
        
        And the terms used to calculate them:
           Where x=fragment coverage, y=open chromatin site overlap fraction.
        4) x_sum
        5) y_sum
        6) x_squared_sum
        7) y_squared_sum
        8) xy_sum
        9) Number of included bins
"""
)
epilog_guide.add_vertical_space(1)

examples = Examples()
examples.add_example(
    example="""--bam_file path/to/subject_1/<file_name>.bam
--resources_dir path/to/resource/directory
--out_dir path/to/output/directory
--mosdepth_path /home/<username>/mosdepth/mosdepth
--ld_library_path /home/<username>/anaconda3/envs/<env_name>/lib/
--n_jobs 10""",
)

EPILOG = epilog_guide.construct_guide() + examples.construct()


def main(args):
    out_path = pathlib.Path(args.out_dir)
    dataset_dir = out_path / "dataset"
    resources_dir = pathlib.Path(args.resources_dir)

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="extract_features-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running inference feature extraction on a single sample")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # We only save the files that are the same across mask types once
    # So ATAC only has the feature dataset path
    dnase_outputs = DatasetOutputPaths.create_default(
        dataset_dir=dataset_dir,
        mask_type="DNase",
    )
    atac_outputs = DatasetOutputPaths(
        dataset=dataset_dir / "ATAC" / "feature_dataset.npy",
    )
    output_path_collections = {"DNase": dnase_outputs, "ATAC": atac_outputs}

    paths = IOPaths(
        in_files={
            "bam_file": args.bam_file,
            "gc_correction_bin_edges_path": resources_dir / "gc_contents_bin_edges.npy",
            "insert_size_correction_bin_edges_path": resources_dir
            / "insert_size_bin_edges.npy",
            "exclude_outlier_indices": resources_dir
            / "outliers"
            / "outlier_indices.npz",
            "exclude_zero_indices": resources_dir
            / "outliers"
            / "zero_coverage_indices.npz",
            "num_rows_per_chrom_file": resources_dir
            / "rows_per_chrom_pre_exclusion.txt",
            "ATAC_cell_type_order": resources_dir / "ATAC.idx_to_cell_type.csv",
            "DNase_cell_type_order": resources_dir / "DNase.idx_to_cell_type.csv",
        },
        in_dirs={
            "resources_dir": resources_dir,
            "bins_by_chromosome_dir": resources_dir / "bin_indices_by_chromosome",
            "outliers_dir": resources_dir / "outliers",
            "DNase_masks": resources_dir
            / "chromatin_masks"
            / "DNase"
            / "sparse_overlaps_by_chromosome",
            "ATAC_masks": resources_dir
            / "chromatin_masks"
            / "ATAC"
            / "sparse_overlaps_by_chromosome",
        },
        out_dirs={
            "out_path": out_path,
            "coverage_dir": out_path / "coverage",
            "dataset_dir": dataset_dir,
            "DNase_dataset_dir": dataset_dir / "DNase",
            "ATAC_dataset_dir": dataset_dir / "ATAC",
        },
        out_files={
            **dnase_outputs.get_path_dict(key_prefix="DNase_"),
            **atac_outputs.get_path_dict(key_prefix="ATAC_"),
            "dataset_out_path": dataset_dir / "feature_dataset.npy",
            "standardization_params": dataset_dir / "standardization_params.json",
        },
    )
    if args.mosdepth_path is not None:
        paths.set_path("mosdepth", args.mosdepth_path, collection="in_files")
    if args.ld_library_path is not None:
        paths.set_path("ld_library", args.ld_library_path, collection="in_dirs")

    # Load paths to masks/tracks

    mask_to_cell_type_to_idx = {}
    mask_to_cell_type_mask_dirs = {}

    for mask_type in ["DNase", "ATAC"]:
        # Data frame with features indices for cell types
        mask_to_cell_type_to_idx[mask_type] = pd.read_csv(
            paths[f"{mask_type}_cell_type_order"]
        )

        # Create expected paths to cell type mask directories
        # Maintaining the insertion order is paramount
        mask_to_cell_type_mask_dirs[mask_type] = OrderedDict(
            [
                (cell_type, paths[f"{mask_type}_masks"] / cell_type)
                for cell_type in mask_to_cell_type_to_idx[mask_type]["cell_type"]
            ]
        )

        # Suffix keys with mask type and add to paths
        paths.set_paths(
            {
                (key + "_" + mask_type): path
                for key, path in mask_to_cell_type_mask_dirs[mask_type].items()
            },
            collection="in_dirs",
        )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    # Show overview of the paths
    messenger(paths)

    # Check that all autosomes are present, named using the "chr" prefix
    check_autosomes_in_bam(bam_path=paths["bam_file"], messenger=messenger)

    # Load the total number of intervals expected (pre-exclusion) per chromosome
    messenger(
        "Start: Loading expected number of intervals per chromosome before exclusion"
    )
    chrom_to_num_rows: Dict[str, int] = (
        pd.read_csv(
            paths["num_rows_per_chrom_file"],
            sep="\t",
            header=None,
            names=["chromosome", "num_intervals"],
        )
        .set_index("chromosome")["num_intervals"]
        .to_dict()
    )
    messenger(
        "Got: "
        + ", ".join([f"{key}:{int(num)}" for key, num in chrom_to_num_rows.items()]),
        indent=2,
    )

    mosdepth_paths = None
    if args.mosdepth_path is not None:
        mosdepth_paths = MosdepthPaths(
            mosdepth_path=paths["mosdepth"],
            ld_lib_path=paths.get_path(name="ld_library", raise_on_fail=False),
        )

    messenger(
        "Start: Extracting coverage and average overlapping insert sizes with mosdepth"
    )
    with timer.time_step(indent=4, name_prefix="mosdepth"):
        # Calculate whether to parallelize and how many threads to use
        # per mosdepth call (pref. 4 threads + 1 additional core)
        mosdepth_threads = 4 if args.n_jobs >= 10 else int((args.n_jobs - 2) / 2)
        max_workers = 1 if mosdepth_threads < 2 else 2
        if max_workers == 1:
            messenger(
                "`--n_jobs < 6`: Not enough cores to run the two mosdepth calls in parallel. "
                "Running sequentially. Increase `--n_jobs` to 10+ for a big speedup.",
                indent=4,
                add_msg_fn=warnings.warn,
            )

        # Output paths to sparse coverage files per chromosome
        coverage_by_chrom_paths = {
            f"chr{chrom}": paths["coverage_dir"]
            / "sparse_coverage_by_chromosome"
            / f"chr{chrom}.npz"
            for chrom in range(1, 23)
        }

        # Output paths to overlapping sparse insert size files per chromosome
        insert_sizes_by_chrom_paths = {
            f"chr{chrom}": paths["coverage_dir"]
            / "sparse_insert_sizes_by_chromosome"
            / f"chr{chrom}.npz"
            for chrom in range(1, 23)
        }

        with messenger.indentation(add_indent=4):
            # Call mosdepth twice for coverage and ISS extraction, respectively
            # Using ThreadPoolExecutor to process files concurrently
            mosdepth_kwargs = [
                {
                    "in_file": paths["bam_file"],
                    "out_dir": paths["coverage_dir"],
                    # Paths to chromosome-wise sparse coverage array files
                    "chrom_to_files_out": coverage_by_chrom_paths
                    if coverage_type == "coverage"
                    else insert_sizes_by_chrom_paths,
                    "length_limits": FRAGMENT_LENGTH_LIMITS,
                    "chrom_to_num_chrom_bins": chrom_to_num_rows,
                    "n_jobs": mosdepth_threads,
                    "mosdepth_paths": mosdepth_paths,
                    "insert_size_mode": coverage_type == "insert_sizes",
                    "clean_intermediates": not args.keep_intermediates,
                    "messenger": messenger,
                }
                for coverage_type in ["coverage", "insert_sizes"]
            ]
            run_parallel_tasks(
                task_list=mosdepth_kwargs,
                worker=run_mosdepth,
                max_workers=max_workers,
                messenger=messenger,
                extra_verbose=False,
            )

    messenger("Start: Calculating features")
    messenger("-------------", indent=4)
    with timer.time_step(indent=4, name_prefix="dataset_creation"):
        for mask_type in ["DNase", "ATAC"]:
            with timer.time_step(
                indent=6,
                name_prefix=f"{mask_type}_dataset_creation",
                message=f"{mask_type} took: ",
            ):
                messenger(f"{mask_type} features", indent=4)
                messenger("-------------", indent=4)
                with messenger.indentation(add_indent=8):
                    create_dataset_for_inference(
                        chrom_coverage_paths=coverage_by_chrom_paths,
                        chrom_insert_size_paths=insert_sizes_by_chrom_paths,
                        cell_type_paths=mask_to_cell_type_mask_dirs[mask_type],
                        output_paths=output_path_collections[mask_type],
                        bins_info_dir_path=paths["bins_by_chromosome_dir"],
                        cell_type_to_idx=mask_to_cell_type_to_idx[mask_type],
                        gc_correction_bin_edges_path=paths[
                            "gc_correction_bin_edges_path"
                        ],
                        insert_size_correction_bin_edges_path=paths[
                            "insert_size_correction_bin_edges_path"
                        ],
                        exclude_paths=[
                            paths["exclude_outlier_indices"],
                            paths["exclude_zero_indices"],
                        ],
                        n_jobs=args.n_jobs,
                        messenger=messenger,
                    )
            messenger("-------------", indent=4)

    messenger("Start: Collecting features across ATAC and DNase")
    with timer.time_step(indent=4, name_prefix="stack_mask_types"):
        feature_dataset = np.hstack(
            [
                np.load(
                    output_path_collections[mask_type].dataset,
                    allow_pickle=True,
                ).astype(np.float64)
                for mask_type in ["ATAC", "DNase"]
            ]
        )

    messenger(
        "Score statistics (before standardization): "
        f"Mean: {feature_dataset[0].mean()}; "
        f"Std: {feature_dataset[0].std()}; "
        f"Min: {feature_dataset[0].min()}; "
        f"Max: {feature_dataset[0].max()}; "
    )

    messenger("Start: Standardizing correlation coefficients")
    # Standardize pearson correlations to make them LIONHEART scores
    feature_dataset[0, :], center, scaling_factor = standardize_sample(
        feature_dataset[0, :]
    )

    messenger("Start: Writing features to disk")
    np.save(paths["dataset_out_path"], feature_dataset)

    # Write standardization parameters as well
    # to allow inversing the standardization
    with open(str(paths["standardization_params"]), "w") as outfile:
        std_params = {"mean": float(center), "std": float(scaling_factor)}
        json.dump(std_params, outfile)

    if not args.keep_intermediates:
        messenger("Start: Removing intermediate files")
        with messenger.indentation(add_indent=4):
            messenger("Removing coverage files")
            paths.rm_dir("coverage_dir", messenger=messenger)
            paths.rm_dir("ATAC_dataset_dir", messenger=messenger)
            paths.rm_dir("DNase_dataset_dir", messenger=messenger)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
