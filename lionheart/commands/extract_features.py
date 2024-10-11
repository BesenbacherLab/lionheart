"""
Script that extracts features for a single sample.

"""

from typing import Dict, Optional, Callable, Tuple
from collections import OrderedDict
import logging
import pathlib
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np
import scipy.sparse
from utipy import Messenger, StepTimer, IOPaths, mk_dir, rm_dir

from lionheart.utils.bed_ops import (
    get_file_num_lines,
    split_by_chromosome,
    read_bed_as_df,
)
from lionheart.features.create_dataset_inference import (
    create_dataset_for_inference,
    DatasetOutputPaths,
)
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples, Guide


@dataclass
class MosdepthPaths:
    mosdepth_path: pathlib.Path
    ld_lib_path: Optional[pathlib.Path]

    def __str__(self) -> str:
        string = ""
        if self.ld_lib_path is not None:
            string += f"LD_LIBRARY_PATH={self.ld_lib_path.resolve()}/ "
        string += str(self.mosdepth_path.resolve())
        return string


def run_mosdepth(
    in_file: pathlib.Path,
    by_file: pathlib.Path,
    out_dir: pathlib.Path,
    insert_size_mode: bool,
    n_jobs: int,
    mosdepth_paths: Optional[MosdepthPaths] = None,
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
) -> Tuple[Dict[str, pathlib.Path], pathlib.Path]:
    if n_jobs < 0:
        n_jobs = 4
    n_jobs = min(4, n_jobs)

    coverage_type = "coverage"
    if insert_size_mode:
        coverage_type = "insert_sizes"

    coverage_out_file = pathlib.Path(out_dir) / f"{coverage_type}.regions.bed"

    mosdepth_reference = "mosdepth" if mosdepth_paths is None else str(mosdepth_paths)

    check_paths_for_subprocess(in_file, out_dir)
    mosdepth_call = " ".join(
        [
            "cd",
            str(out_dir),
            ";",
            f"{mosdepth_reference}",
            "--by",
            f"{by_file}",
            "--threads",
            f"{n_jobs}",
            "--mapq",
            "20",
            "--min-frag-len 100",
            "--max-frag-len 220" + (" --insert-size-mode" if insert_size_mode else ""),
            "--no-per-base",
            f"{out_dir/coverage_type}",  # Output prefix
            str(in_file),
        ]
    )
    messenger("Calling mosdepth")
    call_subprocess(mosdepth_call, "`mosdepth` failed")

    messenger("Unzipping output temporarily")
    call_subprocess(f"gunzip -f {coverage_out_file}.gz", "`gunzip -f` failed")

    # Get number of lines (bins) in output
    coverage_num_lines = get_file_num_lines(in_file=coverage_out_file)

    messenger("Splitting output by chromosome")
    df_splits_path = out_dir / f"df_{coverage_type}_by_chromosome"
    mk_dir(
        path=df_splits_path,
        arg_name="df_by_chromosome - mosdepth splits",
        raise_on_exists=False,
    )
    split_by_chromosome(in_file=coverage_out_file, out_dir=df_splits_path)

    messenger("Checking that the splits have the expected number of bins")
    # NOTE: Theoretically this would not catch weird cases where
    # the overall number of bins is correct but the splitting
    # was e.g. random (non-zero) (Unlikely!)

    chrom_split_files = {
        f"chr{chrom}": df_splits_path / f"chr{chrom}.bed" for chrom in range(1, 23)
    }

    split_sizes = [get_file_num_lines(path) for path in chrom_split_files.values()]

    if not all([s > 0 for s in split_sizes]):
        raise RuntimeError("One of the chromosome coverage files was empty.")

    if sum(split_sizes) != coverage_num_lines:
        raise RuntimeError(
            "Splitting coverage file by chromosome failed. "
            f"Original file had ({coverage_num_lines}) bins, but "
            "the total number of bins from all chromosomes (after split) "
            f"was ({sum(split_sizes)}). Please try again."
        )

    coverage_splits_path = out_dir / f"sparse_{coverage_type}_by_chromosome"
    mk_dir(
        path=coverage_splits_path,
        arg_name=f"sparse_{coverage_type}_by_chromosome",
        raise_on_exists=False,
    )

    # Clean up intermediate file
    coverage_out_file.unlink()

    # Save coverage as sparse arrays
    coverage_paths = sparsify_coverage_files(
        chrom_split_files=chrom_split_files,
        coverage_splits_path=coverage_splits_path,
        messenger=messenger,
    )

    # Clean up intermediate files
    messenger("Clean up intermediate files")
    rm_dir(
        path=df_splits_path,
        arg_name="df_splits_path",
        raise_missing=True,
        messenger=messenger,
    )

    return coverage_paths


def sparsify_coverage_files(
    chrom_split_files: Dict[str, pathlib.Path],
    coverage_splits_path: pathlib.Path,
    messenger: Messenger,
) -> Dict[str, pathlib.Path]:
    messenger("Reading each split separately and saving coverage as sparse array")
    coverage_paths = {}
    for chrom, file_path in chrom_split_files.items():
        assert chrom[:3] == "chr"
        chrom_df = read_bed_as_df(
            path=file_path,
            # mosdepth writes to 5th column if 4+ columns exist
            col_names=["chromosome", "start", "end", "fourth_column", "coverage"],
        )
        coverage = chrom_df.coverage.to_numpy().astype(np.float64)
        coverage = scipy.sparse.csr_matrix(coverage)
        coverage_path = coverage_splits_path / f"{chrom}.npz"
        scipy.sparse.save_npz(coverage_path, coverage)
        coverage_paths[chrom] = coverage_path
    return coverage_paths


def standardize_sample(x):
    assert x.ndim == 1
    scaling_factor = np.std(x)
    center = np.mean(x)
    x -= center
    x /= scaling_factor
    return x, center, scaling_factor


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
        help=(
            "Path to directory with framework resources." "\nMust be downloaded first."
        ),
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
    """feature_dataset.npy : `numpy.ndarray` with shape (10, 489)
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
    dhs_outputs = DatasetOutputPaths.create_default(
        dataset_dir=dataset_dir, mask_type="DHS"
    )
    atac_outputs = DatasetOutputPaths(
        dataset=dataset_dir / "ATAC" / "feature_dataset.npy",
    )
    output_path_collections = {"DHS": dhs_outputs, "ATAC": atac_outputs}

    paths = IOPaths(
        in_files={
            "bam_file": args.bam_file,
            "binned_whole_genome": resources_dir
            / "whole_genome.mappable.binned_10bp.bed.gz",
            "gc_correction_bin_edges_path": resources_dir
            / "whole_genome.mappable.binned_10bp.gc_contents_bin_edges.npy",
            "insert_size_correction_bin_edges_path": resources_dir
            / "whole_genome.mappable.binned_10bp.insert_size_bin_edges.npy",
            "exclude_outlier_indices": resources_dir
            / "exclude_bins"
            / "outlier_indices.npz",
            "exclude_zero_indices": resources_dir
            / "exclude_bins"
            / "zero_coverage_bins_indices.npz",
            "ATAC_cell_type_order": resources_dir / "ATAC.idx_to_cell_type.csv",
            "DHS_cell_type_order": resources_dir / "DHS.idx_to_cell_type.csv",
        },
        in_dirs={
            "resources_dir": resources_dir,
            "bins_by_chromosome_dir": resources_dir / "bins_by_chromosome_10bp",
            "exclude_bins_dir": resources_dir / "exclude_bins",
            "chromatin_masks": resources_dir / "chromatin_masks",
            "consensus_super_dir": resources_dir / "consensus_bins",
            "consensus_ATAC_dir": resources_dir / "consensus_bins" / "ATAC",
            "consensus_DHS_dir": resources_dir / "consensus_bins" / "DHS",
        },
        out_dirs={
            "out_path": out_path,
            "coverage_dir": out_path / "coverage",
            "dataset_dir": dataset_dir,
            "atac_dataset_dir": dataset_dir / "ATAC",
            "dhs_dataset_dir": dataset_dir / "DHS",
        },
        out_files={
            **dhs_outputs.get_path_dict(key_prefix="DHS_"),
            **atac_outputs.get_path_dict(key_prefix="ATAC_"),
            "dataset_out_path": dataset_dir / "feature_dataset.npy",
            "standardization_params": dataset_dir / "standardization_params.json",
        },
    )
    if args.mosdepth_path is not None:
        paths.set_path("mosdepth", args.mosdepth_path, collection="in_files")
    if args.ld_library_path is not None:
        paths.set_path("ld_library", args.ld_library_path, collection="in_dirs")

    mask_to_cell_type_to_idx = {}
    mask_to_cell_type_mask_dirs = {}

    for mask_type in ["DHS", "ATAC"]:
        # Data frame with features indices for cell types
        mask_to_cell_type_to_idx[mask_type] = pd.read_csv(
            paths[f"{mask_type}_cell_type_order"]
        )

        # Create expected paths to cell type mask directories
        # Maintaining the insertion order is paramount
        mask_to_cell_type_mask_dirs[mask_type] = OrderedDict(
            [
                (cell_type, paths["chromatin_masks"] / mask_type / cell_type)
                for cell_type in mask_to_cell_type_to_idx[mask_type]["cell_type"]
                if cell_type != "consensus"
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
    paths.mk_output_dirs(collection="out_dirs")

    # Show overview of the paths
    messenger(paths)

    mosdepth_paths = None
    if args.mosdepth_path is not None:
        mosdepth_paths = MosdepthPaths(
            mosdepth_path=paths["mosdepth"],
            ld_lib_path=paths.get_path(name="ld_library", raise_on_fail=False),
        )

    messenger("Start: Extracting coverage with mosdepth")
    with timer.time_step(indent=4, name_prefix="mosdepth_cov"):
        with messenger.indentation(add_indent=4):
            coverage_by_chrom_paths = run_mosdepth(
                in_file=paths["bam_file"],
                by_file=paths["binned_whole_genome"],
                out_dir=paths["coverage_dir"],
                n_jobs=args.n_jobs,
                mosdepth_paths=mosdepth_paths,
                insert_size_mode=False,
            )

    messenger("Start: Extracting average overlapping insert sizes with mosdepth")
    with timer.time_step(indent=4, name_prefix="mosdepth_iss"):
        with messenger.indentation(add_indent=4):
            insert_sizes_by_chrom_paths = run_mosdepth(
                in_file=paths["bam_file"],
                by_file=paths["binned_whole_genome"],
                out_dir=paths["coverage_dir"],
                n_jobs=args.n_jobs,
                mosdepth_paths=mosdepth_paths,
                insert_size_mode=True,
            )

    messenger("Start: Calculating features")
    messenger("-------------", indent=4)
    with timer.time_step(indent=4, name_prefix="dataset_creation"):
        for mask_type in ["DHS", "ATAC"]:
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
                    gc_correction_bin_edges_path=paths["gc_correction_bin_edges_path"],
                    insert_size_correction_bin_edges_path=paths[
                        "insert_size_correction_bin_edges_path"
                    ],
                    consensus_dir_path=paths[f"consensus_{mask_type}_dir"],
                    exclude_paths=[
                        paths["exclude_outlier_indices"],
                        paths["exclude_zero_indices"],
                    ],
                    n_jobs=args.n_jobs,
                    messenger=messenger,
                )
            messenger("-------------", indent=4)

    messenger("Start: Collecting features across ATAC and DHS")
    with timer.time_step(indent=4, name_prefix="stack_mask_types"):
        feature_dataset = np.hstack(
            [
                np.load(
                    output_path_collections[mask_type].dataset, allow_pickle=True
                ).astype(np.float32)
                for mask_type in ["ATAC", "DHS"]
            ]
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
            paths.rm_dir("atac_dataset_dir", messenger=messenger)
            paths.rm_dir("dhs_dataset_dir", messenger=messenger)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
