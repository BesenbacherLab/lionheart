"""
Workflow target creators for creating resources for LIONHEART.

"""

import pathlib
import re
from typing import Dict, Optional, Union, List

from gwf import Workflow


def bin_genome(
    gwf: Workflow,
    scripts_dir: Union[str, pathlib.Path],
    bam_file: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    mosdepth_path: Optional[Union[str, pathlib.Path]],
    ld_library_path: Optional[Union[str, pathlib.Path]],
    reference_file: Union[str, pathlib.Path],
    chrom_sizes_file: Union[str, pathlib.Path],
    exclusion_files: List[Union[str, pathlib.Path]],
    walltime: str = "12:00:00",
    memory: str = "70g",
    cores: int = 4,
) -> dict:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    bam_file
        Path to a small BAM file to get bin coordinates from. Must have >= 1 fragment per chromosome.
    out_dir
        Directory to save output in.
    mosdepth_path
        Path to mosdepth application. Note that we use a modified
        version of mosdepth - the original version will not work here.
        Path example: If you have downloaded the forked mosdepth repository to your
        user directory and compiled mosdepth as specified,
        supply something like '/home/<username>/mosdepth/mosdepth'.
    ld_library_path
        You may need to specify the `LD_LIBRARY_PATH`, which is
        the path to the `lib` directory in the directory of your
        anaconda environment.
        Supply something like '/home/<username>/anaconda3/envs/<env_name>/lib/'.
    walltime
        A string specifying the available time for the target.
        For large samples, this might need to be increased.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
        Mosdepth uses 4 threads.
    """
    bam_file = pathlib.Path(bam_file).resolve()
    out_dir = pathlib.Path(out_dir).resolve()
    scripts_dir = pathlib.Path(scripts_dir).resolve()

    input_files = [bam_file, reference_file, chrom_sizes_file, *exclusion_files]

    chroms = [f"chr{i}" for i in range(1, 23)]
    chrom_bin_files = {chrom: out_dir / f"{chrom}.parquet" for chrom in chroms}

    out_files = {
        "gc_bin_edges": out_dir / "gc_contents_bin_edges.npy",
        "iss_bin_edges": out_dir / "insert_size_bin_edges.npy",
        "coordinates": out_dir / "bin_coordinates.bed",
    }

    expected_output_files = list(out_files.values()) + list(chrom_bin_files.values())

    app_args = ""
    if mosdepth_path is not None:
        app_args += f"--mosdepth_path {mosdepth_path} "
    if ld_library_path is not None:
        app_args += f"--ld_library_path {ld_library_path} "

    (
        gwf.target(
            legalize_target_name("lionheart_bin_genome"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
            cores=cores,
        )
        << log_context(
            f"""
        python {scripts_dir / "bin_genome.py"} --bam_file {bam_file} --out_dir {out_dir} --chrom_sizes_file {chrom_sizes_file} --exclusion_bed_files {" ".join(to_strings(exclusion_files))} --reference_file {reference_file} --bin_size 10 --gc_bin_size 100 --num_gc_bins 100 {app_args}
        """
        )
    )

    out_files["chromosome_files"] = chrom_bin_files
    return out_files


def bin_chromatin_tracks(
    gwf: Workflow,
    scripts_dir: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    coordinates_file: Union[str, pathlib.Path],
    tracks_dir: Union[str, pathlib.Path],
    meta_data_file: Union[str, pathlib.Path],
    chrom_sizes_file: Union[str, pathlib.Path],
    track_type: str,
    walltime: str = "12:00:00",
    memory: str = "100g",
    cores: int = 12,
) -> dict:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    bam_file
        Path to a small BAM file to get bin coordinates from. Must have >= 1 fragment per chromosome.
    out_dir
        Directory to save output in.
    walltime
        A string specifying the available time for the target.
        For large samples, this might need to be increased.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
        As many as possible!
    """
    tracks_dir = pathlib.Path(tracks_dir).resolve()
    coordinates_file = pathlib.Path(coordinates_file).resolve()
    out_dir = pathlib.Path(out_dir).resolve()
    scripts_dir = pathlib.Path(scripts_dir).resolve()

    track_files = list(tracks_dir.glob("*.bed"))

    input_files = [meta_data_file, chrom_sizes_file] + track_files

    out_files = {
        "binned_chrom_cell_type_paths": out_dir
        / "binned_per_chrom_and_cell_type_paths.tsv"
    }

    expected_output_files = list(out_files.values())

    (
        gwf.target(
            legalize_target_name(f"lionheart_bin_chromatin_tracks_{track_type}"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
            cores=cores,
        )
        << log_context(
            f"""
        python {scripts_dir / "bin_chromatin_tracks.py"} --coordinates_file {coordinates_file} --tracks_dir {tracks_dir} --out_dir {out_dir} --meta_data_file {meta_data_file} --chrom_sizes_file {chrom_sizes_file} --bin_size 10 --num_jobs {cores}
        """
        )
    )

    return out_files


def find_outlier_candidates(
    gwf: Workflow,
    scripts_dir: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    dataset_name: str,
    bam_files: List[Union[str, pathlib.Path]],
    mosdepth_path: Optional[Union[str, pathlib.Path]],
    ld_library_path: Optional[Union[str, pathlib.Path]],
    coordinates_file: Union[str, pathlib.Path],
    walltime: str = "02:00:00",
    memory: str = "15g",
    cores: int = 1,
) -> dict:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    bam_file
        Path to a small BAM file to get bin coordinates from. Must have >= 1 fragment per chromosome.
    out_dir
        Directory to save output in.
    mosdepth_path
        Path to mosdepth application. Note that we use a modified
        version of mosdepth - the original version will not work here.
        Path example: If you have downloaded the forked mosdepth repository to your
        user directory and compiled mosdepth as specified,
        supply something like '/home/<username>/mosdepth/mosdepth'.
    ld_library_path
        You may need to specify the `LD_LIBRARY_PATH`, which is
        the path to the `lib` directory in the directory of your
        anaconda environment.
        Supply something like '/home/<username>/anaconda3/envs/<env_name>/lib/'.
    walltime
        A string specifying the available time for the target.
        For large samples, this might need to be increased.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
        As many as possible!
    """
    coordinates_file = pathlib.Path(coordinates_file).resolve()
    out_dir = pathlib.Path(out_dir).resolve()
    scripts_dir = pathlib.Path(scripts_dir).resolve()

    # ld lib path variable
    path_var = ""
    if ld_library_path:
        path_var = f"LD_LIBRARY_PATH={pathlib.Path(ld_library_path).resolve()}/ "

    all_outfiles = {}

    for bam_idx, bam_file in enumerate(bam_files):
        bam_file = pathlib.Path(bam_file).resolve()
        bam_out_dir = out_dir / str(bam_idx)

        input_files = [bam_file, coordinates_file]
        expected_output_files = [
            bam_out_dir / "zeros.txt",
            bam_out_dir / "candidates.txt",
        ]
        all_outfiles[bam_idx] = expected_output_files

        (
            gwf.target(
                legalize_target_name(
                    f"lionheart_detect_outliers_{dataset_name}_bam_{bam_idx}"
                ),
                inputs=to_strings(input_files),
                outputs=to_strings(expected_output_files),
                walltime=walltime,
                memory=memory,
                cores=cores,
            )
            # input_file mosdepth_path threshold keep_file out_dir
            << log_context(
                f"""
            {path_var}{scripts_dir / "detect_outlier_candidates.sh"} {bam_file} {mosdepth_path} 0.001 {coordinates_file} {bam_out_dir}
            """
            )
        )

    return all_outfiles


def collect_outliers_for_dataset(
    gwf: Workflow,
    scripts_dir: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    dataset_name: str,
    candidate_files: Dict[int, List[Union[str, pathlib.Path]]],
    walltime: str = "02:00:00",
    memory: str = "15g",
    cores: int = 1,
) -> dict:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    bam_file
        Path to a small BAM file to get bin coordinates from. Must have >= 1 fragment per chromosome.
    out_dir
        Directory to save output in.
    walltime
        A string specifying the available time for the target.
        For large samples, this might need to be increased.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
    """
    out_dir = pathlib.Path(out_dir).resolve()
    scripts_dir = pathlib.Path(scripts_dir).resolve()

    flattened_candidate_files = [
        file for files in candidate_files.values() for file in files
    ]
    candidate_dirs = list(
        set([pathlib.Path(file).parent for file in flattened_candidate_files])
    )

    input_files = flattened_candidate_files
    output_files = {
        "outlier_indices": out_dir / "outlier_indices.npy",
        "zero_coverage_indices": out_dir / "zero_coverage_indices.npy",
    }
    expected_output_files = list(output_files.values())

    (
        gwf.target(
            legalize_target_name(f"lionheart_collect_outliers_{dataset_name}"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
            cores=cores,
        )
        << log_context(
            f"""
        python {scripts_dir / "collect_outliers.py"} --candidate_dirs {" ".join(to_strings(candidate_dirs))} --out_dir {out_dir} --thresholds 1e-4 {1 / 263_108_376} --out_ofs 0.25 0.1
        """
        )
    )

    # TODO: Update num bins in thresholds to the new coordinates file

    return output_files


def collect_outliers_across_datasets(
    gwf: Workflow,
    scripts_dir: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    dataset_to_outlier_paths: Dict[str, Dict[str, Union[str, pathlib.Path]]],
    walltime: str = "02:00:00",
    memory: str = "15g",
    cores: int = 1,
) -> dict:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    bam_file
        Path to a small BAM file to get bin coordinates from. Must have >= 1 fragment per chromosome.
    out_dir
        Directory to save output in.
    walltime
        A string specifying the available time for the target.
        For large samples, this might need to be increased.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
        As many as possible!
    """
    out_dir = pathlib.Path(out_dir).resolve()
    scripts_dir = pathlib.Path(scripts_dir).resolve()

    flattened_candidate_files = [
        file for files in dataset_to_outlier_paths.values() for file in files.values()
    ]
    candidate_dirs = list(
        set([pathlib.Path(file).parent for file in flattened_candidate_files])
    )

    input_files = flattened_candidate_files
    print(flattened_candidate_files)
    output_files = {
        "outlier_indices": out_dir / "outlier_indices.npy",
        "zero_coverage_indices": out_dir / "zero_coverage_indices.npy",
    }
    expected_output_files = list(output_files.values())

    (
        gwf.target(
            legalize_target_name("lionheart_collect_outliers_across_dataset"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
            cores=cores,
        )
        << log_context(
            f"""
        python {scripts_dir / "collect_outliers_across_datasets.py"} --outlier_dirs {" ".join(to_strings(candidate_dirs))} --out_dir {out_dir} --outlier_method union --zero_method intersection
        """
        )
    )

    return output_files


def legalize_target_name(target_name):
    """
    Ensure the target name is valid.
    """

    # First check if target name is legal.
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9._]*$", target_name):
        return target_name

    # Target name is not legal. Replace illegal characters by underscores.
    result = ""

    # First character is special.
    if re.match(r"[a-zA-Z_]", target_name[0]):
        result += target_name[0]
    else:
        result += "_"

    # Check remaining characters.
    for i in range(1, len(target_name)):
        if re.match(r"[a-zA-Z0-9._]", target_name[i]):
            result += target_name[i]
        else:
            result += "_"

    return result


def to_strings(ls):
    """
    Convert a list of elements to strings
    by applying `str()` to each element.
    """
    return [str(s) for s in ls]


def log_context(call: str):
    """
    Add call context to the `gwf` log files (found at `.gwf/logs/<target_name>.stdout`).
    """
    escaped_call = call.replace('"', r"\"").replace("'", r"\'").strip()
    return f"""
printf '%s\n' '---'
date
printf "JobID: $SLURM_JOB_ID \n"
printf "Command:\n{escaped_call}\n"
printf '%s' '---\n\n'
{call}
"""
