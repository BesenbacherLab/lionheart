"""
Workflow target creators for extracting features and
predicting cancer probability for a single BAM file.

"""

import pathlib
import re
from typing import Optional, Union, List

from gwf import Workflow

from lionheart.utils.global_vars import INCLUDED_MODELS


# TODO: If we update the ROC curve, change the note in docstring for `thresholds`


def extract_features(
    gwf: Workflow,
    sample_id: str,
    bam_file: Union[str, pathlib.Path],
    resources_dir: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
    mosdepth_path: Optional[Union[str, pathlib.Path]],
    ld_library_path: Optional[Union[str, pathlib.Path]],
    walltime: str = "12:00:00",
    memory: str = "60g",
    cores: int = 10,
) -> None:
    """
    Create target for extracting features for a single sample (i.e. one BAM file).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    sample_id
        ID for current sample to use in target naming. Must be unique.
    bam_file
        Path to a single BAM file to extract features for.
    resources_dir
        Path to the directory with the framework resources.
        TODO: Add download instructions here.
    out_dir
        Directory to save output in.
        Two sub directories "dataset" and "logs" are created.
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
        Tip: Run for a single sample (e.g., the largest)
        first, to ensure the assigned time is enough.
    memory
        The memory (RAM) available to the target.
    cores
        The number of cores available to the target.
        Parallelization is used during feature calculation
        across the many cell type masks.
    """
    bam_file = pathlib.Path(bam_file).resolve()
    resources_dir = pathlib.Path(resources_dir).resolve()
    out_dir = pathlib.Path(out_dir).resolve()

    # Note: We also need the chromatin masks and consensus bins
    # But adding too many files to gwf can make it slow
    # due to excessive IO when testing for file changes
    input_files = [
        bam_file,
        resources_dir / "whole_genome.mappable.binned_10bp.bed.gz",
        resources_dir / "whole_genome.mappable.binned_10bp.gc_contents_bin_edges.npy",
        resources_dir / "whole_genome.mappable.binned_10bp.insert_size_bin_edges.npy",
        resources_dir / "ATAC.idx_to_cell_type.csv",
        resources_dir / "DHS.idx_to_cell_type.csv",
        resources_dir / "exclude_bins" / "outlier_indices.npz",
        resources_dir / "exclude_bins" / "zero_coverage_bins_indices.npz",
    ]

    expected_output_files = [
        out_dir / "dataset" / "feature_dataset.npy",
        out_dir / "dataset" / "gc_correction_factor.npy",
        out_dir / "dataset" / "gc_bin_midpoints.npy",
        out_dir / "dataset" / "coverage_stats.json",
        out_dir / "dataset" / "insert_size.mean_shift_correction_factors.npy",
        out_dir / "dataset" / "insert_size.noise_correction_factors.npy",
        out_dir / "dataset" / "insert_size.skewness_correction_factors.npy",
        out_dir / "dataset" / "insert_size.observed_bias.npy",
        out_dir / "dataset" / "insert_size.target_bias.npy",
        out_dir / "dataset" / "insert_size.optimal_params.csv",
        out_dir / "dataset" / "insert_size.bin_midpoints.npy",
        out_dir / "dataset" / "megabin_normalization_offset_combinations.csv",
    ]

    app_args = ""
    if mosdepth_path is not None:
        app_args += f"--mosdepth_path {mosdepth_path} "
    if ld_library_path is not None:
        app_args += f"--ld_library_path {ld_library_path} "

    (
        gwf.target(
            legalize_target_name(f"lionheart_extract_features_{sample_id}"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
            cores=cores,
        )
        << log_context(
            f"""
        lionheart extract_features --bam_file {bam_file} --resources_dir {resources_dir} --out_dir {out_dir} {app_args}--n_jobs {cores}
        """
        )
    )


def predict_sample(
    gwf: Workflow,
    sample_id: str,
    sample_dir: Union[str, pathlib.Path],
    resources_dir: Union[str, pathlib.Path],
    out_dir: Optional[Union[str, pathlib.Path]] = None,
    thresholds: List[str] = [
        "max_j",
        "spec_0.95",
        "spec_0.99",
        "sens_0.95",
        "sens_0.99",
        "0.5",
    ],
    model_name=INCLUDED_MODELS[0],
    walltime: str = "00:59:00",
    memory: str = "3g",
):
    """
    Create target for predicting the probability of cancer for a single sample.
    Note: Features must first be extracted (see `extract_features`).

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    sample_id
        ID for current sample to use in target naming. Must be unique.
    sample_dir
        Path to the directory with the sample's features
        (NOTE: `out_dir` from `extract_features()`).
        Expects a sub directory named `"dataset"` with the
        `"feature_dataset.npy"` file.
    resources_dir
        Path to the directory with the framework resources.
        TODO: Add download instructions here.
    out_dir :  default=None
        Directory to save output in.
        When `None`, the output is saved in `sample_dir`.
    thresholds
        List of threshold specifications for converting
        the predicted probability to binary (Control / Cancer)
        classifications.
        See note* below for why these thresholds should be taken with a grain of salt.
        `'max_j'`:
            The threshold at the max. Youden's J (`sensitivity + specificity + 1`).
        `'spec_xx'`:
            Prefix a specificity-based threshold with 'spec_' (e.g., 'spec_0.95').
            The first threshold that should lead to a specificity above this level is chosen.
            The specificity should be within the [0., 1.] range.
        `'sens_xx'`:
            Prefix a sensitivity-based threshold with 'sens_'.
            The first threshold that should lead to a specificity above this level is chosen.
            The sensitivity should be within the [0., 1.] range.
        floating point:
            Pass a specific threshold (as str, e.g., "0.5").
            The nearest threshold in the ROC curve is used.
        NOTE*: The thresholds are taken from the included ROC curve,
        which was fitted to the *training* data during full model training
        (as we used all the data for training). As the model is likely to
        be more certain when predicting training data than test data,
        these thresholds are likely not too precise and should
        be taken with a grain of salt.
    model_name
        Name of model to use. Folder name within `resources_dir / models`.
    walltime
        A string specifying the available time for the target.
    memory
        The memory (RAM) available to the target.
    """
    if model_name not in INCLUDED_MODELS:
        # TODO: Allow giving a path to a trained model.
        raise NotImplementedError(
            "Specifying custom model names is not currently supported."
        )
    sample_dir = pathlib.Path(sample_dir).resolve()
    resources_dir = pathlib.Path(resources_dir).resolve()
    out_dir = sample_dir if out_dir is None else pathlib.Path(out_dir).resolve()

    input_files = [
        sample_dir / "dataset" / "feature_dataset.npy",
        resources_dir / "models" / model_name / "model.joblib",
        resources_dir / "models" / model_name / "ROC_curves.json",
    ]

    expected_output_files = [out_dir / "prediction.csv"]

    (
        gwf.target(
            legalize_target_name(f"lionheart_predict_{sample_id}"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
        )
        << log_context(
            f"""
        lionheart predict_sample --sample_dir {sample_dir} --resources_dir {resources_dir} --out_dir {out_dir} --thresholds {' '.join(thresholds)} --identifier {sample_id}
        """
        )
    )


def collect_samples(
    gwf: Workflow,
    sample_dirs: Optional[List[Union[str, pathlib.Path]]],
    prediction_dirs: Optional[List[Union[str, pathlib.Path]]],
    out_dir: Union[str, pathlib.Path],
    walltime: str = "00:59:00",
    memory: str = "10g",
):
    """
    Create target for collecting features and/or predictions across samples.

    Parameters
    ----------
    gwf
        A `gwf` workflow to create targets for.
    sample_dirs
        Paths to directories with extracted features
        (NOTE: `out_dir` from `extract_features()`).
        Expects directories to have a sub directory named `"dataset"`
        with the `"feature_dataset.npy"` file.
    prediction_dirs
        Paths to directories with predictions
        (NOTE: `out_dir` from `predict_sample()`).
        Expects directories to have the `"prediction.csv"` file.
    out_dir
        Directory to save output in.
    walltime
        A string specifying the available time for the target.
    memory
        The memory (RAM) available to the target.
    """
    if sample_dirs is None and prediction_dirs is None:
        raise ValueError("Either `sample_dirs` or `prediction_dirs` must be specified.")

    input_files = []
    path_string = ""

    # Prepare paths
    out_dir = pathlib.Path(out_dir).resolve()

    if sample_dirs is not None:
        sample_dirs = [
            pathlib.Path(sample_dir).resolve() / "dataset" for sample_dir in sample_dirs
        ]
        input_files += [
            sample_dir / "feature_dataset.npy" for sample_dir in sample_dirs
        ]
        path_string += f"--feature_dirs {' '.join(to_strings(sample_dirs))} "

    if prediction_dirs is not None:
        prediction_dirs = [
            pathlib.Path(pred_dir).resolve() for pred_dir in prediction_dirs
        ]
        input_files += [pred_dir / "prediction.csv" for pred_dir in prediction_dirs]
        path_string += f"--prediction_dirs {' '.join(to_strings(prediction_dirs))} "

    expected_output_files = [
        out_dir / "predictions.csv",
        out_dir / "feature_dataset.npy",
    ]

    (
        gwf.target(
            legalize_target_name("lionheart_collect_samples"),
            inputs=to_strings(input_files),
            outputs=to_strings(expected_output_files),
            walltime=walltime,
            memory=memory,
        )
        << log_context(
            f"""
        lionheart collect {path_string} --out_dir {out_dir}
        """
        )
    )


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
