"""
Script for calculating features from binned coverages counted per chromosome for a single subject.

Features:

Pearson R (and uncorrected p-value)
Within-cell_type-mask fraction
Cosine similarity

"""

from typing import Dict, List, Optional, Callable, Tuple
import pathlib
import json
import gc
from dataclasses import dataclass
import numpy as np
import scipy.sparse
import pandas as pd
from joblib import Parallel, delayed

from utipy import Messenger, StepTimer

from lionheart.utils.bed_ops import read_bed_as_df
from lionheart.features.correction.correction import (
    correct_bias,
    calculate_correction_factors,
)
from lionheart.features.correction.insert_size import (
    calculate_insert_size_correction_factors,
)
from lionheart.features.correction.poisson import ZIPoissonPMF
from lionheart.features.correction.normalize_megabins import normalize_megabins
from lionheart.features.running_pearson_r import RunningPearsonR
from lionheart.features.running_stats import RunningStats

# Constants
THRESHOLD: float = 1 / 263_108_376
MEGABIN_SIZES: Tuple[int, int] = (5000000, 500000)


@dataclass
class DatasetOutputPaths:
    dataset: pathlib.Path
    coverage_stats: Optional[pathlib.Path] = None
    gc_factor_ids: Optional[pathlib.Path] = None
    gc_bin_midpoints: Optional[pathlib.Path] = None
    gc_correction_factors: Optional[pathlib.Path] = None
    insert_size_noise_correction_factors: Optional[pathlib.Path] = None
    insert_size_skewness_correction_factors: Optional[pathlib.Path] = None
    insert_size_mean_correction_factors: Optional[pathlib.Path] = None
    insert_size_observed_bias: Optional[pathlib.Path] = None
    insert_size_target_bias: Optional[pathlib.Path] = None
    insert_size_optimal_params: Optional[pathlib.Path] = None
    insert_size_factor_ids: Optional[pathlib.Path] = None
    insert_size_bin_midpoints: Optional[pathlib.Path] = None
    megabin_normalization_offset_combinations: Optional[pathlib.Path] = None
    feature_readme: Optional[pathlib.Path] = None

    def get_path_dict(self, key_prefix="") -> Dict[str, pathlib.Path]:
        """
        Get all specified paths (not `None`).
        """
        return {
            key_prefix + key: path
            for key, path in self.__dict__.items()
            if path is not None
        }

    @staticmethod
    def create_default(dataset_dir: pathlib.Path, mask_type: Optional[str] = "DHS"):
        if not isinstance(dataset_dir, pathlib.Path):
            raise TypeError(
                f"`dataset_dir` must have type `pathlib.Path`. Got: {type(dataset_dir)}."
            )

        return DatasetOutputPaths(
            dataset=(
                dataset_dir / mask_type / "feature_dataset.npy"
                if mask_type is not None
                else dataset_dir / "feature_dataset.npy"
            ),
            # Shared for mask types
            coverage_stats=dataset_dir / "coverage_stats.json",
            feature_readme=dataset_dir / "feature_description.txt",
            gc_factor_ids=dataset_dir / "gc_correction_factors.identifiers.csv",
            gc_bin_midpoints=dataset_dir / "gc_bin_midpoints.npy",
            gc_correction_factors=dataset_dir / "gc_correction_factor.npy",
            insert_size_mean_correction_factors=dataset_dir
            / "insert_size.mean_shift_correction_factors.npy",
            insert_size_noise_correction_factors=dataset_dir
            / "insert_size.noise_correction_factors.npy",
            insert_size_skewness_correction_factors=dataset_dir
            / "insert_size.skewness_correction_factors.npy",
            insert_size_observed_bias=dataset_dir / "insert_size.observed_bias.npy",
            insert_size_target_bias=dataset_dir / "insert_size.target_bias.npy",
            insert_size_optimal_params=dataset_dir / "insert_size.optimal_params.csv",
            insert_size_bin_midpoints=dataset_dir / "insert_size.bin_midpoints.npy",
            insert_size_factor_ids=dataset_dir
            / "insert_size.correction_factors.identifiers.csv",
            megabin_normalization_offset_combinations=dataset_dir
            / "megabin_normalization_offset_combinations.csv",
        )


def _load_sample_chrom_coverage(path: pathlib.Path) -> np.ndarray:
    return np.asarray(scipy.sparse.load_npz(path).todense(), dtype=np.float64).flatten()


def load_bins(
    path: pathlib.Path, messenger: Messenger
) -> Tuple[np.ndarray, np.ndarray]:
    df = read_bed_as_df(
        path=path,
        col_names=["chromosome", "start", "end", "gc", "mappability"],
        messenger=messenger,
    )
    assert (
        len(df.chromosome.unique()) == 1
    ), f"Found more than one ({len(df.chromosome.unique())}) chromosome in: {path}"
    return (
        df["gc"].to_numpy().astype(np.float64),
        df["start"].to_numpy().astype(np.int64),
    )


def _load_cell_type_chrom_mask(path: pathlib.Path) -> np.ndarray:
    # Load sparse array
    open_chromatin_mask = scipy.sparse.load_npz(path).todense()
    return np.asarray(open_chromatin_mask, dtype=np.float64).flatten()


def _update_r_calculator(
    sample_cov,
    cell_type,
    r_calculator,
    cell_type_cov: Optional[np.ndarray] = None,
    path: Optional[pathlib.Path] = None,
    chrom_r_calculator=None,
    exclude_indices=None,
    consensus_indices=None,  # Additional exclusion indices (*post* exclude_indices exclusion)
):
    assert sum([cell_type_cov is None, path is None]) == 1
    if path is not None:
        cell_type_cov = _load_cell_type_chrom_mask(path)

    if exclude_indices is not None:
        cell_type_cov = np.delete(cell_type_cov, exclude_indices)

    if consensus_indices is not None:
        cell_type_cov = np.delete(cell_type_cov, consensus_indices)

    if len(sample_cov) != len(cell_type_cov):
        raise RuntimeError(
            f"The loaded overlap mask ({len(cell_type_cov)}) did not "
            f"have the expected number of bins ({len(sample_cov)}). "
            f"File: {cell_type} at {path}."
        )

    try:
        r_calculator.add_data(x=sample_cov, y=cell_type_cov)
    except ValueError as e:
        raise ValueError(
            f"Failed to add data to Pearson R calculator: {cell_type} at {path}. {e}"
        )

    # TODO: Decide whether to include per-chromosome features
    if chrom_r_calculator is not None:
        try:
            chrom_r_calculator.add_data(x=sample_cov, y=cell_type_cov)
        except ValueError as e:
            raise ValueError(
                "Failed to add data to chromosome-wise "
                "Pearson R calculator: "
                f"{cell_type} at {path}. {e}"
            )

    return cell_type, r_calculator, chrom_r_calculator


def create_dataset_for_inference(
    chrom_coverage_paths: Dict[str, pathlib.Path],
    chrom_insert_size_paths: Dict[str, pathlib.Path],
    cell_type_paths: Dict[str, pathlib.Path],
    output_paths: DatasetOutputPaths,
    bins_info_dir_path: pathlib.Path,
    cell_type_to_idx: pd.DataFrame,
    gc_correction_bin_edges_path: pathlib.Path,
    insert_size_correction_bin_edges_path: pathlib.Path,
    consensus_dir_path: pathlib.Path,
    exclude_paths: List[pathlib.Path],
    n_jobs: int = 1,
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
):
    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Order chromosomes numerically - which is assumed to also be
    # (somewhat) biggest to smallest
    chroms_ordered = sorted(chrom_coverage_paths.keys(), key=lambda x: int(x[3:]))

    # Paths of cell_type masks per chromosome
    cell_type_chromosome_beds = {
        f"{cell_type}_{chrom}": pathlib.Path(f"{path}/{chrom}.npz")
        for cell_type, path in cell_type_paths.items()
        for chrom in chroms_ordered
    }

    # Check that the masks exist to fail quickly
    missing_masks = [p for p in cell_type_chromosome_beds.values() if not p.exists()]
    if missing_masks:
        raise RuntimeError(
            f"These ({len(missing_masks)}) cell_type mask arrays could not be found: "
            f"{missing_masks}"
        )

    # Bin info paths per chromosome
    chrom_bins_paths = {
        chrom: pathlib.Path(bins_info_dir_path) / (chrom + ".bed.gz")
        for chrom in chroms_ordered
    }

    messenger("Loading GC content bin edges")
    try:
        gc_correction_bin_edges = np.load(
            gc_correction_bin_edges_path, allow_pickle=True
        ).astype(np.float64)
    except:
        messenger("Failed to load GC correction bin edges.")
        raise

    messenger("Loading insert size bin edges")
    try:
        insert_size_correction_bin_edges = np.load(
            insert_size_correction_bin_edges_path, allow_pickle=True
        ).astype(np.float64)
    except:
        messenger("Failed to load insert size correction bin edges.")
        raise

    if exclude_paths:
        messenger(f"Loading exclude indices from {len(exclude_paths)} file(s)")
        exclude_dicts = []

        for path in exclude_paths:
            try:
                exclude_dicts.append(np.load(path, allow_pickle=True))
            except:
                messenger(f"Failed to load exclusion indices from: {path}")
                raise

        exclude_bins_by_chrom = {}
        for chrom in chroms_ordered:
            excl_arrays = [
                excl_dict[chrom].flatten()
                for excl_dict in exclude_dicts
                if chrom in excl_dict.files
            ]
            if excl_arrays:
                exclude_bins_by_chrom[chrom] = np.unique(np.concatenate(excl_arrays))
            else:
                exclude_bins_by_chrom[chrom] = np.array([], dtype=np.int64)

    consensus_chromosome_files = {
        chrom: f"{consensus_dir_path}/{chrom}.npz" for chrom in chroms_ordered
    }

    messenger("Preparing feature calculators")

    r_calculators: Dict[str, RunningPearsonR] = {}
    stats_calculator = RunningStats(ignore_nans=True)
    extra_calculator_names = ["consensus"]

    for cell_type in list(cell_type_paths.keys()) + extra_calculator_names:
        r_calculators[cell_type] = RunningPearsonR(ignore_nans=True)

    # Initialize Poisson distribution
    # In very rare cases with NaNs (like -9223372036854775808)
    # we might just want to truncate them,
    # but we will ask for a warning, so we could potentially
    # see the size of the problem
    # Also, we only allow very few negatives in total,
    # so we don't miss systematic errors
    messenger("Preparing outlier detector")
    poiss = ZIPoissonPMF(handle_negatives="warn_truncate", max_num_negatives=50)

    megabin_offset_combination_averages_collection = {}
    gc_correction_factors_collection = {}
    insert_size_noise_correction_factors_collection = {}
    insert_size_skewness_correction_factors_collection = {}
    insert_size_mean_correction_factors_collection = {}
    insert_size_observed_bias_collection = {}
    insert_size_target_bias_collection = {}
    insert_size_optimal_params_collection = {}

    with timer.time_step(indent=4, name_prefix="load_and_add"):
        for chrom in chroms_ordered:
            with timer.time_step(indent=8, name_prefix=f"{chrom}"):
                messenger(f"{chrom}:", add_indent=-2)

                # Load coverages
                # Even when GC-corrected coverages are passed
                # we need this to find the average overlapping insert sizes below
                sample_cov = _load_sample_chrom_coverage(chrom_coverage_paths[chrom])

                # Save the non-corrected raw integer counts
                # Needed when calculating insert size correction model
                sample_cov_raw_counts = sample_cov.copy()

                # Load insert sizes
                sample_insert_sizes = None
                if chrom_insert_size_paths is not None:
                    with timer.time_step(
                        indent=4,
                        name_prefix=f"load_insert_sizes_{chrom}",
                    ):
                        sample_insert_sizes = _load_sample_chrom_coverage(
                            chrom_insert_size_paths[chrom]
                        )

                        # Convert from sums to means
                        # They should have the same non-zero bins
                        sample_insert_sizes[sample_cov > 0] /= sample_cov[
                            sample_cov > 0
                        ]
                        messenger(
                            f"Loaded and averaged insert sizes. Non-zero bin statistics: "
                            f"min={sample_insert_sizes[sample_insert_sizes > 0].min()}, "
                            f"max={sample_insert_sizes.max()}, "
                            f"mean={sample_insert_sizes[sample_insert_sizes > 0].mean()}",
                            indent=8,
                        )

                # Load reference knowledge about the bins
                (sample_gc, sample_start_coordinates) = load_bins(
                    chrom_bins_paths[chrom], messenger=messenger
                )

                consensus_overlap = _load_cell_type_chrom_mask(
                    consensus_chromosome_files[chrom]
                )

                # Fit of the zero-inflated Poisson distribution
                # Must be done before exclusions and corrections to have the same thresholds
                # as done in the outlier detection
                poiss.reset().partial_fit(x=np.round(sample_cov).astype(np.int64))

                exclude_indices = None
                if exclude_paths:
                    exclude_indices = exclude_bins_by_chrom[chrom]

                    sample_cov = np.delete(sample_cov, exclude_indices)
                    sample_cov_raw_counts = np.delete(
                        sample_cov_raw_counts, exclude_indices
                    )
                    sample_gc = np.delete(sample_gc, exclude_indices)
                    sample_insert_sizes = np.delete(
                        sample_insert_sizes, exclude_indices
                    )
                    sample_start_coordinates = np.delete(
                        sample_start_coordinates, exclude_indices
                    )
                    consensus_overlap = np.delete(consensus_overlap, exclude_indices)

                    messenger(f"Excluded bins: {len(exclude_indices)}")

                # Find the last count value that
                # is above the probability threshold
                # Note: `poiss` iterating gives a count value (from 0 -> inf)
                # and the probability of that value
                poiss = iter(poiss)
                # Start at the mean, so we don't see
                # low values with low probabilities
                poiss.set_iter_pos(pos=int(np.floor(np.nanmean(sample_cov))))
                while True:
                    val, prob = next(poiss)
                    # When the probability is below the threshold
                    # we grab the previous value
                    if prob < THRESHOLD:
                        truncation_val = val - 1
                        break

                messenger(
                    (
                        f"Truncating {np.count_nonzero(sample_cov > truncation_val)} "
                        f"bins to a max. of {truncation_val}"
                    ),
                )

                # Truncate outliers
                sample_cov[sample_cov > truncation_val] = truncation_val

                # We will use this in the insert size correction model building
                # where the estimation of central limit theorem depends on having
                # the correct coverage counts. But we don't want the extreme
                # outliers in that context, so we set them to NaN (instead of
                # truncating them)
                sample_cov_raw_counts[sample_cov_raw_counts > truncation_val] = np.nan

                # Calculate and apply GC correction factor for current chromosome
                with timer.time_step(indent=4):
                    # Init collection for correction factors
                    gc_correction_factors_collection[chrom] = {}

                    # Chromosome level correction factor
                    messenger(
                        "Calculating GC correction factors for chromosome "
                        f"({len(sample_gc)} bins) ",
                    )
                    (
                        gc_bin_midpoints,
                        gc_correction_factors_collection[chrom]["chromosome"],
                    ) = calculate_correction_factors(
                        bias_scores=sample_gc,
                        coverages=sample_cov,
                        bin_edges=gc_correction_bin_edges,
                    )

                    # Apply chromosome-level GC correction to entire chromosome
                    messenger(
                        "Correcting GC bias with chromosome-level correction factor",
                    )
                    sample_cov = correct_bias(
                        coverages=sample_cov,
                        correct_factors=gc_correction_factors_collection[chrom][
                            "chromosome"
                        ],
                        bias_scores=sample_gc,
                        bin_edges=gc_correction_bin_edges,
                    )

                    messenger(
                        f"NaNs after correction: {np.isnan(sample_cov).sum()}",
                        add_indent=4,
                    )

                # Free up memory
                del sample_gc

                # Calculate and apply mappability correction factor for current chromosome
                with timer.time_step(indent=4):
                    # Init collection for correction factors
                    insert_size_noise_correction_factors_collection[chrom] = {}
                    insert_size_skewness_correction_factors_collection[chrom] = {}
                    insert_size_mean_correction_factors_collection[chrom] = {}
                    insert_size_observed_bias_collection[chrom] = {}
                    insert_size_target_bias_collection[chrom] = {}
                    insert_size_optimal_params_collection[chrom] = {}

                    # Chromosome level correction factor
                    messenger(
                        "Calculating insert size correction factors for chromosome "
                        f"({len(sample_insert_sizes)} bins) ",
                    )
                    insert_size_out = calculate_insert_size_correction_factors(
                        coverages=(
                            # Use non-corrected coverages for central limit theorem effect estimation
                            sample_cov_raw_counts
                        ),
                        insert_sizes=sample_insert_sizes,
                        bin_edges=insert_size_correction_bin_edges,
                        base_sigma=8.026649608460776,  # `std / sqrt(depth)` for a handful of DELFI Cristiano samples - currently hardcoded
                        df=5,
                        nan_extremes=True,
                    )

                    # Add the correction factors and bias to collections
                    insert_size_noise_correction_factors_collection[chrom][
                        "chromosome"
                    ] = insert_size_out["noise_correction_factor"]
                    insert_size_skewness_correction_factors_collection[chrom][
                        "chromosome"
                    ] = insert_size_out["skewness_correction_factor"]
                    insert_size_mean_correction_factors_collection[chrom][
                        "chromosome"
                    ] = insert_size_out["mean_correction_factor"]
                    insert_size_observed_bias_collection[chrom]["chromosome"] = (
                        insert_size_out["observed_bias"]
                    )
                    insert_size_target_bias_collection[chrom]["chromosome"] = (
                        insert_size_out["target_bias"]
                    )
                    insert_size_optimal_params_collection[chrom]["chromosome"] = (
                        insert_size_out["optimal_fit_params"]
                    )
                    insert_size_bin_midpoints = insert_size_out["bin_midpoints"]

                    # Add statistics to optimal settings dict for later
                    # modeling/plotting purposes
                    insert_size_optimal_params_collection[chrom]["chromosome"][
                        "statistic__mean_non_zero_coverage"
                    ] = np.nanmean(sample_cov[sample_insert_sizes > 0])
                    insert_size_optimal_params_collection[chrom]["chromosome"][
                        "statistic__std_non_zero_coverage"
                    ] = np.nanstd(sample_cov[sample_insert_sizes > 0])
                    insert_size_optimal_params_collection[chrom]["chromosome"][
                        "statistic__mean_non_zero_insert_size"
                    ] = np.nanmean(sample_insert_sizes[sample_insert_sizes > 0])
                    insert_size_optimal_params_collection[chrom]["chromosome"][
                        "statistic__std_non_zero_insert_size"
                    ] = np.nanstd(sample_insert_sizes[sample_insert_sizes > 0])

                    # Apply chromosome-level insert size correction to entire chromosome
                    messenger(
                        "Reducing insert size bias with chromosome-level noise, "
                        "skewness and mean-shift correction factors",
                    )

                    # Correct the noise
                    sample_cov[sample_insert_sizes > 0] = correct_bias(
                        coverages=sample_cov[sample_insert_sizes > 0],
                        correct_factors=insert_size_noise_correction_factors_collection[
                            chrom
                        ]["chromosome"],
                        bias_scores=sample_insert_sizes[sample_insert_sizes > 0],
                        bin_edges=insert_size_correction_bin_edges,
                    )

                    # Correct the skewness
                    sample_cov[sample_insert_sizes > 0] = correct_bias(
                        coverages=sample_cov[sample_insert_sizes > 0],
                        correct_factors=insert_size_skewness_correction_factors_collection[
                            chrom
                        ]["chromosome"],
                        bias_scores=sample_insert_sizes[sample_insert_sizes > 0],
                        bin_edges=insert_size_correction_bin_edges,
                    )

                    # Correct the mean shift
                    sample_cov[sample_insert_sizes > 0] = correct_bias(
                        coverages=sample_cov[sample_insert_sizes > 0],
                        correct_factors=insert_size_mean_correction_factors_collection[
                            chrom
                        ]["chromosome"],
                        bias_scores=sample_insert_sizes[sample_insert_sizes > 0],
                        bin_edges=insert_size_correction_bin_edges,
                    )

                    messenger(
                        f"NaNs after correction: {np.isnan(sample_cov).sum()}",
                        indent=8,
                    )

                # Free up memory
                del sample_cov_raw_counts

                # Normalize large-scale average shifts
                # e.g. due to copy number alterations
                # Must come after insert size correction!
                # Scaling by _mean_ is probably the best version
                # as it keeps the relative spread and keeps
                # values positive (last part not too important at this stage)
                messenger("Normalizing megabin averages")
                with timer.time_step(indent=4):
                    # The `megabin_offset_combination_averages` has the centering average for
                    # all combinations of the megabin offsets (so one average per stride-sized bin)
                    # Can be used to plot a smoothed average across the genome
                    # TODO: Improve this comment for clarity :-)
                    (
                        sample_cov,
                        megabin_offset_combination_averages,
                    ) = normalize_megabins(
                        df=pd.DataFrame(
                            {
                                "chromosome": chrom,
                                "start": sample_start_coordinates,
                                "coverage": sample_cov,
                            }
                        ),
                        mbin_size=MEGABIN_SIZES[0],
                        stride=MEGABIN_SIZES[1],
                        old_col="coverage",
                        new_col="coverage",
                        center=None,
                        scale="mean",
                        copy=False,
                        return_coverage=True,
                    )

                    megabin_offset_combination_averages_collection[chrom] = (
                        megabin_offset_combination_averages
                    )

                # Update coverage variance
                stats_calculator.add_data(x=sample_cov)

                # Extract indices of the bins that overlap with consensus sites
                consensus_overlap_indices = np.nonzero(consensus_overlap)[0]
                messenger(
                    f"Found {len(consensus_overlap_indices)} consensus site bins",
                )

                # Update calculator for consensus sites
                (
                    _,
                    r_calculators["consensus"],
                    _,
                ) = _update_r_calculator(
                    sample_cov=sample_cov,
                    cell_type_cov=consensus_overlap,
                    cell_type="consensus",
                    r_calculator=r_calculators["consensus"],
                    exclude_indices=None,  # Already excluded
                )

                # Remove consensus site bins from sample coverage array
                sample_cov = np.delete(sample_cov, consensus_overlap_indices)

                # Update calculators for all cell types
                res = Parallel(n_jobs=n_jobs)(
                    delayed(_update_r_calculator)(
                        sample_cov=sample_cov,
                        path=cell_type_chromosome_beds[f"{cell_type}_{chrom}"],
                        cell_type=cell_type,
                        r_calculator=r_calculators[cell_type],
                        exclude_indices=exclude_indices,
                        consensus_indices=consensus_overlap_indices,
                    )
                    for cell_type in cell_type_paths.keys()
                )

                for r in res:
                    (cell_type, r_calculator, _) = r
                    r_calculators[cell_type] = r_calculator

            # Free up memory
            del (
                sample_cov,
                sample_insert_sizes,
                sample_start_coordinates,
                consensus_overlap,
                consensus_overlap_indices,
            )
            gc.collect()

    messenger("Initializing output array")

    # Prepare features array
    # 0) Pearson R and 1) p-value, 2) within-cell-mask fraction, 3) cosine similarity
    # 4) x_sum, 5) y_sum, 6) x_squared_sum, 7) y_squared_sum, 8) xy_sum, 9) n

    num_features = 10

    features_arr = np.zeros(
        shape=(
            num_features,
            (len(cell_type_paths) + 1),  # +1 is consensus
        ),
        dtype=np.float64,
    )

    messenger("Calculating Pearson R per cell_type")

    # Calculate features across the genome
    # 0) Pearson's R
    # 1) and its p-value
    # 2) The normalized dot product
    # 3) Cosine Similarity
    # And the terms used:
    # 4) x_sum
    # 5) y_sum
    # 6) x_squared_sum
    # 7) y_squared_sum
    # 8) xy_sum
    # 9) n
    with timer.time_step(indent=4, name_prefix="calc_genome_features"):
        for cell_type_idx, cell_type in zip(
            cell_type_to_idx["idx"],
            cell_type_to_idx["cell_type"],
        ):
            calculator = r_calculators[cell_type]
            r, p = calculator.pearson_r
            fraction_within = calculator.xy_sum / calculator.n

            # Add features to array
            feature_values = [
                r,
                p,
                fraction_within,
                calculator.cosine_similarity,
                calculator.x_sum,
                calculator.y_sum,
                calculator.x_squared_sum,
                calculator.y_squared_sum,
                calculator.xy_sum,
                calculator.n,
            ]
            for i, val in enumerate(feature_values):
                features_arr[i, cell_type_idx] = val

    messenger("Saving the output to disk")

    # Write array with values
    np.save(output_paths.dataset, features_arr)

    # Save coverage stats
    if output_paths.coverage_stats is not None:
        with open(output_paths.coverage_stats, "w") as fp:
            json.dump(stats_calculator.stats, fp)

    # Write GC correction files
    all_gc_factors = np.vstack(
        [
            gc_correction_factors_collection[chrom][band]
            for chrom in chroms_ordered
            for band in sorted(gc_correction_factors_collection[chrom].keys())
        ]
    )
    if output_paths.gc_correction_factors is not None:
        np.save(output_paths.gc_correction_factors, all_gc_factors)
    if output_paths.gc_bin_midpoints is not None:
        np.save(output_paths.gc_bin_midpoints, gc_bin_midpoints)

    # Identifiers for indexing gc correction factors arrays
    gc_factor_ids = pd.DataFrame(
        [
            (chrom, band)
            for chrom in chroms_ordered
            for band in sorted(gc_correction_factors_collection[chrom].keys())
        ],
        columns=["chromosome", "band"],
    )
    if output_paths.gc_factor_ids is not None:
        gc_factor_ids.to_csv(output_paths.gc_factor_ids, index=False)

    # Write insert size correction files

    # Identifiers for indexing correction factors arrays
    insert_size_factor_ids = pd.DataFrame(
        [
            (chrom, band)
            for chrom in chroms_ordered
            for band in sorted(
                insert_size_noise_correction_factors_collection[chrom].keys()
            )
        ],
        columns=["chromosome", "band"],
    )
    if output_paths.insert_size_factor_ids is not None:
        insert_size_factor_ids.to_csv(output_paths.insert_size_factor_ids, index=False)

    if output_paths.insert_size_bin_midpoints is not None:
        np.save(output_paths.insert_size_bin_midpoints, insert_size_bin_midpoints)

    # Save optimal fitting parameters from insert size correction

    if output_paths.insert_size_optimal_params is not None:

        def identify_params_dicts(d, chrom, band):
            d["chromosome"] = chrom
            d["band"] = band
            return d

        pd.DataFrame(
            [
                identify_params_dicts(
                    insert_size_optimal_params_collection[chrom][band], chrom, band
                )
                for chrom in chroms_ordered
                for band in sorted(insert_size_optimal_params_collection[chrom].keys())
            ]
        ).to_csv(output_paths.insert_size_optimal_params, index=False)

    # Save each of the correction factors and estimated biases
    for insert_size_name, insert_size_factor_collection in [
        (
            "insert_size_noise_correction_factors",
            insert_size_noise_correction_factors_collection,
        ),
        (
            "insert_size_skewness_correction_factors",
            insert_size_skewness_correction_factors_collection,
        ),
        (
            "insert_size_mean_correction_factors",
            insert_size_mean_correction_factors_collection,
        ),
        (
            "insert_size_observed_bias",
            insert_size_observed_bias_collection,
        ),
        (
            "insert_size_target_bias",
            insert_size_target_bias_collection,
        ),
    ]:
        # Correction factors
        all_insert_size_factors = np.vstack(
            [
                insert_size_factor_collection[chrom][band]
                for chrom in chroms_ordered
                for band in sorted(insert_size_factor_collection[chrom].keys())
            ]
        )
        _path = getattr(output_paths, insert_size_name)
        if _path is not None:
            np.save(
                _path,
                all_insert_size_factors,
            )

    # Write megabin centering/scaling parameters
    all_megabin_offset_combinations = pd.concat(
        megabin_offset_combination_averages_collection, ignore_index=True, axis=0
    )
    if output_paths.megabin_normalization_offset_combinations is not None:
        all_megabin_offset_combinations.to_csv(
            output_paths.megabin_normalization_offset_combinations, index=False
        )

    if output_paths.feature_readme is not None:
        with open(output_paths.feature_readme, "w") as f:
            f.write(
                """The features in the dataset are as follows:
        0) Pearson's R
        1) and its p-value
        2) The normalized dot product (i.e. within-mask fraction)
        3) Cosine Similarity
        And the terms used:
        4) x_sum
        5) y_sum
        6) x_squared_sum
        7) y_squared_sum
        8) xy_sum
        9) n
                """
            )
