from typing import List, Optional, Union, Tuple
import warnings
import numpy as np
import pandas as pd

from utipy import random_alphanumeric


# TODO Should stride be called offset_size?


def normalize_megabins(
    df: pd.DataFrame,
    mbin_size: int,
    stride: Optional[int] = None,
    old_col: str = "coverage",
    new_col: str = "coverage",
    center: Optional[str] = None,
    scale: Optional[str] = None,
    copy: bool = True,
    clip_above_quantile: Optional[float] = None,
    return_coverage: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, pd.DataFrame]]:
    """
    Normalize coverages in megabins, E.g., to reduce
    the effect of copy number alterations.

    Either:
        1) standardize with centering of the mean or median
           and (optionally) scaling by the standard deviation or IQR.
        2) simply scale by the mean or median.

    The statistical descriptors are calculated
    in megabins with a series of offsets (`stride`).
    NOTE: The statistical descriptors are of the <=0.99 quantile data (clipped to remove big outliers)

    NOTE: When both center and scale are `None`, the coverage is not touched.
    I this case, you can use this function to just get the stride-sized bin measures.

    Example of mega bin creation with `mbin_size=3` and `stride=1`: (TODO: Adapt to full normalization version)

        (Note that we would normally expect argument values on the scale
        of 5 million and 0.5 million bases).

        Bin factors:
        [1, 1, 1, 2, 2, 2, 3, 3, 3]
        [1, 1, 2, 2, 2, 3, 3, 3, 4]
        [1, 2, 2, 2, 3, 3, 3, 4, 4]

        For each bin in each bin factor, we calculate the average of
        the corresponding coverages (Note: completely made up numbers):  # TODO: Use real numbers

        [20.3, 20.3, 20.3, 3.2,  3.2, 3.2, 8.4, 8.4, 8.4]
        [22.1, 22.1, 6.4,  6.4,  6.4, 7.5, 7.5, 7.5, 11.8]
        [27.2, 15.3, 15.3, 15.3, 7.1, 7.1, 7.1, 9.8, 9.8]

        And the final average to center for each coverage bin is
        then the average of its three averages:

        [23.3, 19.2, 14.0, 8.3,  5.6, 5.9, 7.7, 8.6, 10.0]

        The centering is done by subtracting these means
        from the coverages.

    """

    if center is None and scale is None:
        raise ValueError("At least one of {`center`, `scale`} must be specified.")
    assert center is None or center.lower() in ["mean", "median"]
    assert scale is None or scale.lower() in ["std", "iqr", "mean", "median"]
    if scale is not None and scale in ["mean", "median"] and center is not None:
        raise ValueError(
            "When `scale` is either 'mean' or 'median', `center` should be `None`."
        )
    if scale is not None and scale in ["std", "iqr"] and center is None:
        warnings.warn(
            f"Megabin scaling by `{scale}` without centering may not be meaningful."
        )

    measures = []
    if center is not None:
        center = center.lower()
        measures.append(center)
    if scale is not None:
        scale = scale.lower()
        measures.append(scale)

    # Make sure we don't alter original data frame
    if copy:
        df = df.copy()

    # Calculate the statistical descriptors in megabins
    df_aggregates, mbin_offset_combination_averages = describe_megabins(
        df=df,
        mbin_size=mbin_size,
        stride=stride,
        old_col=old_col,
        clip_above_quantile=clip_above_quantile,
        measures=measures,
        copy=False,
    )

    # Create new normalized column
    df.loc[:, new_col] = df[old_col]
    if center is not None:
        if center == "mean":
            # Center column by mean
            df[new_col] -= df_aggregates["mbin_overall_mean"]
        elif center == "median":
            # Center column by median
            df[new_col] -= df_aggregates["mbin_overall_median"]
    if scale is not None:
        if scale == "std":
            # Scale column by standard deviation
            df[new_col] /= df_aggregates["mbin_overall_std"]
        elif scale == "iqr":
            # Scale column by interquartile range
            df[new_col] /= df_aggregates["mbin_overall_iqr"]
        elif scale == "mean":
            # Scale column by average value
            df[new_col] /= df_aggregates["mbin_overall_mean"]
        elif scale == "median":
            # Scale column by interquartile range
            df[new_col] /= df_aggregates["mbin_overall_median"]

    # If specified, we return the coverage as a numpy array
    if return_coverage:
        return df[new_col].to_numpy(), mbin_offset_combination_averages

    return df, mbin_offset_combination_averages


def describe_megabins(
    df: pd.DataFrame,
    mbin_size: int,
    stride: Optional[int] = None,
    old_col: str = "coverage",
    clip_above_quantile: Optional[float] = None,
    measures: Optional[List[str]] = None,
    copy: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Check stride size
    if stride is None or stride == 0:
        stride = mbin_size
    elif stride < 0:
        raise ValueError("`stride` was negative.")

    # Make sure we don't alter original data frame
    if copy:
        df = df.copy()

    if measures is None:
        measures = ["mean", "median", "std", "iqr"]

    # We need to ensure the row order before and after binning
    # So we add a tmp index
    tmp_index_name = f"tmp__index__{random_alphanumeric(size=7)}"
    df.loc[:, tmp_index_name] = range(len(df))

    # Find megabin averages per chromosome
    df_aggregates: pd.DataFrame = (
        df.groupby(["chromosome"], group_keys=False)
        .apply(
            _calculate_mbin_parameters_for_chr,
            mbin_size=mbin_size,
            stride=stride,
            measures=measures,
            clip_above_quantile=clip_above_quantile,
            old_col=old_col,
        )
        .reset_index(drop=True)
    )

    # Ensure order is the same as previously
    # And remove tmp index
    df_aggregates = df_aggregates.sort_values([tmp_index_name]).drop(
        columns=[tmp_index_name]
    )

    # Remove tmp index from df as well
    # NOTE: Must be done in-place in case `df` is not a copy
    df.drop(columns=[tmp_index_name], inplace=True)

    # Get created columns with averages
    new_columns = list(set(df_aggregates.columns).difference(set(df.columns)))
    idx_columns = sorted([cname for cname in new_columns if "_idx" in cname])

    measure_to_columns = {
        measure: sorted([cname for cname in new_columns if f"_{measure}" in cname])
        for measure in measures
    }

    overall_mbin_aggregate_columns = []

    # Calculate the average averages for each interval (row)
    for param_name, param_columns in measure_to_columns.items():
        overall_param_column = f"mbin_overall_{param_name}"
        overall_mbin_aggregate_columns.append(overall_param_column)
        df_aggregates.loc[:, overall_param_column] = df_aggregates.loc[
            :, param_columns
        ].mean(axis=1, skipna=True)

    # Create a data frame with all the megabin combinations across offsets
    # That is, one overall average for each striding/offsetting (~ genome / stride bins)
    # E.g., for plotting the "smoothed" averages across the genome
    mbin_offset_combination_averages = (
        df_aggregates.groupby(["chromosome"] + idx_columns)
        .head(1)
        .reset_index()
        .loc[:, ["chromosome", "start"] + overall_mbin_aggregate_columns]
    )

    return df_aggregates, mbin_offset_combination_averages


def _calculate_mbin_parameters_for_chr(
    df_for_chr: pd.DataFrame,
    mbin_size: int,
    stride: int,
    old_col: str,
    measures: List[str],
    clip_above_quantile: Optional[float] = None,
) -> pd.DataFrame:
    # Extract range of start coordinates
    min_start = int(df_for_chr["start"].min())
    max_start = int(df_for_chr["start"].max())

    # The number of totals strides/offsets
    num_stridings = int(np.ceil(mbin_size / stride))

    # Add column of megabin averages for each stride
    for striding in range(num_stridings):
        # We start prior to the min start coordinate
        # to get same "smoothing" of the averages for the first bins
        start_offset = mbin_size - stride * striding if striding > 0 else 0
        df_for_chr = _bin_with_current_stride_start(
            df_for_chr=df_for_chr,
            mbin_size=mbin_size,
            first_start_pos=min_start - start_offset,
            max_start_pos=max_start,
            stride_id=striding,
            old_col=old_col,
            measures=measures,
            clip_above_quantile=clip_above_quantile,
        )

    return df_for_chr


def _bin_with_current_stride_start(
    df_for_chr: pd.DataFrame,
    mbin_size: int,
    first_start_pos: int,
    max_start_pos: int,
    stride_id: int,
    old_col: str,
    measures: List[str],
    clip_above_quantile: Optional[float] = None,
) -> pd.DataFrame:
    # NOTE: `first_start` can be negative if we
    # have "padding" to have the same resolution
    # in the beginning of the chromosome
    total_length_covered = max_start_pos - first_start_pos

    # We can go from just before the first existing bin to just after the last bin
    # So the first and last mbins may have fewer actual bins in them
    num_mbins = int(np.ceil(total_length_covered / mbin_size))

    # Find edges of megabins
    mbin_edges = [first_start_pos + mbin_size * bi for bi in range(num_mbins + 1)]

    # Names of created columns
    idx_col_name = f"mbin_{stride_id}_idx"
    measure_colnames = {measure: f"mbin_{stride_id}_{measure}" for measure in measures}

    # Find the megabin each bin (start coordinate) belongs to
    df_for_chr.loc[:, idx_col_name] = np.digitize(df_for_chr["start"], mbin_edges)

    # Prepare a small factory to get a clip+nan-aware function
    def _mk(fn):
        def _f(x):
            arr = x.to_numpy().astype(float)
            if clip_above_quantile is not None:
                uq = np.nanquantile(arr, clip_above_quantile)
                arr[arr > uq] = uq
            return fn(arr)

        return _f

    # Build our agg dict using the nan-versions of common functions
    agg_dict = {
        measure_colnames[m]: (old_col, _mk(_measure_to_fn[m])) for m in measures
    }
    stats = df_for_chr.groupby(idx_col_name).agg(**agg_dict)

    df_for_chr = df_for_chr.join(stats, on=idx_col_name)

    return df_for_chr


def naniqr(x: np.ndarray) -> float:
    q75, q25 = np.nanpercentile(x, [75, 25])
    return q75 - q25


_measure_to_fn = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "std": np.nanstd,
    "iqr": naniqr,
}


# TODO: Make zero-inflated mean somehow?
# Fit a zero-inflated distribution that gives a zero-adjusted mean
# Perhaps remove outliers by removing >=0.99 quantile before
# fitting the distribution, as we want the general tendency in the mbin
# See: Statistical Analysis of Zero-Inflated Nonnegative Continuous Data
# def zero_inflated_mean(arr, clip_above_quantile=0.99):
