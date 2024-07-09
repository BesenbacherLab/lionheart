from typing import Optional, Tuple, Union
import numpy as np
from scipy.stats import binned_statistic


def calculate_correction_factors(
    bias_scores: np.ndarray,
    coverages: np.ndarray,
    bin_edges: np.ndarray,
    nan_extremes: Union[str, bool] = True,
    div_by_mean: bool = True,
    smoothe_factors: bool = True,
    smoothing_settings: dict = {
        "kernel_size": 5,
        "kernel_type": "gaussian",
        "kernel_std": 1.0,
    },
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a correction factor for each bias score bin
    that will make the coverages independent of the bias scores.
    I.e. when plotting `coverages` by `bias_scores` the corrected
    version should be a flat line.

    Examples of bias scores are GC contents or mappability scores.

    Parameters
    ----------
    bias_scores
        Array with bias score that determine the bins to
        average `coverages` in, given the `bin_edges`.
    coverages
        Array with the coverages to average within bins.
    bin_edges
        The values in `bias_scores` that the bins are split at.
    div_by_mean
        Whether to divide the correction factor by its mean
        to center the factors around 1.
    smoothe_factors
        Whether to apply kernel smoothing to the signal.
    smoothing_settings
        Dict with smoothing settings. Should contain the
        following settings:
            kernel_size : int
            kernel_type : str (one of {"gaussian", "uniform"})
            kernel_std : float (when `kernel_type="gaussian"`)

    Returns
    -------
    `numpy.ndarray`
        Bin midpoints. The midpoint indices of the bins, e.g. for plotting.
    `numpy.ndarray`
        Bin correction factors. The calculated correction factors (bin averages).
        The extreme left and right bins are NaNs, as well as any
        factors calculated to be 0.
    """
    midpoints, correct_factors = average_bins(
        x=bias_scores,
        y=coverages,
        bin_edges=bin_edges,
        nan_zeroes=True,
        nan_extremes=nan_extremes,
    )

    # Apply smoothing to the correction factor
    if smoothe_factors:
        correct_factors = smoothe_signal(x=correct_factors, **smoothing_settings)

    if div_by_mean:
        correct_factors /= np.nanmean(correct_factors)

    return midpoints, correct_factors


def correct_bias(
    coverages: np.ndarray,
    correct_factors: np.ndarray,
    bias_scores: Optional[np.ndarray] = None,
    bin_indices: Optional[np.ndarray] = None,
    bin_edges: Optional[np.ndarray] = None,
    normalize_correct_factors: bool = True,
    truncate_negative_coverages: bool = True,
) -> np.ndarray:
    """
    Correct `coverages` by binned correction factors.

    Examples of bias scores are GC contents or mappability scores.

    "Bias score bins" refers to the binned observed values of bias scores
    e.g. [0, 0.1, 0.2..., 1.0] of which `correct_factors` contains the
    (normalized, smoothed) average coverage and `bin_edges` specifies
    the edges of.

    Parameters
    ----------
    coverages
        Array with the values to correct.
        E.g. an array with fragment coverages.
    correct_factors
        Array with a value for each bin (see `bin_indices`)
        to divide `coverages` with.
    bias_scores
        Array with bias scores for each value in `coverages`.
        Used to identify which bias score bin each the values are part of.
        Only one of {`bias_scores`, `bin_indices`} should be specified.
    bin_indices
        Array with the bias score bin indices.
        Should match the bins in `correct_factors`.
        Only one of {`bias_scores`, `bin_indices`} should be specified.
    bin_edges
        The values in `bias_scores` that the bins are split at.
    truncate_negative_coverages
        Whether to truncate `coverages < 0` to 0.
        The current correction style works best for non-negative numbers.

    Returns
    -------
    `numpy.ndarray`
        Corrected `coverages` values.
    """
    assert isinstance(coverages, np.ndarray)
    assert isinstance(correct_factors, np.ndarray)

    assert (
        sum([bin_indices is not None, bias_scores is not None]) == 1
    ), "Exactly one of `bin_indices` and `bias_scores` should be specified."
    if bias_scores is not None:
        assert bin_edges is not None and isinstance(bin_edges, np.ndarray)
        assert isinstance(bias_scores, np.ndarray)
    else:
        assert isinstance(bin_indices, np.ndarray)
        assert (
            bin_edges is None
        ), "`bin_edges` should only be specified along with `bias_scores`."

    if bias_scores is not None:
        bin_indices = np.digitize(bias_scores, bins=bin_edges, right=False) - 1

        # Truncate bin indices
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices > (len(bin_edges) - 2)] = len(bin_edges) - 2

    # Normalize the correction factors
    # So each value is around 1
    if normalize_correct_factors:
        correct_factors /= np.nanmean(correct_factors)

    # Prepare array for correction factors
    corrections = np.zeros_like(bin_indices, dtype=np.float32)

    # Find the unique bin indices
    unique_bin_indices = list(np.unique(bin_indices))

    # Find correction factor for each bin
    for binn in unique_bin_indices:
        corrections[bin_indices == binn] = correct_factors[binn]

    # Avoid zero-division
    corrections[corrections == 0] = np.nan

    # Truncate negative coverages
    if truncate_negative_coverages:
        coverages[coverages < 0] = 0.0

    # Apply correction
    if coverages.ndim == 3:
        for end_type in range(coverages.shape[1]):
            coverages[:, end_type, :] /= corrections
    else:
        coverages /= corrections

    return coverages


def bin_midpoints(edges: np.ndarray) -> np.ndarray:
    """
    Calculates midpoints for each bin based on the edges.
    This is useful on the x-axis when plotting the
    averaged/corrected values.

    Parameters
    ----------
    edges
        Array with bin edge values.

    Returns
    -------
    `numpy.ndarray`
        Bin midpoints.
        I.e. starting edge + (1/2 * diff to end point).
    """
    diffs = np.diff(edges) / 2
    return edges[:-1] + diffs


def average_bins(
    x: np.ndarray,
    y: np.ndarray,
    bin_edges: np.ndarray,
    nan_zeroes: bool = False,
    nan_extremes: Union[str, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate averages of `y` within bins of `x` given by `bin_edges`.

    Uses `numpy.nanmean` to ignore NaNs.

    Parameters
    ----------
    x
        Array with values that determine the bins
        to average `y` in, given the `bin_edges`.
    y
        Array with the values to average within bins.
    bin_edges
        The values in `x` that the bins are split at.
    nan_zeroes
        Whether to set zero-means to NaNs.
    nan_extremes
        Whether to set the averages for the extreme left
        and/or extreme right bins to NaN.
        When `True`, both extremes become NaN.
        When a string from {'min','max'}, the selected
        extreme becomes NaN.

    Returns
    -------
    `numpy.ndarray`
        Bin midpoints. The midpoint indices of the bins, e.g. for plotting.
    `numpy.ndarray`
        Bin averages. The calculated bin averages.
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(bin_edges, np.ndarray)
    assert x.ndim == 1, f"`x` must have exactly one dimension. Had {x.ndim}."
    assert y.ndim == 1, f"`y` must have exactly one dimension. Had {y.ndim}."
    assert (
        bin_edges.ndim == 1
    ), f"`bin_edges` must have exactly one dimension. Had {bin_edges.ndim}."
    assert isinstance(nan_extremes, (str, bool))
    if isinstance(nan_extremes, str) and nan_extremes not in ["min", "max"]:
        raise ValueError(
            "When `nan_extremes` is supplied as string, it must be either "
            f"'min' or 'max'. Got: '{nan_extremes}'."
        )

    bin_edges = np.unique(bin_edges)
    binned = binned_statistic(x=x, values=y, statistic=np.nanmean, bins=bin_edges)
    averages = binned.statistic
    edges = binned.bin_edges
    midpoints = bin_midpoints(edges)

    # Set 0's to NaN
    if nan_zeroes:
        averages[averages == 0.0] = np.nan

    # Set first and/or last bin to NaN (if specified)
    averages = set_extremes_to_nan(averages, nan_extremes=nan_extremes)

    return midpoints, averages


def set_extremes_to_nan(x: np.ndarray, nan_extremes: Union[bool, str]) -> np.ndarray:
    if nan_extremes:  # Either True or a nonempty string
        # Set first and/or last bin to NaN
        if isinstance(nan_extremes, str):
            if nan_extremes == "min":
                x[0] = np.nan
            elif nan_extremes == "max":
                x[-1] = np.nan
        else:
            x[0] = np.nan
            x[-1] = np.nan
    return x


def smoothe_signal(
    x: np.ndarray,
    kernel_size: int = 5,
    kernel_type: str = "gaussian",
    kernel_std: float = 1.0,
) -> np.ndarray:
    """
    Apply kernel smoothing to a numpy array.

    Parameters
    ----------
    x
        Array with values to smoothe.
    kernel_size
        Size of the kernel.
    kernel_type
        One of {"uniform", "gaussian"}.
        `uniform`: All weights in the kernel have the same weight.
        `gaussian`: A Gaussian kernel is applied such that the
            center value has the most weight. Set the spread with `kernel_std`.
    kernel_std
        Standard deviation of the Gaussian kernel when `kernel_type="gaussian"`.
    """
    if kernel_type == "uniform":
        kernel = np.ones(kernel_size) / kernel_size
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel_1d(n=kernel_size, std=kernel_std)
    else:
        raise ValueError(f"Unknown `kernel_type`: {kernel_type}")

    # Apply smoothing
    smooth_x = np.convolve(x, kernel, mode="same")

    # Set new NaNs to value in original data
    smooth_x[np.isnan(smooth_x)] = x[np.isnan(smooth_x)]

    return smooth_x


def gaussian_kernel_1d(n: int = 5, std: float = 1.0) -> np.ndarray:
    """
    Return a Gaussian kernel.
    """
    ax = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    return gauss / np.sum(gauss)
