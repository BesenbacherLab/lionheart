from typing import List, Callable, Tuple, Union
import numpy as np
from scipy.stats import t
from scipy.special import erf
from scipy.optimize import minimize

from lionheart.features.correction.correction import (
    average_bins,
    smoothe_signal,
    set_extremes_to_nan,
    correct_bias,
)


def calculate_insert_size_correction_factors(
    coverages: np.ndarray,
    insert_sizes: np.ndarray,
    bin_edges: np.ndarray,
    base_sigma: float = 8.026649608460776,
    df: int = 5,
    final_mean_insert_size: float = 166.0,
    nan_extremes: Union[str, bool] = False,
) -> dict:
    """

    nan_extremes
        Whether to set the correction factors for the extreme left
        and/or extreme right bins to NaN.
        When `True`, both extremes become NaN.
        When a string from {'min','max'}, the selected
        extreme becomes NaN.
    """
    coverages = coverages.copy()
    insert_sizes = insert_sizes.copy()
    bin_edges = bin_edges.copy()

    # Truncate any negative coverages (from prior corrections)
    coverages[coverages < 0] = 0

    assert isinstance(nan_extremes, (str, bool))
    if isinstance(nan_extremes, str) and nan_extremes not in ["min", "max"]:
        raise ValueError(
            "When `nan_extremes` is supplied as string, it must be either "
            f"'min' or 'max'. Got: '{nan_extremes}'."
        )

    # Whether to include the extreme bins
    # in the objective function
    slicer_fn = lambda x: x  # noqa: E731
    if nan_extremes:
        if isinstance(nan_extremes, str):
            if nan_extremes == "min":
                slicer_fn = lambda x: x[1:]  # noqa: E731
            else:
                slicer_fn = lambda x: x[:-1]  # noqa: E731
        else:
            slicer_fn = lambda x: x[1:-1]  # noqa: E731

    lower_bound, upper_bound = min(bin_edges), max(bin_edges)

    # Fit skewed student's t distribution
    (
        fitted_dist,
        observed_dist,
        bin_midpoints_,
        optimal_params,
    ) = _calculate_composite_distribution(
        coverages=coverages,
        insert_sizes=insert_sizes,
        bin_edges=bin_edges,
        base_sigma=base_sigma,
        df=df,
        slicer_fn=slicer_fn,
    )

    # Calculate correction factor for the noise
    noise_correction = (observed_dist / np.nanmean(observed_dist)) / (
        fitted_dist / np.nanmean(fitted_dist)
    )
    noise_correction /= np.nanmean(noise_correction)

    # Calculate correction factor for the skew
    inverse_skewness_weights = _calculate_inverse_skewness(
        bin_midpoints_=bin_midpoints_, skewness=optimal_params["skewness"]
    )

    if not int(np.sum(~np.isnan(inverse_skewness_weights))):
        raise ValueError("No elements in `inverse_skewness_weights` that was not NaN.")

    # For the last correction step (of means)
    # we need to first correct the coverages
    # and then refit a distribution

    # Correct noise
    coverages[insert_sizes > 0] = correct_bias(
        coverages=coverages[insert_sizes > 0],
        correct_factors=noise_correction,
        bias_scores=insert_sizes[insert_sizes > 0],
        bin_edges=bin_edges,
    )

    # Correct skewness
    coverages[insert_sizes > 0] = correct_bias(
        coverages=coverages[insert_sizes > 0],
        correct_factors=inverse_skewness_weights,
        bias_scores=insert_sizes[insert_sizes > 0],
        bin_edges=bin_edges,
    )

    # Calculate first corrected bias distribution for output
    # To allow plotting the correction steps
    _, first_corrected_dist = average_bins(
        x=insert_sizes[insert_sizes > 0],
        y=coverages[insert_sizes > 0],
        bin_edges=bin_edges,
    )
    first_corrected_dist /= np.nanmean(first_corrected_dist)

    # Fit skewed student's t distribution to
    # noise- and skewness corrected coverages
    (
        refitted_dist,
        intially_corrected_dist,
        _,
        refit_optimal_params,
    ) = _calculate_composite_distribution(
        coverages=coverages,
        insert_sizes=insert_sizes,
        bin_edges=bin_edges,
        base_sigma=base_sigma,
        df=df,
        skewness_loss_weight=0.1,  # Heavy penalization!
        slicer_fn=slicer_fn,
    )

    # Use the optimal parameters from the refit
    # to generate the final target distribution
    # with a given mean and zero skewness
    target_dist = _calculate_mixture_distribution(
        bin_midpoints_=bin_midpoints_,
        coverages=coverages,
        base_sigma=base_sigma,
        mean_insert_size=final_mean_insert_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        df=df,
        skewness=0.0,
        scale_multiplier=refit_optimal_params["scale_multiplier"],
    )
    target_dist /= np.nanmean(target_dist)

    # Calculate correction factor for the mean shift
    mean_correction = (
        intially_corrected_dist / np.nanmean(intially_corrected_dist)
    ) / (target_dist / np.nanmean(target_dist))
    mean_correction /= np.nanmean(mean_correction)

    # Set extreme bins to NaN (when specified)
    noise_correction = set_extremes_to_nan(noise_correction, nan_extremes=nan_extremes)
    inverse_skewness_weights = set_extremes_to_nan(
        inverse_skewness_weights, nan_extremes=nan_extremes
    )
    mean_correction = set_extremes_to_nan(mean_correction, nan_extremes=nan_extremes)

    # Add optimal refitting parameters to optimal params dict
    optimal_params = {f"initial_fit__{key}": val for key, val in optimal_params.items()}
    refit_optimal_params = {
        f"refit__{key}": val for key, val in refit_optimal_params.items()
    }
    optimal_params.update(refit_optimal_params)

    return {
        "noise_correction_factor": noise_correction,
        "skewness_correction_factor": inverse_skewness_weights,
        "mean_correction_factor": mean_correction,
        "observed_bias": observed_dist,
        "first_corrected_bias": first_corrected_dist,
        "target_bias": target_dist,
        "first_fitted_bias": fitted_dist,
        "second_fitted_bias": refitted_dist,
        "bin_midpoints": bin_midpoints_,
        "optimal_fit_params": optimal_params,
    }


def _calculate_composite_distribution(
    coverages: np.ndarray,
    insert_sizes: np.ndarray,
    bin_edges: np.ndarray,
    base_sigma: float,
    df: int = 5,
    skewness_loss_weight: float = 0.005,
    slicer_fn: Callable = lambda x: x,  # Could use this to optimize distance around the peak only
    final_smoothing_settings: dict = {
        "kernel_size": 5,
        "kernel_type": "gaussian",
        "kernel_std": 1.0,
    },
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    # Find upper and lower bounds for the truncated distribution
    lower_bound, upper_bound = min(bin_edges), max(bin_edges)

    # Calculate observed bias distribution
    bin_midpoints_, observed_dist = average_bins(
        x=insert_sizes[insert_sizes > 0],
        y=coverages[insert_sizes > 0],
        bin_edges=bin_edges,
    )

    observed_dist /= np.nanmean(observed_dist)

    composite_distribution, optimal_params = _optimize(
        bin_midpoints_=bin_midpoints_,
        coverages=np.round(coverages[(coverages > 0.5) & (insert_sizes > 0)]).astype(
            np.int64
        ),
        observed_dist=observed_dist,
        base_sigma=base_sigma,
        df=df,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        start_mean_insert_size=insert_sizes[
            (coverages > 0.5) & (insert_sizes > 0)
        ].mean(),
        start_skewness=-0.5,
        skewness_loss_weight=skewness_loss_weight,
        slicer_fn=slicer_fn,
    )

    # Smoothe the fitted mixture distribution
    composite_distribution = smoothe_signal(
        composite_distribution / np.nanmean(composite_distribution),
        **final_smoothing_settings,
    )

    return (
        composite_distribution,
        observed_dist,
        bin_midpoints_,
        optimal_params,
    )


def _calculate_mixture_distribution(
    bin_midpoints_: np.ndarray,
    coverages: np.ndarray,
    base_sigma: float,
    mean_insert_size: float = 166,
    lower_bound: int = 100,
    upper_bound: int = 220,
    df: int = 5,
    skewness: float = 0.0,
    scale_multiplier: float = 1,
) -> np.ndarray:
    # Calculate the frequency of each coverage value to use as weights
    coverage_counts = np.bincount(coverages[~np.isnan(coverages)].astype(int))
    coverage_probs = coverage_counts / np.nansum(coverage_counts)

    # Initialize the composite distribution
    composite_distribution = np.zeros_like(bin_midpoints_, dtype=float)

    # Calculate the scaling factors for the t-distribution within the bounds
    t_scale_factor = t.cdf(
        upper_bound, df, loc=mean_insert_size, scale=base_sigma * scale_multiplier
    ) - t.cdf(
        lower_bound, df, loc=mean_insert_size, scale=base_sigma * scale_multiplier
    )

    # Generate the distribution for each available coverage value
    # And sum them, weighted by the probability for each coverage value
    for coverage, probability in enumerate(coverage_probs):
        if probability > 0 and coverage > 0:  # Make sure to skip coverage of 0
            # Adjust the standard deviation for the coverage depth and apply the scale multiplier
            # The bin-wise spread of insert sizes is inversely proportional to the
            # square root of the coverage depth (Central Limit Theorem)
            depth_sigma = base_sigma / np.sqrt(coverage) * scale_multiplier

            # Calculate the t-distribution PDF
            distribution = _skewed_t_pdf(
                bin_midpoints_=bin_midpoints_,
                df=df,
                loc=mean_insert_size,
                scale=depth_sigma,
                skewness=skewness,
            )

            # Truncate the distribution manually
            distribution[bin_midpoints_ < lower_bound] = 0
            distribution[bin_midpoints_ > upper_bound] = 0

            # Normalize the distribution for the truncated range
            distribution /= t_scale_factor * depth_sigma

            # Weight the distribution by its probability and add to the composite
            composite_distribution += distribution * probability

    # Normalize the composite distribution
    composite_distribution /= np.nanmean(composite_distribution)

    return composite_distribution


def _objective_function(
    opt_args: List[float],
    bin_midpoints_: np.ndarray,
    coverages: np.ndarray,
    observed_dist: np.ndarray,
    base_sigma: float,
    lower_bound: int,
    upper_bound: int,
    df: int,
    skewness_loss_weight: float,
    slicer_fn: Callable = lambda x: x,
):
    # Unpack the arguments for optimization
    scale_multiplier, skewness, mean_insert_size = opt_args

    # Fit the distribution with the given optimization parameters
    fitted_dist = _calculate_mixture_distribution(
        bin_midpoints_=bin_midpoints_,
        coverages=coverages,
        base_sigma=base_sigma,
        mean_insert_size=mean_insert_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        df=df,
        skewness=skewness,
        scale_multiplier=scale_multiplier,
    )

    # Normalize to mean=1
    fitted_dist /= np.nanmean(fitted_dist)
    observed_dist /= np.nanmean(observed_dist)

    # Calculate the error between the fitted and observeds distributions
    error = (
        np.nanmean((slicer_fn(fitted_dist) - slicer_fn(observed_dist)) ** 2)
        + skewness_loss_weight * skewness**2
    )

    return error


def _optimize(
    bin_midpoints_: np.ndarray,
    coverages: np.ndarray,
    observed_dist: np.ndarray,
    base_sigma: float,
    df: int,
    lower_bound: int,
    upper_bound: int,
    start_mean_insert_size: float = 166.0,
    start_scale_multiplier: float = 8.0,
    start_skewness: float = 0.0,
    skewness_loss_weight: float = 0.005,
    slicer_fn: Callable = lambda x: x,
):
    # Initialize optimization parameters
    initial_guess = [start_scale_multiplier, start_skewness, start_mean_insert_size]

    # Use the minimize function to find the optimal scale_multiplier
    result = minimize(
        _objective_function,
        initial_guess,
        args=(
            bin_midpoints_,
            coverages,
            observed_dist,
            base_sigma,
            lower_bound,
            upper_bound,
            df,
            skewness_loss_weight,
            slicer_fn,
        ),
    )

    # The optimal parameters for the sample
    optimal_scale_multiplier, optimal_skewness, optimal_mean_insert_size = result.x
    optimal_params = {
        "scale_multiplier": optimal_scale_multiplier,
        "skewness": optimal_skewness,
        "mean_insert_size": optimal_mean_insert_size,
    }

    # Use the optimal parameters to generate the best fitting distribution
    best_fitted_distribution = _calculate_mixture_distribution(
        bin_midpoints_=bin_midpoints_,
        coverages=coverages,
        base_sigma=base_sigma,
        mean_insert_size=optimal_mean_insert_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        df=df,
        skewness=optimal_skewness,
        scale_multiplier=optimal_scale_multiplier,
    )

    # Normalize to mean=1
    best_fitted_distribution /= np.nanmean(best_fitted_distribution)

    return best_fitted_distribution, optimal_params


def _skewed_t_pdf(
    bin_midpoints_: np.ndarray, df: int, loc: float, scale: float, skewness: float
) -> np.ndarray:
    """
    Skewed Student's t probability density function.

    As implemented by chatGPT4 with the note that it may not
    be the most theoretically sound approach but should work
    in practice for simple things like our case.
    """
    # Calculate the t-distribution PDF
    t_dist_pdf = t.pdf(bin_midpoints_, df, loc, scale)

    # Calculate the skewness weight
    skewness_weight = 1 + erf(
        (skewness * (bin_midpoints_ - loc)) / (scale * np.sqrt(2))
    )

    # Apply the skewness weight to the t-distribution PDF
    skewed_pdf = t_dist_pdf * skewness_weight

    # Normalize the skewed PDF
    skewed_pdf /= np.trapz(skewed_pdf, bin_midpoints_)

    return skewed_pdf


def _calculate_inverse_skewness(
    bin_midpoints_: np.ndarray, skewness: float
) -> np.ndarray:
    return bin_midpoints_ * skewness + np.mean(bin_midpoints_) * (1 - skewness) + 1
