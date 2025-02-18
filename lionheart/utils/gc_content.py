from numbers import Number
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def get_gc_content_all_intervals(
    bed_df: pd.DataFrame,
    tb,
    ignore_non_acgt: bool = False,
    handle_zero_division: Union[str, float] = "raise",
) -> np.ndarray:
    """
    Get GC content for each interval in a BED data frame.

    Parameters
    ----------
    bed_df
        `pandas.DataFrame` with intervals from a BED file.
        As read with `read_bed_as_df()`.
    tb
        2Bit file with reference genome to look up GC content in.
    ignore_non_acgt
        Whether to ignore Ns when calculating GC content fractions.
        `False`: Divide G+C counts by the length of the interval.
        `True`: Divide G+C counts by the sum of counts for all bases (G,C,A,T).
    handle_zero_division
        How to handle `ZeroDivisionError`s.
        Either provide the string 'raise' (simply raises the `ZeroDivisionError`)
        or provide a float value (e.g. `numpy.nan` or `0.0`).
        Zero-division happens when `end-start == 0` or
        the counts of ACGT bases are all 0.

    Returns
    -------
    `numpy.ndarray`
        GC content percentage (0-1) per interval.
    """

    # Extract GC contents for each interval
    gc_contents = bed_df.apply(
        lambda row: _get_interval_gc_content(
            tb=tb,
            chrom=row["chromosome"],
            start=row["start"],
            end=row["end"],
            ignore_non_acgt=ignore_non_acgt,
            handle_zero_division=handle_zero_division,
        ),
        axis=1,
    ).to_numpy()

    # Convert to numpy array
    return np.concatenate(gc_contents.tolist(), axis=0)


def _get_interval_gc_content(
    tb,
    chrom: str,
    start: int,
    end: int,
    ignore_non_acgt: bool = False,
    handle_zero_division: Union[str, float] = "raise",
) -> float:
    """
    Extract GC content from specific interval in a 2bit reference genome.

    Parameters
    ----------
    tb
        2Bit file with reference genome to look up GC content in.
    chrom
        Chromosome of the interval to get GC from.
    start
        Starting position of the interval to get GC content from.
        Inclusive (i.e. 0-based) position.
    end
        End position of the interval to get GC content from.
        Exclusive (i.e. 1-based) position.
    ignore_non_acgt
        Whether to ignore Ns when calculating GC content fractions.
        `False`: Divide G+C counts by the length of the interval.
        `True`: Divide G+C counts by the sum of counts for all bases (G,C,A,T).
    handle_zero_division
        How to handle `ZeroDivisionError`s.
        Either provide the string 'raise' (simply raises the `ZeroDivisionError`)
        or provide a float value (e.g. `numpy.nan` or `0.0`).
        Zero-division happens when `end-start == 0` or
        the counts of ACGT bases are all 0.

    Returns
    -------
    float
        GC content
    """
    base_contents = tb.bases(chrom, start, end, False)

    # Find denominator in the fraction calculation
    # Total number of bases
    denominator = end - start
    if ignore_non_acgt:
        # Use sum of base counts instead
        denominator = sum(base_contents.values())

    # Handle zero-division
    if denominator == 0 and not isinstance(handle_zero_division, str):
        if not isinstance(handle_zero_division, float):
            raise TypeError(
                "`handle_zero_division` must be either a string ('raise')"
                f" or a float. Had type: {type(handle_zero_division)}."
            )
        return handle_zero_division
    return (base_contents["G"] + base_contents["C"]) / denominator


def find_greedy_bin_edges(
    x: np.ndarray,
    num_bins: int,
    range_: Tuple[Number, Number] = (0, 1),
    add_edges: Optional[List[Number]] = None,
) -> np.ndarray:
    """
    Finds edges (values of `x`) that splits `x` into a number of equally sized bins
    (equal in number of elements; except first and last bins).

    The first edge is the first element in `range_` and the last edge
    is the last element in `range_`.

    NOTE: Only *unique edges* are returned. When many elements
    have the same values, this might results in the merging of
    bins. To avoid rounding error duplicates (e.g. 0.1 and 0.099999999...)
    the values are rounded to the 10th decimal prior to deduplication.

    Parameters
    ----------
    x
        Array with values to find bin edges for.
    num_bins
        The number of bins.
    range_
        A tuple with the min and max values.
        These are also the first and last edges.
    add_edges
        List of additional edges to add to the output.
        The edges are sorted.

    Returns
    -------
    `numpy.ndarray`
        *Unique* bin edges.
    """
    maxes = [a.max() for a in np.array_split(sorted(x), num_bins)]
    maxes = [range_[0]] + maxes
    maxes[-1] = range_[1]
    edges = np.unique(np.round(maxes, decimals=10))
    if add_edges is not None:
        assert isinstance(add_edges, list) and isinstance(add_edges[0], Number)
        edges = np.unique(list(edges) + add_edges)
    return edges
