#!/usr/bin/env python3

"""
Utility for calculating CDF probabilities in the outlier detection script without worrying about precision/overflow.
"""

import argparse
import numpy as np
from scipy.stats import poisson


def main():
    parser = argparse.ArgumentParser(
        description="Precompute tail probabilities for a ZIP model using a Poisson distribution."
    )
    parser.add_argument(
        "--mean",
        type=float,
        required=True,
        help="Mean for the Poisson part of the model (e.g., 18.5883)",
    )
    parser.add_argument(
        "--p_nonzero",
        type=float,
        required=True,
        help="Probability of a nonzero count (e.g., nonzeros/total)",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        required=True,
        help="Maximum count value observed (e.g., 874)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="lookup_table.tsv",
        help="Output TSV file for the lookup table",
    )
    args = parser.parse_args()

    # Create an array of count values from 0 to max_count.
    counts = np.arange(0, args.max_count + 1)

    # Compute the ZIP probability for each count:
    # ZIP_prob(k) = p_nonzero * ( exp(-mean) * mean^k / k! )
    zip_probs = args.p_nonzero * poisson.pmf(counts, args.mean)

    # Compute the tail probability for each count (cumulative sum from k=c to k=max_count):
    # We do this by reversing, cumulative summing, and then reversing the result.
    tail_probs = np.flip(np.cumsum(np.flip(zip_probs)))

    # Write the lookup table to a TSV file: one line per count with columns: count, tail_probability.
    with open(args.out_file, "w") as f:
        f.write("count\ttail_prob\n")
        for c, tp in zip(counts, tail_probs):
            f.write(f"{c}\t{tp}\n")

    print(f"Lookup table written to {args.out_file}")


if __name__ == "__main__":
    main()
