import pathlib
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import scipy.sparse


def convert_nonzero_bins_to_sparse_array(
    num_bins: int,
    scaling_constant: Optional[int],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    array_type: str = "csc",
) -> Tuple[int, int]:
    """
    Reads chrom.sparsed.txt from input_dir, converts it into a CSC column vector,
    and saves chrom.npz in output_dir.

    Parameters
    ----------
    num_bins
        Number of bins in chromosome. Used for shape of sparse array.
    scaling_constant
        Value to divide each non-zero value by. E.g., the bin size.
    input_path
        Path to file with index and overlap/coverage values.
    output_path
        Path to write the .npz file at.
    array_type
        Whether to create a CSC ("csc") or CSR ("csr") array.

    Returns
    -------
    int
        Number of non-zero rows.
    int
        Sum over overlap/coverage values before scaling.
    """
    assert num_bins > 0
    assert array_type in ["csc", "csr"]

    # Load two columns: [index, overlap]
    data = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=["index", "overlap"],
        dtype={0: "int64", 1: "float32"},
    )
    assert len(data) > 1

    # Extract columns to separate numpy arrays
    rows = data.loc[:, "index"].to_numpy()
    overlaps = data.loc[:, "overlap"].to_numpy()

    # Total overlaps
    overlap_sum = overlaps.sum(dtype=np.float64)

    # Make into overlap percentage
    if scaling_constant is not None:
        overlaps /= scaling_constant

    # Build a single column vector, so all entries are in col 0
    cols = np.zeros_like(rows, dtype=int)

    # The shape is (num_bins, 1)
    shape = (num_bins, 1)

    # 1) Construct in COO format
    coo = scipy.sparse.coo_matrix((overlaps, (rows, cols)), shape=shape)

    # 2) Convert to CSC or CSR
    cs_mat = coo.tocsc() if array_type == "csc" else coo.tocsr()

    # Save to disk in NPZ format
    scipy.sparse.save_npz(output_path, cs_mat)

    # Return counts of non-zero bins and overlapping positions
    return len(rows), overlap_sum
