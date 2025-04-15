import json
import pathlib
from typing import Dict

import numpy as np


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def load_chrom_indices(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """
    Load `.npz` file with indices per chromosome.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from `chrom -> indices`.
    """
    chrom_to_indices_arr = np.load(path, allow_pickle=True)
    return {
        chrom: chrom_to_indices_arr[chrom].flatten()
        for chrom in [f"chr{i}" for i in range(1, 23)]
        if chrom in chrom_to_indices_arr.files
    }
