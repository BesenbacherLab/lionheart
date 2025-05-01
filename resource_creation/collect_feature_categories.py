"""
Annotates idx->cell-type files with feature category information.
"""

import argparse
import logging
import pathlib
import pandas as pd
from utipy import StepTimer, IOPaths, Messenger

# Requires installation of lionheart
from lionheart.utils.dual_log import setup_logging

# Get path to directory that contain this script
# as the sparsify overlaps awk script is in the same directory
script_dir = pathlib.Path(__file__).parent.resolve()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--index_to_cell_type_files",
        required=True,
        nargs="*",
        type=str,
        help="Paths to `.csv` files with array index per cell type. "
        "Must have the columns: {'idx', 'cell_type'}. "
        "Columns should be comma-separated. "
        "Paths should be space-separated.",
    )
    parser.add_argument(
        "--meta_data_files",
        required=True,
        nargs="*",
        type=str,
        help="Paths to `.tsv` file with category meta data for the chromatin tracks. "
        "Must have the columns: {'annotated_biosample_name', 'category', 'seq_type', "
        "'biosample_type', 'cancer_derived', 'embryo_derived'}, and match "
        "the respective `--index_to_cell_type_files` file. "
        "Columns should be tab-separated."
        "Paths should be space-separated.",
    )
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="Path to save new meta data file at.",
    )

    args = parser.parse_args()

    if not len(args.index_to_cell_type_files) == len(args.meta_data_files):
        raise ValueError(
            "Please specify the same number of `--index_to_cell_type_files` "
            "and `--meta_data_files` (with the same order)."
        )

    out_dir = pathlib.Path(args.out_file).parent

    # Prepare logging messenger
    setup_logging(
        dir=str(out_dir / "logs"),
        fname_prefix="annotate_feature_categories-",
    )
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running collection and annotation of cell type meta data")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks

    index_files = {
        f"index_{i}": file for i, file in enumerate(args.index_to_cell_type_files)
    }
    category_files = {
        f"category_{i}": file for i, file in enumerate(args.meta_data_files)
    }

    paths = IOPaths(
        in_files={**index_files, **category_files},
        out_dirs={
            "out_dir": out_dir,
        },
        out_files={"feature_category_file": args.out_file},
    )

    paths.mk_output_dirs("out_dirs")

    all_datasets = []

    max_idx = -1
    messenger("Start: Reading in file pairs")
    for file_idx in range(len(index_files)):
        idx_data = pd.read_csv(paths[f"index_{file_idx}"], sep=",")
        idx_data["idx"] = idx_data["idx"].astype(int)

        category_data = pd.read_csv(paths[f"category_{file_idx}"], sep="\t")
        category_data.rename(
            {"annotated_biosample_name": "cell_type"}, inplace=True, axis=1
        )
        # Remove sample ID and unwanted columns
        category_data = (
            category_data.loc[
                :,
                [
                    "cell_type",
                    "category",
                    "seq_type",
                    "biosample_type",
                    "cancer_derived",
                    "embryo_derived",
                ],
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        messenger(
            f"Indices shape: {idx_data.shape} ; Groupings shape: {category_data.shape}",
            indent=2,
        )

        # Join to get the category info per cell type
        current_data = (
            idx_data.merge(category_data, how="left", on="cell_type")
            .loc[
                :,
                [
                    "idx",
                    "cell_type",
                    "category",
                    "seq_type",
                    "biosample_type",
                    "cancer_derived",
                    "embryo_derived",
                ],
            ]
            .sort_values("idx")  # Should be already but worth the sanity check
        )

        # Update idx
        current_data["idx"] = current_data["idx"] + (max_idx + 1)
        max_idx = int(current_data["idx"].max())

        # Set consensus values
        current_data.loc[
            current_data["cell_type"] == "consensus", ["category", "seq_type"]
        ] = ["Consensus", current_data.loc[0, "seq_type"]]
        mask = current_data["cell_type"] == "consensus"
        current_data.loc[mask] = current_data.loc[mask].fillna(False)

        messenger(
            f"Min idx: {current_data.idx.min()} ; Max idx: {current_data.idx.max()}",
            indent=2,
        )

        all_datasets.append(current_data)

    combined_data = pd.concat(all_datasets, axis=0, ignore_index=True)

    messenger(f"Final shape: {combined_data.shape}")

    messenger(f"Writing to: {paths['feature_category_file']}")
    combined_data.to_csv(paths["feature_category_file"], sep="\t", index=False)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


if __name__ == "__main__":
    main()
