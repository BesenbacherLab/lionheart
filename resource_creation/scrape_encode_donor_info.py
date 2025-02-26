import os
import json
import requests
from tqdm import tqdm
import pandas as pd
import argparse
import logging
import pathlib
from collections.abc import MutableMapping
from utipy import StepTimer, IOPaths, Messenger

# Requires installation of lionheart
from lionheart.utils.dual_log import setup_logging


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(""" """)
    parser.add_argument(
        "--meta_data_file",
        type=str,
        required=True,
        help="Path the meta data file with human donor info.",
    )
    parser.add_argument(
        "--out_file",
        required=True,
        type=str,
        help="Path to output `.tsv` file with scraped information.",
    )
    args = parser.parse_args()

    # Prepare logging messenger
    setup_logging(
        dir=str(pathlib.Path(args.out_file).parent / "logs"),
        fname_prefix="scrape_encode-",
    )
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running scraping of ENCODE human donors information")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Create paths container with checks
    paths = IOPaths(
        in_files={
            "meta_data_file": args.meta_data_file,
        },
        out_files={
            "output_file": args.out_file,
        },
        out_dirs={"out_dir": pathlib.Path(args.out_file).parent},
    )

    # Show overview of the paths
    messenger(paths)

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs")

    # Load meta data
    messenger("Start: Loading meta data for extraction of donor IDs")
    meta = pd.read_csv(paths["meta_data_file"], sep="\t")
    donor_ids = set(meta["Donor(s)"].apply(lambda s: s.split("/")[-2]))
    messenger(f"Got {len(donor_ids)} unique donor IDs")

    messenger("Start: Downloading meta data from ENCODE")
    with timer.time_step(indent=2):
        donor_metadata = download_encode_donor_metadata(donor_ids)

    messenger("Start: Flattening json to data frame")
    with timer.time_step(indent=2):
        donor_metadata_dicts = [
            call_flatten_dict(donor_meta, donor_id)
            for donor_id, donor_meta in donor_metadata.items()
        ]
        donor_metadata_df = pd.DataFrame(donor_metadata_dicts)

    messenger(donor_metadata_df.head(10))

    messenger("Start: Saving data frame to disk")
    donor_metadata_df.to_csv(paths["output_file"], sep="\t")

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")


def download_encode_donor_metadata(donor_ids, save_dir="encode_donors"):
    """
    Downloads ENCODE human donor metadata in JSON format and stores it in a dictionary.

    Parameters
    ----------
    donor_ids: list of str
        List of ENCODE donor IDs (e.g., ['ENCDO449WOZ', 'ENCDO123ABC']).
    save_dir: str
        Directory to save JSON files.

    Returns
    -------
    donor_data: dict
        Dictionary with donor IDs as keys and JSON metadata as values.
    """
    base_url = "https://www.encodeproject.org/human-donors/"
    os.makedirs(save_dir, exist_ok=True)
    donor_data = {}

    for donor_id in tqdm(donor_ids, desc="Downloading donor metadata"):
        url = f"{base_url}{donor_id}/?format=json"
        save_path = os.path.join(save_dir, f"{donor_id}.json")

        # Download only if not already saved
        if not os.path.exists(save_path):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                donor_json = response.json()

                # Save JSON file
                with open(save_path, "w") as f:
                    json.dump(donor_json, f, indent=4)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {donor_id}: {e}")
                continue
        else:
            # Load from existing file
            with open(save_path, "r") as f:
                donor_json = json.load(f)

        # Store in dictionary
        donor_data[donor_id] = donor_json

    return donor_data


def flatten_dict(d, parent_key="", sep="__"):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to a comma-separated string
            items.append((new_key, ", ".join(map(str, v)) if v else None))
        else:
            items.append((new_key, v))
    return dict(items)


def call_flatten_dict(d, id_):
    d_flat = flatten_dict(d)
    d_flat["donor_id"] = id_
    return d_flat


if __name__ == "__main__":
    main()
