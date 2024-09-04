"""
Script that loads features from multiple samples and
combines them to a numpy array.

"""

from collections import OrderedDict
import logging
import pathlib
import json
import shutil
import numpy as np
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths

from lionheart.utils.dual_log import setup_logging
from lionheart.features.create_dataset_inference import DatasetOutputPaths
from lionheart.utils.cli_utils import Examples


def setup_parser(parser):
    parser.add_argument(
        "--feature_dirs",
        type=str,
        nargs="*",
        help=(
            "Paths to directories with extracted features. "
            "This is the `out_dir` created by `extract_features`. "
            "Features are collected in the order of these paths."
        ),
    )
    parser.add_argument(
        "--prediction_dirs",
        type=str,
        nargs="*",
        help=(
            "Paths to directories with predictions. "
            "This is the `out_dir` created by `predict_sample`. "
            "Predictions are collected in the order of these paths."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the collected features at. "
            "A `log` directory will be placed in the same directory."
        ),
    )
    parser.set_defaults(func=main)


# TODO: Add more examples
examples = Examples()
examples.add_example(
    description="Simplest example:",
    example="""--feature_dirs path/to/subject_1/features path/to/subject_2/features
--prediction_dirs path/to/subject_1/predictions path/to/subject_2/predictions
--out_dir path/to/output/directory
""",
)
EPILOG = examples.construct()


def collect_features(args, out_path, messenger, timer):
    messenger("Start: Preparing collection of features and correction factors")
    feature_dirs = [pathlib.Path(path).resolve() for path in args.feature_dirs]
    if len(set(feature_dirs)) < len(feature_dirs):
        raise ValueError("Found duplicate elements in `--feature_dirs`.")

    # NOTE: Values are DatasetOutputPaths objects!
    sample_path_collections = OrderedDict(
        [
            (
                f"{i}__{feature_dir}",
                DatasetOutputPaths.create_default(
                    dataset_dir=feature_dir, mask_type=None
                ),
            )
            for i, feature_dir in enumerate(feature_dirs)
        ]
    )

    output_file_paths = DatasetOutputPaths.create_default(
        dataset_dir=out_path.resolve(), mask_type=None
    ).get_path_dict(key_prefix="out__")

    feature_in_files = {
        # Extract all paths from sample_path_collections.values()
        k: v
        for path_dict in [
            sample_path_collection.get_path_dict(key_prefix=f"{i}_")
            for i, sample_path_collection in enumerate(sample_path_collections.values())
        ]
        for (k, v) in path_dict.items()
    }
    feature_in_dirs = {
        f"feature_dir_{i}": feature_dir for i, feature_dir in enumerate(feature_dirs)
    }

    paths = IOPaths(
        in_files=feature_in_files,
        in_dirs=feature_in_dirs,
        out_dirs={
            "out_path": out_path,
        },
        out_files={
            **output_file_paths,
            "feature_dirs_out": out_path / "feature_directories.index.csv",
        },
    )

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Collecting features and correction factors across samples")
    with timer.time_step(indent=4, name_prefix="stack_features"):
        first_sample_path_key = list(sample_path_collections.keys())[0]
        for out_file_type in output_file_paths.keys():
            with timer.time_step(indent=8, name_prefix=f"{out_file_type}"):
                file_type = out_file_type[5:]  # remove prefix "out__"
                messenger(f"Collecting {file_type}:", indent=8)
                file_path_extension = (
                    sample_path_collections[first_sample_path_key]
                    .get_path_dict()[file_type]
                    .suffix
                )

                if file_path_extension == ".txt":
                    continue

                elif file_path_extension == ".npy":
                    collected_dataset = np.stack(
                        [
                            np.load(
                                paths[sample_path_key.split("__")[0] + "_" + file_type],
                                allow_pickle=True,
                            ).astype(np.float32)
                            for sample_path_key in sample_path_collections.keys()
                        ],
                        axis=0,
                    )
                    np.save(paths[out_file_type], collected_dataset)

                elif file_path_extension == ".csv":
                    collected_dataset = pd.concat(
                        [
                            pd.read_csv(
                                paths[sample_path_key.split("__")[0] + "_" + file_type]
                            ).assign(
                                original_path=paths[
                                    sample_path_key.split("__")[0] + "_" + file_type
                                ]
                            )
                            for sample_path_key in sample_path_collections.keys()
                        ],
                        ignore_index=True,
                    )
                    collected_dataset.to_csv(paths[out_file_type], index=False)

                elif file_path_extension == ".json":
                    # Read json files
                    json_data = {}
                    for sample_path_key in sample_path_collections.keys():
                        file_path = paths[
                            sample_path_key.split("__")[0] + "_" + file_type
                        ]
                        with file_path.open("r") as f:
                            json_data[str(file_path)] = json.load(f)

                    # Save the combined data as a new JSON file
                    with open(paths[out_file_type], "w") as f:
                        json.dump(json_data, f, indent=4)

        # Save index of the feature directories
        # to ensure we know the order of features
        pd.DataFrame(
            {
                "Original Feature Directory": [
                    paths[key] for key in feature_in_dirs.keys()
                ]
            }
        ).to_csv(paths["feature_dirs_out"], index=True)


def collect_predictions(args, out_path, messenger, timer):
    messenger("Start: Preparing collection of predictions")
    prediction_dirs = [pathlib.Path(path).resolve() for path in args.prediction_dirs]
    if len(set(prediction_dirs)) < len(prediction_dirs):
        raise ValueError("Found duplicate elements in `--prediction_dirs`.")

    sample_paths = OrderedDict(
        [
            (f"{i}__{prediction_dir}", prediction_dir / "prediction.csv")
            for i, prediction_dir in enumerate(prediction_dirs)
        ]
    )
    prediction_in_dirs = {
        f"prediction_dir_{i}": prediction_dir
        for i, prediction_dir in enumerate(prediction_dirs)
    }
    readme_path = prediction_dirs[0] / "README.txt"

    paths = IOPaths(
        in_files={**dict(sample_paths), "readme_path": readme_path},
        in_dirs=prediction_in_dirs,
        out_dirs={
            "out_path": out_path,
        },
        out_files={
            "predictions_out": out_path / "predictions.csv",
            "readme_out": out_path / "predictions.README.txt",
        },
    )

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Collecting predictions across samples")
    with timer.time_step(indent=4, name_prefix="stack_predictions"):
        collected_dataset = pd.concat(
            [
                pd.read_csv(paths[sample_path_key]).assign(
                    original_path=paths[sample_path_key]
                )
                for sample_path_key in sample_paths.keys()
            ],
            ignore_index=True,
        )
        collected_dataset.to_csv(paths["predictions_out"], index=False)

    messenger("Start: Copying README.txt file for predictions to output directory")
    # Copy the prediction readme to the new directory
    shutil.copy(str(paths["readme_path"]), str(paths["readme_out"]))


def main(args):
    if not args.feature_dirs and not args.prediction_dirs:
        raise ValueError(
            "At least one of `--feature_dirs` or `--prediction_dirs` must be specified."
        )
    collecting_string = " and ".join(
        [
            part
            for part in [
                "extracted features" if args.feature_dirs else "",
                "predictions" if args.prediction_dirs else "",
            ]
            if part
        ]
    )

    out_path = pathlib.Path(args.out_dir)

    # Create output directory
    paths = IOPaths(
        out_dirs={
            "out_path": out_path,
        }
    )
    paths.mk_output_dirs(collection="out_dirs")

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="collect_samples-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger(f"Running collection of {collecting_string} for specified samples")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    # Process feature directory paths and check for duplicates
    if args.feature_dirs:
        collect_features(args, out_path, messenger, timer)
    if args.prediction_dirs:
        collect_predictions(args, out_path, messenger, timer)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
