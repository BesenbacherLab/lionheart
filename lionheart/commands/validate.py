"""
Script that validates a model on one or more specified validation datasets.

"""

import logging
import pathlib
import pandas as pd
from utipy import Messenger, StepTimer, IOPaths
from generalize.evaluate.evaluate import Evaluator
from lionheart.modeling.run_predict_single_model import (
    extract_custom_threshold_paths,
    run_predict_single_model,
)
from lionheart.utils.dual_log import setup_logging
from lionheart.utils.cli_utils import Examples, Guide, parse_thresholds
from lionheart.utils.global_vars import INCLUDED_MODELS, LABELS_TO_USE
from lionheart.modeling.prepare_modeling_command import prepare_validation_command
from lionheart.modeling.prepare_modeling import prepare_modeling
from lionheart.utils.utils import load_json

# TODO Not implemented
# - Add the figure of sens/spec thresholds in train and test ROCs


def setup_parser(parser):
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to `feature_dataset.npy` file containing the collected features. "
        "\nExpects shape <i>(?, 10, 898)</i> (i.e., <i># samples, # feature sets, # features</i>). "
        "\nOnly the first feature set is used. "
        "\nNOTE: To validate on the included validation dataset, set --use_included_validation instead. "
        "Only one dataset can be validated on at a time.",
    )
    parser.add_argument(
        "--meta_data_path",
        type=str,
        help="Path(s) to csv file(s) where:"
        "\n  1) the first column contains the <b>sample IDs</b>"
        "\n  2) the second column contains the <b>cancer status</b>\n      One of: {<i>'control', 'cancer', 'exclude'</i>}"
        "\n  3) the third column contains the <b>cancer type</b> "
        + (
            (
                "for subtyping (see --subtype)"
                "\n     Either one of:"
                "\n       {<i>'control', 'colorectal cancer', 'bladder cancer', 'prostate cancer',"
                "\n       'lung cancer', 'breast cancer', 'pancreatic cancer', 'ovarian cancer',"
                "\n       'gastric cancer', 'bile duct cancer', 'hepatocellular carcinoma',"
                "\n       'head and neck squamous cell carcinoma', 'nasopharyngeal carcinoma',"
                "\n       'exclude'</i>} (Must match exactly (case-insensitive) when using included features!) "
                "\n     or a custom cancer type."
                "\n     <b>NOTE</b>: When not running subtyping, any character value is fine."
            )
            if False  # ENABLE_SUBTYPING
            else "[NOTE: Not currently used so can be any string value!]."
        )
        + "\n  4) the (optional) fourth column contains the <b>subject ID</b> "
        "(for when subjects have more than one sample)"
        "\nWhen --dataset_paths has multiple paths, there must be "
        "one meta data path per dataset, in the same order."
        "\nSamples with the <i>'exclude'</i> label are excluded from the training."
        "\nNOTE: To validate on the included validation dataset, set --use_included_validation instead. "
        "Only one dataset can be validated on at a time.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Path to directory to store the validation outputs in."
            "\nA `log` directory will be placed in the same directory."
        ),
    )
    parser.add_argument(
        "--use_included_validation",
        action="store_true",
        help="Whether to use the included validation dataset."
        "\nWhen specified, the --resources_dir must also be specified. "
        "\nWhen NOT specified, only the manually specified datasets are used.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of dataset (when specified). <i>Optional</i> but helps interpretability of outputs."
        "\nUse quotes (e.g., 'name of dataset') in case of whitespace.",
    )
    parser.add_argument(
        "--resources_dir",
        type=str,
        help="Path to directory with framework resources.",
    )
    parser.add_argument(
        "--model_name",
        choices=INCLUDED_MODELS,
        type=str,
        help="Name of the included model to validate."
        "\nNOTE: Only one of `--model_name` and `--custom_model_dir` can be specified.",
    )
    parser.add_argument(
        "--custom_model_dir",
        type=str,
        help="Path to a directory with a custom model to use. "
        "\nThe directory must include the files `model.joblib`, `ROC_curves.json`, and `training_info.json`. "
        "\nThe directory name will be used to identify the model in the output."
        "\nNOTE: Only one of `--model_name` and `--custom_model_dir` can be specified.",
    )
    parser.add_argument(
        "--custom_threshold_dirs",
        type=str,
        nargs="*",
        help="Path(s) to a directory with `ROC_curves.json` and `probability_densities.csv` "
        "files made with `lionheart customize_thresholds` "
        "for extracting the probability thresholds."
        "\nThe output will have predictions for thresholds "
        "based on each of the available ROC curves and probability densities "
        "from the training data and these directories.",
    )
    threshold_defaults = [
        "max_j",
        "spec_0.95",
        "spec_0.99",
        "sens_0.95",
        "sens_0.99",
        "0.5",
    ]
    parser.add_argument(
        "--thresholds",
        type=str,
        nargs="*",
        default=threshold_defaults,
        help="The probability thresholds to use in cancer detection."
        f"\nDefaults to these {len(threshold_defaults)} thresholds:\n  {', '.join(threshold_defaults)}"
        "\n'max_j' is the threshold at the max. of Youden's J (`sensitivity + specificity + 1`)."
        "\nPrefix a specificity-based threshold with <b>'spec_'</b>. \n  The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "\nPrefix a sensitivity-based threshold with <b>'sens_'</b>. \n  The first threshold "
        "that should lead to a specificity above this level is chosen. "
        "\nWhen passing specific float thresholds, the nearest threshold "
        "in the ROC curve is used. "
        "\n<b>NOTE</b>: The thresholds are extracted from the included ROC curve,"
        "\nwhich was fitted to the <b>training</b> data during model training.",
    )
    parser.add_argument(
        "--aggregate_by_subjects",
        action="store_true",
        help="Whether to aggregate <i>predictions</i> per subject before evaluations. "
        "\nThe predicted probabilities are averaged per subject."
        "\n<u><b>Ignored</b></u> when no subject IDs are present in the meta data.",
    )
    # Declare defaults for cv-only args to allow sharing preparation function
    parser.set_defaults(
        feature_type="LIONHEART",
        feature_categories=[],
        loco=False,
        loco_train_only_classes=False,
    )
    parser.set_defaults(func=main)


epilog_guide = Guide()
epilog_guide.add_title("OUTPUT:")
epilog_guide.add_description(
    """evaluation_scores.csv : data frame
    This data frame contains the evaluation scores from the validation:

    Columns:
        Model: Name of the applied model used for predictions.
        Task: The task performed by the model.
        ROC Curve: Name of the Receiver Operating Characteristic curve used to calculate the probability threshold.
        Threshold Name: The name of the threshold (i.e. probability cutoff) used for decision making.
        Threshold: The actual probability cutoff used to determine the predicted class.
        ...
        Positive Class: The name of the positive class used to calculate the metrics.
        Num Classes: The number of classes.
    
predictions.csv : data frame
    This data frame contains the predicted probabilities and their respective threshold-wise predicted cancer statuses.
    
    Columns:
        Model, Task, Threshold Name, ROC Curve, Threshold: Same as above.
        Prediction: The predicted cancer status.
        P(Cancer): The probability of the sample being from a cancer patient.
        Sample ID: The unique sample identifier.
        Target: The actual cancer status of the sample.
        Subject ID: The unique subject identifier (when specified in the meta data).
        Exp. Specificity: The expected specificity at the probability threshold.
        Exp. Sensitivity: The expected sensitivity at the probability threshold.
        Exp. Accuracy for Class at Probability: The expected accuracy of predicting the specific class at the specific probability.
            I.e., for all samples with this specific probability (interpolated), what percentage were from the predicted class?
            Given a new prediction of the class with this probability, we would expect it to be correct that percentage of the time.
            Calculated based on probability density estimates from the same data as the ROC curve was calculated from.
            Informs about the reliability of the class prediction (in addition to the probability).
    
"""
)
epilog_guide.add_vertical_space(1)

examples = Examples()
examples.add_example(
    description="Validate your model on included validation dataset:",
    example="""--out_dir path/to/model_validation
--resources_dir path/to/resource/directory
--custom_model_dir path/to/new_model
--use_included_validation
""",
)
examples.add_example(
    description="Validate included model on your dataset:",
    example=f"""--dataset_path path/to/dataset/feature_dataset.npy 
--meta_data_path path/to/dataset/meta_data.csv
--dataset_name 'the_dataset'
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory
--model_name {INCLUDED_MODELS[0]}
""",
)

EPILOG = epilog_guide.construct_guide() + examples.construct()


def main(args):
    out_path = pathlib.Path(args.out_dir)

    if sum([args.model_name is not None, args.custom_model_dir is not None]) != 1:
        raise ValueError(
            "Exactly one of {`--model_name`, `--custom_model_dir`} "
            "should be specified at a time."
        )
    if args.model_name is not None:
        if args.resources_dir is None:
            raise ValueError(
                "When `--model_name` is specified, "
                "`--resources_dir` must also be specified."
            )

        resources_dir = pathlib.Path(args.resources_dir)
        model_dir = resources_dir / "models" / args.model_name
        model_name = args.model_name

    else:
        if args.resources_dir is not None:
            resources_dir = pathlib.Path(args.resources_dir)
        model_dir = pathlib.Path(args.custom_model_dir)
        model_name = model_dir.stem

    if sum([args.dataset_path is not None, args.use_included_validation]) != 1:
        raise ValueError(
            "Exactly one of {`--dataset_path`, `--use_included_validation`} "
            "should be specified at a time."
        )

    # Prepare logging messenger
    setup_logging(dir=str(out_path / "logs"), fname_prefix="validate-")
    messenger = Messenger(verbose=True, indent=0, msg_fn=logging.info)
    messenger("Running validation")
    messenger.now()

    # Init timestamp handler
    # Note: Does not handle nested timing!
    timer = StepTimer(msg_fn=messenger)

    # Start timer for total runtime
    timer.stamp()

    paths = IOPaths(
        out_dirs={"out_path": out_path},
    )
    if resources_dir is not None:
        paths.set_path("resources_dir", resources_dir, "in_dirs")

    # Add dataset paths to lists for code reuse
    args.dataset_paths = []
    args.meta_data_paths = []
    args.dataset_names = None
    if args.dataset_path is not None:
        args.dataset_paths += [args.dataset_path]
    if args.meta_data_path is not None:
        args.meta_data_paths += [args.meta_data_path]
    if args.dataset_name is not None:
        args.dataset_names = [args.dataset_name]

    dataset_paths, meta_data_paths = prepare_validation_command(
        args=args,
        paths=paths,
        messenger=messenger,
    )

    custom_threshold_dirs, custom_roc_paths, custom_prob_density_paths = (
        extract_custom_threshold_paths(args)
    )

    # Create output directory
    paths.mk_output_dirs(collection="out_dirs", messenger=messenger)

    prepared_modeling_dict = prepare_modeling(
        dataset_paths=dataset_paths,
        out_path=out_path,
        meta_data_paths=meta_data_paths,
        feature_name_to_feature_group_path=None,
        task="binary_classification",
        labels_to_use=LABELS_TO_USE,
        feature_sets=[0],
        aggregate_by_groups=args.aggregate_by_subjects,
        expected_shape={1: 10, 2: 898},  # 10 feature sets, 898 cell types
        mk_plots_dir=False,
        timer=timer,
        messenger=messenger,
    )

    # Paths is a new object from prepare_modeling
    paths = prepared_modeling_dict["paths"]
    paths.set_path("prediction_path", out_path / "predictions.csv", "out_files")
    paths.set_path("evaluation_path", out_path / "evaluation_scores.csv", "out_files")
    paths.set_path("resources_dir", resources_dir, "in_dirs")

    # NOTE: These names must match what's used in
    # predict_sample (since they both use run_predict_single_model())
    paths.set_paths(
        {
            f"model_{model_name}": model_dir / "model.joblib",
            f"roc_curve_{model_name}": model_dir / "ROC_curves.json",
            f"prob_densities_{model_name}": model_dir / "probability_densities.csv",
            f"training_info_{model_name}": model_dir / "training_info.json",
            **custom_roc_paths,
            **custom_prob_density_paths,
        },
        "in_files",
    )
    paths.set_paths(custom_threshold_dirs, "in_dirs")

    # Show overview of the paths
    messenger(paths)

    messenger("Start: Interpreting `--thresholds`")
    thresholds_to_calculate = parse_thresholds(args.thresholds)

    messenger("Start: Loading training information")
    model_name_to_training_info = {
        model_name: load_json(paths[f"training_info_{model_name}"])
    }

    # Construct data frame with sample identifiers for the predictions data frame
    sample_identifiers = pd.DataFrame(
        {
            "Sample ID": prepared_modeling_dict["sample_ids"].flatten(),
            "Target": prepared_modeling_dict["labels"].flatten(),
        }
    )

    # Ensure we have a row of sample identifiers per dataset
    assert len(sample_identifiers) == len(prepared_modeling_dict["dataset"])

    # Add optional columns
    if prepared_modeling_dict["groups"] is not None:
        sample_identifiers["Subject ID"] = prepared_modeling_dict["groups"]
    if prepared_modeling_dict["split"] is not None:
        sample_identifiers["Dataset"] = prepared_modeling_dict["split"]

    prediction_dfs = run_predict_single_model(
        features=prepared_modeling_dict["dataset"],
        sample_identifiers=sample_identifiers,
        model_name=model_name,
        model_name_to_training_info=model_name_to_training_info,
        custom_roc_paths=custom_roc_paths,
        custom_prob_density_paths=custom_prob_density_paths,
        thresholds_to_calculate=thresholds_to_calculate,
        paths=paths,
        messenger=messenger,
        timer=timer,
        model_idx=0,
    )

    # Combine data frames and clean it up a bit
    all_predictions_df = pd.concat(prediction_dfs, axis=0, ignore_index=True)

    # Reorder columns
    prob_columns = [col_ for col_ in all_predictions_df.columns if col_[:2] == "P("]
    first_columns = (
        [
            "Model",
            "Task",
            "Threshold Name",
            "ROC Curve",
            "Prediction",
        ]
        + prob_columns
        + list(sample_identifiers.columns)
    )
    remaining_columns = [
        col_ for col_ in all_predictions_df.columns if col_ not in first_columns
    ]
    all_predictions_df = all_predictions_df.loc[:, first_columns + remaining_columns]

    messenger("Start: Saving predicted probability to disk")
    all_predictions_df.to_csv(paths["prediction_path"], index=False)

    # NOTE: The dataset is ordered by target class
    messenger("First few predictions:\n", all_predictions_df.head(3), indent=2)

    messenger("Start: Evaluating predictions")

    # Load and prepare `New Label Index to New Label` mapping
    label_idx_to_label = model_name_to_training_info[model_name]["Labels"][
        "New Label Index to New Label"
    ]
    # Ensure keys are integers
    label_idx_to_label = {int(key): val for key, val in label_idx_to_label.items()}

    evals = []

    if len(prob_columns) != 1:
        # TODO The evaluation will currently fail so work around it later
        raise NotImplementedError(
            "Multiple probability columns are not currently supported."
        )

    # Model loop is just future proofing
    for model_nm in all_predictions_df["Model"].unique():
        for roc_name in all_predictions_df["ROC Curve"].unique():
            for thresh_name in all_predictions_df["Threshold Name"].unique():
                thresh_rows = all_predictions_df.loc[
                    (all_predictions_df["Model"] == model_nm)
                    & (all_predictions_df["Threshold Name"] == thresh_name)
                    & (all_predictions_df["ROC Curve"] == roc_name)
                ]

                eval_ = Evaluator.evaluate(
                    targets=thresh_rows["Target"].to_numpy(),
                    predictions=thresh_rows[prob_columns[0]].to_numpy(),
                    groups=thresh_rows["Subject ID"].to_numpy()
                    if args.aggregate_by_subjects
                    and prepared_modeling_dict["groups"] is not None
                    else None,
                    positive=1,
                    thresh=thresh_rows["Threshold"].to_numpy()[0],
                    labels=label_idx_to_label,
                    task="binary_classification",
                )["Scores"]

                eval_["Model"] = model_nm
                eval_["Threshold Name"] = thresh_name
                eval_["ROC Curve"] = roc_name
                eval_["Task"] = thresh_rows["Task"].to_numpy()[0]

                evals.append(eval_)

    all_evaluations = pd.concat(evals).reset_index(drop=True)

    first_columns = [
        "Model",
        "Task",
        "ROC Curve",
        "Threshold Name",
        "Threshold",
    ]
    remaining_columns = [
        col_ for col_ in all_evaluations.columns if col_ not in first_columns
    ]
    all_evaluations = all_evaluations.loc[:, first_columns + remaining_columns]

    messenger("Start: Saving evaluations to disk")
    all_evaluations.to_csv(paths["evaluation_path"], index=False)

    messenger("Evaluations:\n", all_evaluations, indent=2)

    timer.stamp()
    messenger(f"Finished. Took: {timer.get_total_time()}")
