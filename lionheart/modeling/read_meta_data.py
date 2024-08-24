import pathlib
from typing import Callable, Optional, Union, List, Tuple
import pandas as pd
from utipy import Messenger


def read_meta_data(
    path: Union[str, pathlib.PurePath],
    task: str = "binary_classification",
    sep: str = ",",
    targets_as_str: bool = False,
    name: Optional[str] = None,
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
) -> Tuple[List[str], List[Union[str, int]], List[str], dict, dict]:
    """
    Read csv file with meta data and extract the sample IDs, targets and (potentially) groups.

    Note: To match with datasets, the sample ID order should be the same as the
    sample IDs used during creation of the datasets.

    Parameters
    ----------
    path
        For classification tasks:
            Path to `.csv` file where:
                1) the first column contains the sample IDs\n
                2) the second column contains their cancer status (e.g. from {'control', 'cancer', 'exclude'})\n
                3) the third column contains the cancer type (for subtyping)\n
                4) the (optional) fourth column contains the group (e.g. Subject ID when
                subjects have multiple samples).\n
            Note that the cancer status is returned for `task='binary_classification'`
            and cancer type is returned for `task='multiclass_classification'`.
        For regression tasks:
            Path to `.csv` file where:
                1) the first column contains the sample IDs\n
                2) the second column contains their target value\n
                3) the (optional) third column contains the group (e.g. Subject ID when
                subjects have multiple samples).\n
        Other columns are ignored.\n
        The file must contain a header but the actual column names are ignored.
    task
        Whether the meta data is for "binary_classification",
        "multiclass_classification", or "regression".
    targets_as_str : bool
        Whether to convert targets to strings.
    name
        Name of the dataset. Purely for messaging (printing/logging) purposes.
    messenger
        A messenger to print out the header of the meta data (indicates errors
        when the meta data does not have a header (which is required)).\n
        By default, the `verbose` setting is set to `False`
        resulting in no messaging.

    Returns
    -------
    list
        Sample IDs.
    list
        Target values with same order as the sample IDs.
        For binary classification: Cancer statuses
        For multiclass classification: Cancer types
        For regression: The given target values
    list or `None`
        Group IDs (str) with same order as the sample IDs.
        When no third column is present in the file,
        `None` is returned.
    dict
        Dict mapping sample IDs to their target.
    dict
        Dict mapping targets to their sample IDs.
        When `task` is "regression", this is `None`.
    """

    if task not in ["binary_classification", "multiclass_classification", "regression"]:
        raise ValueError(
            "`task` must be one of {'binary_classification', 'multiclass_classification', 'regression'}."
            f"Got: {task}"
        )

    # Read meta data
    if "classification" in task:
        meta = pd.read_csv(path, sep=sep).iloc[:, 0:4]
        loaded_cols = meta.columns
        meta.columns = ["sample", "cancer_status", "cancer_type", "group"][
            : len(meta.columns)
        ]
        target_name = (
            "cancer_status" if task == "binary_classification" else "cancer_type"
        )
    else:
        meta = pd.read_csv(path, sep=sep).iloc[:, 0:3]
        loaded_cols = meta.columns
        meta.columns = ["sample", "target", "group"][: len(meta.columns)]
        target_name = "target"

    name_string = f"({name}) " if name is not None else ""
    messenger(f"{name_string}Meta data: {len(meta)} rows, header: {list(loaded_cols)}")

    # Create maps from sample IDs to targets
    # and (in classification) vice versa
    target_to_sample_ids = None
    if "classification" in task:
        target_to_sample_ids = {
            k: [x for x, _ in list(v.itertuples(index=False, name=None))]
            for k, v in meta.loc[:, ["sample", target_name]].groupby(target_name)
        }
        sample_id_to_target = {
            sid: k for k, v in target_to_sample_ids.items() for sid in v
        }
    elif task == "regression":
        sample_id_to_target = {k: v for k, v in zip(meta["sample"], meta[target_name])}

    # Get sample IDs
    samples = meta["sample"].tolist()
    targets = meta[target_name].tolist()
    groups = meta["group"].tolist() if "group" in meta.columns else None

    # Convert to strings
    samples = [str(s) for s in samples]
    if groups is not None:
        groups = [str(g) for g in groups]
    if targets_as_str:
        targets = [str(t) for t in targets]

    return samples, targets, groups, sample_id_to_target, target_to_sample_ids
