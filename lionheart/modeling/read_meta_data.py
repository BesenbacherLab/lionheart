import pathlib
from typing import Callable, Optional, Union, List, Tuple
import pandas as pd
from utipy import Messenger


def read_meta_data(
    path: Union[str, pathlib.PurePath],
    task: str = "classification",
    sep: str = ",",
    targets_as_str: bool = False,
    name: Optional[str] = None,
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
) -> Tuple[List[str], List[Union[str, int]], List[str], dict, dict]:
    """
    Read csv file with meta data and extract the sample IDs and their
    targets/targets and (potentially) groups.

    Note: To match with datasets, the sample ID order should be the same as the
    sample IDs used during creation of the datasets.

    Parameters
    ----------
    path
        Path to `.csv` file where 1) the first column contains the sample IDs
        and 2) the second column contains their target, and 3) the (optional) third
        column contains the group (e.g. Subject ID when subjects have
        multiple samples). Other columns are ignored.
        The file must contain a header but the actual column names are ignored.
    task
        Whether the meta data is for "classification" or "regression".
    targets_as_str : bool
        Whether to convert targets to strings.
    name
        Name of the dataset. Purely for messaging (printing/logging) purposes.
    messenger
        A messenger to print out the header of the meta data.
        This could indicate errors when the meta data did not
        have a header (which is required).
        By default, the `verbose` setting is set to `False`
        resulting in no messaging.

    Returns
    -------
    list
        Sample IDs.
    list
        Targets with same order as the sample IDs.
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

    # Read meta data
    meta = pd.read_csv(path, sep=sep).iloc[:, 0:3]
    name_string = f"({name}) " if name is not None else ""
    messenger(f"{name_string}Meta data: {len(meta)} rows, header: {list(meta.columns)}")
    meta.columns = ["sample", "target", "group"][: len(meta.columns)]

    # Create maps from sample IDs to targets
    # and (in classification) vice versa
    target_to_sample_ids = None
    if task == "classification":
        target_to_sample_ids = {
            k: [x for x, _ in list(v.itertuples(index=False, name=None))]
            for k, v in meta.loc[:, ["sample", "target"]].groupby("target")
        }
        sample_id_to_target = {
            sid: k for k, v in target_to_sample_ids.items() for sid in v
        }
    elif task == "regression":
        sample_id_to_target = {k: v for k, v in zip(meta["sample"], meta["target"])}

    # Get sample IDs
    samples = meta["sample"].tolist()
    targets = meta["target"].tolist()
    groups = meta["group"].tolist() if "group" in meta.columns else None

    # Convert to strings
    samples = [str(s) for s in samples]
    if groups is not None:
        groups = [str(g) for g in groups]
    if targets_as_str:
        targets = [str(t) for t in targets]

    return samples, targets, groups, sample_id_to_target, target_to_sample_ids
