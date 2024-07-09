import pathlib
import warnings
from typing import Callable, List, Optional, Union
import pandas as pd
from utipy import Messenger
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess


def read_bed_as_df(
    path: Union[str, pathlib.Path],
    col_names: List[str] = ["chromosome", "start", "end"],
    when_empty: str = "warn_empty",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Read BED file as data frame.

    Lines starting with 't' are considered comments. This should work as all 'chrom' field entries
    should start with either a 'c', an 's' or a digit.
    Based on https://stackoverflow.com/a/58179613

    Raises
    ------
    `RuntimeError`
        When the file is empty and `when_empty='raise'`.

    Parameters
    ----------
    path:
        Path to BED file.
    when_empty: str
        How to react to empty files.
        One of {'raise', 'empty', 'warn_empty'}.
            'empty' and 'warn_empty' returns an empty data frame
            with the columns supplied in `col_names`.

    Returns
    -------
    `pandas.DataFrame`
        The information from the BED file.
    """

    # Get number of columns from first row
    num_cols = get_file_num_columns(path)

    # Handle when file was empty, meaning no information in the file at all
    if num_cols <= 0:
        if when_empty == "raise":
            raise RuntimeError(f"File was empty: {path}")
        if when_empty == "warn_empty":
            messenger(
                f"The following BED/CSV file was empty: {path}. "
                "Returning data frame with expected columns but dtypes may be wrong.",
                add_msg_fn=warnings.warn,
            )
        return pd.DataFrame(columns=col_names)

    # Get columns and column names to use
    use_cols = None
    # Maximally read as many columns as we have names for
    if len(col_names) < num_cols:
        use_cols = range(len(col_names))
    # Greedily supply names for the number of available columns
    elif len(col_names) > num_cols:
        col_names = col_names[:num_cols]

    extra_args = {}

    # Allow reading gzipped files
    if str(path)[-3:] == ".gz":
        extra_args["compression"] = "gzip"

    try:
        df = pd.read_csv(
            path,
            header=None,
            sep="\t",
            comment="t",
            names=col_names,
            low_memory=False,
            usecols=use_cols,
            **extra_args,
        )
    except pd.errors.ParserError as e:
        if "Too many columns specified" in str(e):
            messenger(
                "`Pandas` failed to read `bed_file` with c engine. Trying with python engine.",
                add_msg_fn=warnings.warn,
            )
            df = pd.read_csv(
                path,
                header=None,
                sep="\t",
                comment="t",
                names=col_names,
                usecols=use_cols,
                engine="python",
                **extra_args,
            )
        else:
            raise e

    # Just in case there was some headers (shouldn't be the case)
    # but no rows, we check length of df again
    if len(df) == 0:
        messenger(
            f"The following BED/CSV file was empty: {path}. "
            "Returning data frame with expected columns but dtypes may be wrong.",
            add_msg_fn=warnings.warn,
        )
        df = pd.DataFrame(columns=col_names)

    return df


def get_file_num_lines(in_file):
    """
    Get number of lines in a file using the
    `wc -l <file>` command in a subprocess.
    """
    return int(
        call_subprocess(
            f"wc -l {in_file}", "`wc -l` failed", return_output=True
        ).split()[0]
    )


def get_file_num_columns(in_file) -> int:
    """
    Get number of columns in a BED file using the
    `awk -F'\t' '{print NF; exit}'` command
    in a subprocess. Works better than `.read_line()`
    when one of the columns has missing data (NaN)
    in the first row.
    `in_file` is allowed to be gzipped.

    Note: When the file is empty, 0 is returned!
    """
    # Whether to read from gzipped file or not
    cat_type = "zcat" if str(in_file)[-3:] == ".gz" else "cat"
    call = (
        # If file is not empty
        f"[ -s {in_file} ] && "
        # `(z)cat` the file
        f"({cat_type} {in_file} | "
        # Get the first three rows
        "head -n 3 | awk -F'\t' "
        # Print number of columns
        "'{print NF; exit}') "
        # If file is empty
        # Return -1 so we know the file was empty
        "|| echo 0"
    )
    call_msg = f"{cat_type} <file> | head -n 3 | awk -F'\t' " + "'{print NF; exit}'"
    return int(
        call_subprocess(call, f"`{call_msg}` failed", return_output=True).split()[0]
    )


def split_by_chromosome(in_file, out_dir):
    check_paths_for_subprocess(in_file)
    split_call = " ".join(
        [
            "awk",
            "-F'\t'",
            "-v",
            "OFS='\t'",
            "'{print",
            '>"' + str(out_dir) + '/"$1".bed"}' + "'",
            str(in_file),
        ]
    )
    call_subprocess(split_call, "`awk` failed")
