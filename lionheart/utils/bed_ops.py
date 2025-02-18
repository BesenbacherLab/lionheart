import os
import pathlib
import warnings
from typing import Callable, List, Optional, Union
import numpy as np
import pandas as pd
from utipy import Messenger, random_alphanumeric
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


def write_bed(
    bed_df: pd.DataFrame,
    out_file: Union[str, pathlib.Path],
    enforce_dtypes: bool = True,
    dtypes: Optional[dict] = None,
) -> None:
    """
    Write BED file to disk with consistent arguments.
    Sets:
        sep = "\t"
        header = False
        index = False

    (Optionally) sets column types prior to writing.


    Parameters
    ----------
    bed_df
        `pandas.DataFrame` with BED intervals.
    out_file
        Path to the output file.
    enforce_dtypes
        Whether to set column dtypes.
        Without setting `dtypes`, this only works when `bed_df`
        has 3 or 4 columns.
    dtypes
        `dict` with data types per column. The default
        dtypes dict is updated with this dict.
    """
    if enforce_dtypes:
        bed_df = ensure_col_types(bed_df, dtypes=dtypes)
    bed_df.to_csv(out_file, sep="\t", header=False, index=False)


def ensure_col_types(
    bed_df: pd.DataFrame,
    dtypes: Optional[dict] = None,
):
    """
    Convert BED columns to expected dtypes.

    chromosome: 'str',
    start: 'int32',
    end: 'int32'

    Parameters
    ----------
    bed_df
        `pandas.DataFrame` to convert types of columns of.
    dtypes
        `dict` with data types per column. The default
        dtypes dict is updated with this dict.

    Returns
    -------
    `bed_df` with potentially different column types.
    """
    default_dtypes = {"chromosome": "str", "start": "int32", "end": "int32"}
    if dtypes is not None:
        default_dtypes.update(dtypes)
    keys_to_remove = [key for key in default_dtypes.keys() if key not in bed_df.columns]
    for key in keys_to_remove:
        del default_dtypes[key]
    return bed_df.astype(default_dtypes)


def get_file_num_lines(in_file: Union[str, pathlib.Path]):
    """
    Get number of lines in a file using the
    `wc -l <file>` command in a subprocess.
    """
    return int(
        call_subprocess(
            f"wc -l {in_file}", "`wc -l` failed", return_output=True
        ).split()[0]
    )


def get_file_num_columns(in_file: Union[str, pathlib.Path]) -> int:
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


def split_by_chromosome(
    in_file: Union[str, pathlib.Path], out_dir: Union[str, pathlib.Path]
) -> None:
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


def merge_multifile_intervals(
    in_files: List[Union[str, pathlib.Path]],
    out_file: Union[str, pathlib.Path],
    count_coverage: bool = False,
    keep_zero_coverage: bool = False,
    genome_file: Optional[Union[str, pathlib.Path]] = None,
    min_coverage: float = 0.0,
    max_distance: int = 0,
    sort_numerically: bool = False,
    add_index: bool = False,
):
    """
    Merge the intervals of a multiple BED files.
    Files are combined to a single file and sorted.
    Overlapping intervals are merged with `bedtools::merge`.

    Note: When given a single file, the overlapping intervals are still merged
    but there is no sorting step.

    Parameters
    ----------
    in_files
        Paths to the BED files to merge.
        Must be tab-separated.
    out_file
        Path to the output file. Cannot be the same as any in-files.
    count_coverage
        Whether to count coverage of sub-intervals.
        This will likely create more, shorter intervals with an additional count column.
    keep_zero_coverages
        Whether to keep regions with zero coverage.
    genome_file
        Optional path to the genome file with chromosome sizes.
        Required to find limits of the genome when counting coverage.
    min_coverage
        The number/percentage ( [0.0-1.0[ ) of `in_files` that must
        overlap a position for it to be kept.
        Percentages are multiplied by the number of `in_files` and floored (rounded down).
        Ignored when `count_coverage` is `False`.
    max_distance
        Maximum distance between intervals allowed for intervals
        to be merged. Default is 0. That is, overlapping and/or book-ended
        intervals are merged.
    sort_numerically
        Whether to sort by chromosome number (or alphabetically).
        `True`: chr1, chr2, chr3, etc.
        `False`: chr1, chr10, chr11, etc.
    add_index
        Whether to add an interval index column.
    """
    # `genome_file` is checked in merging function
    check_paths_for_subprocess(in_files, out_file)

    if not count_coverage and min_coverage != 0.0:
        warnings.warn("`min_coverage` is ignored when `count_coverage` is disabled.")
    if count_coverage and min_coverage < 1.0:
        min_coverage = np.floor(len(in_files) * min_coverage)
    min_coverage = int(min_coverage)

    # If only given a single file, just
    # merge the overlapping intervals
    if len(in_files) == 1:
        # Merge the overlapping intervals
        if count_coverage:
            merge_overlapping_intervals_with_coverage(
                in_file=in_files[0],
                out_file=out_file,
                keep_zero_coverages=keep_zero_coverage,
                genome_file=genome_file,
                min_coverage=min_coverage,
                max_distance=max_distance,
                add_index=add_index,
            )
        else:
            merge_overlapping_intervals(
                in_file=in_files[0],
                out_file=out_file,
                max_distance=max_distance,
                add_index=add_index,
            )
        return

    # Combine files into one and sort intervals

    # Create path for temporary combined file
    # TODO This tmp handling is not a pretty solution
    tmp_combine_file = (
        str(out_file)[:-4] + f".combined.tmp_{random_alphanumeric(15)}.bed"
    )
    combine_files(
        in_files,
        tmp_combine_file,
        keep_cols=range(3),
        sort_numerically=sort_numerically,
    )

    # Merge the overlapping intervals
    if count_coverage:
        merge_overlapping_intervals_with_coverage(
            in_file=tmp_combine_file,
            out_file=out_file,
            keep_zero_coverages=keep_zero_coverage,
            genome_file=genome_file,
            min_coverage=min_coverage,
            max_distance=max_distance,
            add_index=add_index,
        )
    else:
        merge_overlapping_intervals(
            in_file=tmp_combine_file,
            out_file=out_file,
            max_distance=max_distance,
            add_index=add_index,
        )

    # Remove the tmp combine file
    os.remove(str(tmp_combine_file))


# TODO Allow setting options for ensure_col_types
# TODO Allow not removing non-standard chromosomes
def combine_files(
    in_files: List[Union[str, pathlib.Path]],
    out_file: Union[str, pathlib.Path],
    keep_cols: Optional[List[int]] = None,
    sort_numerically: bool = False,
) -> None:
    """
    Concatenates BED files and sorts the intervals.

    Parameters
    ----------
    in_files
        Paths to the BED files to concatenate and sort.
        Must be tab-separated.
    out_file
        Path to the output file.
        Cannot be the same as any in-files.
    keep_cols
        Indices of the columns to keep.
        Set to `None` for no column filtering.
    sort_numerically
        Whether to sort by chromosome number (or alphabetically).
        `True`: chr1, chr2, chr3, etc.
        `False`: chr1, chr10, chr11, etc.
    """
    check_paths_for_subprocess(in_files, out_file)
    dfs = []
    for in_file in in_files:
        dfs.append(read_bed_as_df(in_file))
    df = pd.concat(dfs, axis=0)
    if keep_cols is not None:
        df = df.iloc[:, keep_cols]
    df = remove_non_standard_chromosomes(df)
    df = ensure_col_types(df)
    df = sort_intervals(df, sort_numerically=sort_numerically)
    write_bed(df, out_file)


def merge_overlapping_intervals_with_coverage(
    in_file: Union[str, pathlib.Path],
    out_file: Union[str, pathlib.Path],
    genome_file: Union[str, pathlib.Path],
    keep_zero_coverages: bool = False,
    min_coverage: int = 0,
    max_distance: int = 0,
    add_index=False,
) -> None:
    """
    Merge the overlapping intervals of a single file with `bedtools::genomecov`.
    Get coverage counts of each subinterval.

    Parameters
    ----------
    in_file
        Path to the BED file to merge overlapping intervals
        of with counts of coverage.
        Must be tab-separated.
    out_file
        Path to the output file. Cannot be the same as `in_file`.
    genome_file
        Path to the genome file with chromosome sizes.
        Used to find limits of the chromosomes.
    keep_zero_coverages
        Whether to keep regions with zero coverage.
    min_coverage
        The coverage count a position must have for it to be kept.
    max_distance
        Maximum distance between intervals allowed for intervals
        to be merged. Default is 0. That is, overlapping and/or book-ended
        intervals are merged.
    add_index
        Whether to add an interval index column.
    """
    check_paths_for_subprocess([in_file, genome_file], out_file)

    # Coverage filtering
    coverage_filter_str = ""
    if min_coverage > (1 - int(keep_zero_coverages)):
        coverage_filter_str = " ".join(
            ["|", "awk", "-F '\t'", "-v", "OFS='\t'", f"'$4>={min_coverage}'"]
        )

    add_index_str = ""
    if add_index:
        add_index_str = " ".join(
            ["|", "awk", "-F '\t'", "-v", "OFS='\t'", "'{print $0,NR-1}'"]
        )

    merge_call_parts = [
        "bedtools genomecov",
        "-i",
        str(in_file),
        "-bga" if keep_zero_coverages else "-bg",
        f"-g {str(genome_file)}",
        coverage_filter_str,
        # Merge 'bookended' intervals (sequential with different coverage counts)
        "|",
        "bedtools merge",
        "-i",
        "stdin",
        "-d",
        str(max_distance),
        # Add interval index
        add_index_str,
        ">",
        str(out_file),
    ]

    # Remove empty strings and join parts
    merge_call = " ".join([x for x in merge_call_parts if x])
    call_subprocess(merge_call, "`bedtools::genomecov` failed")


def merge_overlapping_intervals(
    in_file: Union[str, pathlib.Path],
    out_file: Union[str, pathlib.Path],
    max_distance: int = 0,
    add_index: bool = False,
) -> None:
    """
    Merge the overlapping intervals of a single file with `bedtools::merge`.

    Parameters
    ----------
    in_file
        Path to the BED file to merge overlapping intervals of.
        Must be tab-separated.
    out_file
        Path to the output file. Cannot be the same as `in_file`.
    max_distance
        Maximum distance between intervals allowed for intervals
        to be merged. Default is 0. That is, overlapping and/or book-ended
        intervals are merged.
    add_index
        Whether to add an interval index column.
    """
    check_paths_for_subprocess(in_file, out_file)

    add_index_str = ""
    if add_index:
        add_index_str = " ".join(
            ["|", "awk", "-F '\t'", "-v", "OFS='\t'", "'{print $0,NR-1}'"]
        )

    merge_call = " ".join(
        [
            "bedtools merge",
            "-i",
            str(in_file),
            "-d",
            str(max_distance),
            add_index_str,
            ">",
            str(out_file),
        ]
    )
    call_subprocess(merge_call, "`bedtools::merge` failed")


def remove_non_standard_chromosomes(
    bed_df: pd.DataFrame, copy: bool = False
) -> pd.DataFrame:
    """
    Remove chromosomes where the name is not "chr" + some digit(s).

    Parameters
    ----------
    bed_df
        `pandas.DataFrame` with intervals. As created with `read_bed_as_df`.
    copy
        Whether to copy the data frame before making changes to it.
        Otherwise (default), the original data frame may be altered.

    Returns
    -------
    `bed_df` with filtered chromosomes.
    """
    if copy:
        bed_df = bed_df.copy()
    bed_df = bed_df[bed_df.chromosome.str.match(r"^chr\d+$", na=False)]
    return bed_df


def sort_intervals(
    bed_df: pd.DataFrame, copy: bool = False, sort_numerically: bool = False
) -> pd.DataFrame:
    """
    Sort BED data frame by chromosome and start position.

    NOTE: Some operations are made inplace and will affect the input data frame.
    Enable `copy` to avoid this.

    Parameters
    ----------
    bed_df
        `pandas.DataFrame` with intervals. As created with `read_bed_as_df`.
        NOTE: Can only have chromosome names consisting of 'chr' + some digit(s)
        e.g. chr1, chr2, etc. Use `remove_non_standard_chromosomes()` to
        remove non-standard chromosomes.
    copy
        Whether to copy the data frame before making changes to it.
        Otherwise (default), the original data frame may be altered.
    sort_numerically
        Whether to sort by chromosome number (or alphabetically).
        `True`: chr1, chr2, chr3, etc.
        `False`: chr1, chr10, chr11, etc.

    Returns
    -------
    `bed_df` sorted by chromosome and start position.
    """
    if copy:
        bed_df = bed_df.copy()
    if sort_numerically:
        bed_df["chr_sort"] = bed_df["chromosome"].map(lambda x: int(x[3:]))
    else:
        bed_df["chr_sort"] = bed_df["chromosome"]
    # Note: Without end and without using stable sort,
    # an end could come before other "lower-valued" ends
    bed_df.sort_values(by=["chr_sort", "start", "end"], inplace=True)
    bed_df.drop(columns=["chr_sort"], inplace=True)
    return bed_df
