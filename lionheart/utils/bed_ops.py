import pathlib
import warnings
from typing import Callable, List, Optional, Union
import numpy as np
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
        Path to BED file with no header.
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
    cat_type = "unpigz -c" if str(in_file)[-3:] == ".gz" else "cat"
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


def split_nonzeros_by_chromosome(
    in_file: Union[str, pathlib.Path],
    out_dir: Union[str, pathlib.Path],
) -> None:
    check_paths_for_subprocess(in_file)

    split_call = " ".join(
        [
            "MAWK=$(command -v mawk >/dev/null 2>&1 && echo mawk || echo awk);",
            "LC_ALL=C",
            "$MAWK",
            "-F'\t'",
            "-v",
            "OFS='\t'",
            '-v outdir="' + str(out_dir) + '"',
            "'{",
            "i[$1]++;",
            # Print (index (zero-indexed), value) for nonzero rows
            'if($NF>0) print i[$1]-1, $NF > (outdir "/" $1 ".txt");',
            "}",
            "END {",
            'for (chr in i) print chr, i[chr] > (outdir "/total_rows.txt");',
            "}'",
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
    rm_non_autosomes: bool = False,
    pre_sort: bool = False,
    use_n_cols: Optional[int] = None,
    add_index: bool = False,
):
    """
    Merge the intervals of one or more BED files.
    Files are combined to a single file and sorted.
    Overlapping intervals are merged with `bedtools::merge`.

    Note: When given a single file, the overlapping intervals are still merged
    but there is no sorting step unless `pre_sort=True`.

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
    rm_non_autosomes
        Whether to remove non-autosomes.
    pre_sort
        Whether to sort before the merging.
        NOTE: Always happens when `in_files` has > 1 elements.
    min_coverage
        The number [>=1.0] / percentage ( [0.0-1.0[ ) of `in_files` that must
        overlap a position for it to be kept.
        Percentages are multiplied by the number of `in_files` and floored (rounded down).
        Ignored when `count_coverage` is `False`.
    max_distance
        Maximum distance between intervals allowed for intervals
        to be merged. Default is 0. That is, overlapping and/or book-ended
        intervals are merged.
    use_n_cols
        Optional number of most-left columns to use. E.g., when
        the files have different columns, we can select just the
        first three (chrom, start, end).
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

    # Merge the overlapping intervals
    if count_coverage:
        merge_overlapping_intervals_with_coverage(
            in_files=in_files,
            out_file=out_file,
            keep_zero_coverages=keep_zero_coverage,
            genome_file=genome_file,
            rm_non_autosomes=rm_non_autosomes,
            pre_sort=pre_sort,
            min_coverage=min_coverage,
            max_distance=max_distance,
            use_n_cols=use_n_cols,
            add_index=add_index,
        )
    else:
        merge_overlapping_intervals(
            in_files=in_files,
            out_file=out_file,
            rm_non_autosomes=rm_non_autosomes,
            pre_sort=pre_sort,
            max_distance=max_distance,
            use_n_cols=use_n_cols,
            add_index=add_index,
        )


def merge_overlapping_intervals_with_coverage(
    in_files: List[Union[str, pathlib.Path]],
    out_file: Union[str, pathlib.Path],
    genome_file: Union[str, pathlib.Path],
    rm_non_autosomes: bool = False,
    pre_sort: bool = False,
    keep_zero_coverages: bool = False,
    min_coverage: int = 0,
    max_distance: int = 0,
    use_n_cols: Optional[int] = None,
    add_index=False,
) -> None:
    """
    Merge the overlapping intervals of a single file with `bedtools::genomecov`.
    Get coverage counts of each subinterval.

    Parameters
    ----------
    in_files
        Path(s) to the BED file(s) to merge overlapping intervals of with counts of coverage.
        When multiple files are specified they are concatenated and sorted before merging.
        Must be tab-separated.
    out_file
        Path to the output file. Cannot be the same as `in_file`.
    genome_file
        Path to the genome file with chromosome sizes.
        Used to find limits of the chromosomes.
    rm_non_autosomes
        Whether to remove non-autosomes.
    pre_sort
        Whether to sort before the merging.
        NOTE: Always happens when `in_files` has > 1 elements.
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
    check_paths_for_subprocess(in_files + [genome_file], out_file)
    concat_str = _cat_files(
        in_files=in_files,
        rm_non_autosomes=rm_non_autosomes,
        always_sort=pre_sort,
        use_n_cols=use_n_cols,
    )

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
        # Cat file(s)
        concat_str,
        # Count coverage
        "bedtools genomecov",
        "-i",
        "-",
        "-bga" if keep_zero_coverages else "-bg",
        f"-g {str(genome_file)}",
        # Remove if too little coverage
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
    in_files: List[Union[str, pathlib.Path]],
    out_file: Union[str, pathlib.Path],
    rm_non_autosomes: bool = False,
    pre_sort: bool = False,
    max_distance: int = 0,
    use_n_cols: Optional[int] = None,
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
    rm_non_autosomes
        Whether to remove non-autosomes.
    pre_sort
        Whether to sort before the merging.
        NOTE: Always happens when `in_files` has > 1 elements.
    max_distance
        Maximum distance between intervals allowed for intervals
        to be merged. Default is 0. That is, overlapping and/or book-ended
        intervals are merged.
    add_index
        Whether to add an interval index column.
    """
    check_paths_for_subprocess(in_files, out_file)

    concat_str = _cat_files(
        in_files=in_files,
        rm_non_autosomes=rm_non_autosomes,
        always_sort=pre_sort,
        use_n_cols=use_n_cols,
    )

    add_index_str = ""
    if add_index:
        add_index_str = " ".join(
            ["|", "awk", "-F '\t'", "-v", "OFS='\t'", "'{print $0,NR-1}'"]
        )

    merge_call = " ".join(
        [
            # Cat file(s)
            concat_str,
            # Flatten intervals
            "bedtools merge",
            "-i",
            "-",
            "-d",
            str(max_distance),
            add_index_str,
            ">",
            str(out_file),
        ]
    )
    call_subprocess(merge_call, "`bedtools::merge` failed")


def subtract_intervals(
    in_file: Union[str, pathlib.Path],
    out_file: Union[str, pathlib.Path],
    exclude_file: Union[str, pathlib.Path],
    rm_full_if_any: bool = False,
) -> None:
    """
    Subtract the intervals of one file from another with `bedtools::subtract`.

    Parameters
    ----------
    in_file
        Path to the BED file to remove intervals from.
        Must be tab-separated.
    out_file
        Path to the output file. Cannot be the same as `in_file`.
    exclude_file
        Path to the BED file with intervals to remove intervals from `in_file`.
        Must be tab-separated.
    rm_full_if_any
        Remove entire feature if any overlap. Uses "-A" flag.
    """
    check_paths_for_subprocess([in_file, exclude_file], out_file)

    subtract_call = " ".join(
        [
            "bedtools subtract",
            f"-a {in_file}",
            f"-b {exclude_file}",
            "-A" if rm_full_if_any else "",
            ">",
            str(out_file),
        ]
    )
    call_subprocess(subtract_call, "`bedtools::subtract` failed")


def _cat_files(
    in_files: List[Union[str, pathlib.Path]],
    rm_non_autosomes: bool = False,
    always_sort: bool = False,
    use_n_cols: Optional[int] = None,
):
    """
    Create string for 'cat'ing the files, removing non-autosomes, and sorting if necessary (or specified).

    Detects ".gz" extension from the first file only. All `in_files` should thus be consistently
    without or without gzip compression!

    use_n_cols
        Use the first n columns only.
        NOTE: When the first column does not contain "chr*" (e.g., index)
        it detects if that pattern is in the second column
        in which case it starts from the second column.

    always_sort
        Whether to sort even when `in_files` only contains 1 file.
        Otherwise, only sorts when multiple files are specified
    """
    assert isinstance(in_files, list)
    cat_fn = "unpigz -c" if str(in_files[0])[-3:] == ".gz" else "cat"

    if use_n_cols is not None:
        awk_cmd = (
            f"awk -F'\t' -v OFS='\t' -v n={use_n_cols} '"
            "{"
            "if ($1 ~ /^chr/) { "
            '   for(i=1;i<=n;i++) { printf "%s%s", $i, (i<n ? OFS : ORS) } '
            "} else if ($2 ~ /^chr/) { "
            '   for(i=2;i<=n+1;i++) { printf "%s%s", $i, (i<n+1 ? OFS : ORS) } '
            "} else { "
            '   for(i=1;i<=n;i++) { printf "%s%s", $i, (i<n ? OFS : ORS) } '
            "}"
            "}'"
        )
        concat_str = (
            "( "
            + " ; ".join(
                [f"{cat_fn} {file} | {awk_cmd}" for file in _to_strings(in_files)]
            )
            + " ) | "
        )
    else:
        concat_str = " ".join([cat_fn] + _to_strings(in_files) + ["|"])

    if rm_non_autosomes:
        concat_str += " awk -F'\t' -v OFS='\t' '$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/' | "
    if len(in_files) > 1 or always_sort:
        concat_str += " sort -k1,1 -k2,2n -k3,3n | "
    return concat_str


def _to_strings(ls):
    """
    Convert a list of elements to strings
    by applying `str()` to each element.
    """
    return [str(s) for s in ls]
