import subprocess
import pathlib


# TODO Add docs
def call_subprocess(call_, message, return_output=False, encoding="UTF-8"):
    # With capturing of output
    if return_output:
        try:
            out = subprocess.check_output(call_, shell=True, encoding=encoding)
        except subprocess.CalledProcessError as e:
            print(f"{message}: {call_}")
            raise e
        return out

    # Without capturing of output
    try:
        subprocess.check_call(call_, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"{message}: {call_}")
        raise e


def check_paths_for_subprocess(in_files, out_file=None, allow_none=False):
    """
    Check that the `in_files` are unique and exist,
    and that none of them are the same as the `out_file`.

    :param in_files: List of file paths to check.
    :param out_file: Path to expected output file.
    :param allow_none: Whether to allow elements in `in_files` to be `None`,
        in which case those elements will be ignored.
    """
    if in_files is None or isinstance(in_files, (str, pathlib.PurePath)):
        in_files = [in_files]
    if allow_none:
        in_files = [f for f in in_files if f is not None]
    assert len(set(in_files)) == len(
        in_files
    ), "`in_files` must be unique. Found duplicates."
    if out_file is not None:
        out_file = pathlib.Path(out_file).resolve()
    for in_file in in_files:
        in_file = pathlib.Path(in_file).resolve()
        assert in_file.exists(), f"`in_file` was not found: {in_file}"
        if out_file is not None:
            assert (
                in_file != out_file
            ), f"`in_file` and `out_file` cannot be the same. Got in: {in_file} ; out: {out_file}"
