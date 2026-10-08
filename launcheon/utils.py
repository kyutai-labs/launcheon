"""Various utils for pretty printing, resolving symlinks, etc."""

import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from functools import partial
from hashlib import sha1
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Sequence

import rich

from launcheon.global_variables import DEFAULT_METRIC_VALUE
from launcheon.parser_format import ParserFormat


def sort_mixed_int_str(lst: list[int | str]) -> list[int | str]:
    """Sort a list of mixed integers and strings such that integers take precedence"""

    def sortkey(a: int | str) -> Any:
        if isinstance(a, int):
            return (0, a)
        return (1, a)

    return sorted(lst, key=sortkey)


def parse_as_tuple(x: str | tuple[str, ...]) -> tuple[str, ...]:
    """Parse the input argument x as a tuple

    :param x: Input argument either directly given as a tuple, or as a comma-separated
    string of values

    :return: A tuple of strings given by `x`
    """
    if isinstance(x, str):
        return tuple(x.split(","))
    if isinstance(x, tuple):
        return x
    if isinstance(x, (set, list)):
        return tuple(x)
    raise ValueError


def print_error(msg: str) -> None:
    """Print an error message"""
    rich.print(f"[bold red]ERROR:[/bold red] {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message"""
    rich.print(f"[bold yellow]WARN:[/bold yellow] {msg}")


def print_file_content(file_path: str, num_lines: int | None = None) -> None:
    """Display the given file's contents

    :param file_path: File to display
    :param num_lines: How many lines of the file to display: Starts from the beginning
        if a positive number, otherwise from the end. If None, displays the whole file
    """
    if os.path.exists(file_path):
        if num_lines is None:
            printcmd = ["cat"]
        elif num_lines > 0:
            printcmd = ["head", "-n", str(num_lines)]
        else:
            printcmd = ["tail", "-n", str(-num_lines)]
        subprocess.run([*printcmd, file_path], check=False)
    else:
        rich.print(f"[bright_black]File {file_path} does not exist yet[/bright_black]")


def is_singleton_kwarg(v: Any) -> bool:
    """Check whether the value `v` of a kwargs found in the `kwargs_dict` is a singleton,
    i.e., if it has a unique value throughout the experiment grid"""
    return (
        (isinstance(v, (tuple, list)) and len(v) == 1)
        or not isinstance(v, Iterable)
        or isinstance(v, str)
    )


def resolve_duplicate_key_conflict(key_a: str, key_b: str, vals_a: Any, vals_b: Any) -> str:
    """Resolve conflict between duplicate keys in the sweep, if possible.
    Otherwise, exits/

    :param key_a: Key a
    :param key_b: Conflicting Key b (either equal to key a, or is a nickname thereof)
    :param vals_a: Sweep associated to key a
    :param vals_b: Sweep associated to key b
    """
    z_a = is_singleton_kwarg(vals_a)
    z_b = is_singleton_kwarg(vals_b)
    # tiebreaking rule
    if int(z_a) + int(z_b) == 1:
        mi, ma = (key_a, key_b) if z_a else (key_b, key_a)
        vi, va = (vals_a, vals_b) if z_a else (vals_b, vals_a)
        print_warning(
            f"Found duplicate key/nickname"
            f" [cyan]{key_a}[/cyan] and [cyan]{key_b}[/cyan] in experiment sweep:\n"
            f"  > {ma} ({va}) takes precedence as "
            f"{mi} is a singleton ({vi})"
        )
        return mi
    # otherwise, cannot resolve
    print_error(
        f"Found duplicate key/nickname"
        f" [cyan]{key_a}[/cyan] and [cyan]{key_b}[/cyan] in experiment sweep. "
        "Cannot resolve the tie:"
        f"\n  - [cyan]{key_a}:[/cyan] {vals_a}"
        f"\n  - [cyan]{key_b}:[/cyan] {vals_b}",
    )
    sys.exit(1)


def readable_range(indices: Sequence[int]) -> str:
    """Returns the given list of indices as a readable inclusive range

    :param indices: A list of integers

    :return: A string representing the range, e.g.:
      * [0, 1, 2, 3, 4] becomes 0-4
      * [1, 2, 5, 6, 7] becomes 1-2,5-7
    """
    range_name = ""
    last_written, last_seen = -1, -1
    for cnt, idx in enumerate(sorted(indices)):
        if last_seen < 0:
            range_name += str(idx)
            last_written = idx
        elif idx > last_seen + 1:
            if last_seen > last_written:
                range_name += f"-{last_seen},{idx}"
            else:
                range_name += f",{idx}"
            last_written = idx
        elif cnt == len(indices) - 1:
            range_name += f"-{idx}"
        last_seen = idx
    return range_name


def parse_range(selection: str, max_range: int | None = None) -> list[int]:
    """This is the inverse of `readable_range`; it parses a range string and
    returns the corresponding (sorted) list of indices"""
    index_selection: list[int] = []
    for s in selection.split(","):
        aux = s.strip().split(":")
        assert len(aux) <= 2
        rng = aux[0].split("-")
        assert len(rng) <= 2
        try:
            if len(rng) == 1:
                index_selection.append(int(rng[0]))  # single int
            else:
                # e.g. when parsing repeated flags, we do not know max_range
                # so the user should always specify complete ranges
                if rng[1] == "":
                    if max_range is None:
                        raise ValueError
                    boundary = max_range
                else:
                    if max_range is not None:
                        boundary = min(max_range, int(rng[1])) + 1
                    else:
                        boundary = int(rng[1]) + 1

                index_selection.extend(
                    list(
                        range(
                            int(rng[0]),
                            boundary,
                            1 if len(aux) == 1 else int(aux[1]),
                        )
                    )
                )
        except ValueError as e:
            raise ValueError("Error in selection format:", selection) from e

    return sorted(list(set(index_selection)))


def find_repeated_flags(
    kwargs_dict: dict[str, Any],
    repeated_flags: dict[str, dict[int | str, Any]] | None = None,
    allow_override_initial_value: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Parse potentially repeated flags in the kwargs dictionary.
    These should be formatted as "flag_name:X": list_of_values, where:
      * `X corresponds to a range of positions
      * `list_of_values` corresponds to the associated values

    This can be used to sweep over repeated flags such as the `append`
    action in argparse, or the `multiple=True` option in click.

    :param kwargs_dict: Dictionary of flag/value combination to analyze
    :param repeated_flags: Dictionary mapping a repeated flag name to its
        tuple of value. Note that this will be changed in-place by this function
    :param allow_override_initial_values: If True, values in `kwargs_dict`
        are allowed to override values in the provided `repeated_flags`

    :return: repeated_flags, potentially modified. And the list of keys
        found repeated between kwargs and repeated_flags (typically, these
        should be deleted from kwargs after this function call)
    """
    if repeated_flags is None:
        repeated_flags = defaultdict(lambda: {})

    has_changed: dict[str, dict[int | str, bool]] = defaultdict(lambda: {})
    positions: Sequence[int | str]

    to_remove = []
    found_repeat_keys = {}
    for key, values in kwargs_dict.items():
        # repeated flags given as a range subset
        if ":" in key:
            to_remove.append(key)
            aux = key.split(":", 1)
            key = aux[0]
            try:
                # default case "param_value:0"
                positions = parse_range(aux[1])
            except ValueError:
                # alternatively we can index by group name "param_value:group_name"
                positions = [aux[1]]
            found_repeat_keys[key] = True
            if is_singleton_kwarg(values):
                values = (values,)
            if len(values) != len(positions):
                print_error(
                    "Mismatch in length for repeated flag "
                    f"[cyan]{to_remove[-1]}[/cyan] with values {values}"
                )
                sys.exit(1)
        # repeated flags given as a full range
        elif isinstance(values, tuple):
            to_remove.append(key)
            found_repeat_keys[key] = True
            positions = list(range(len(values)))
        else:
            continue

        for idx, v in zip(positions, values):
            if idx in repeated_flags[key]:
                # if we are not allowing overrides
                if not allow_override_initial_value:
                    print_error(
                        f"Conflicting value found for repeated flag [cyan]{key}[/cyan]"
                        f" in position {idx} (found {repeated_flags[key]} and {v})"
                    )
                    sys.exit(1)
                # if we are allowing override, we can only allow one change. If any more, this
                # means there is a duplicate in kwargs_dict itself
                elif allow_override_initial_value and has_changed[key].get(idx, False):
                    print_error(
                        f"Conflicting value found for repeated flag [cyan]{key}[/cyan]"
                        f" in position {idx} (found {repeated_flags[key]} and {v})"
                    )
                    sys.exit(1)
            repeated_flags[key][idx] = v
            has_changed[key][idx] = True

    for k in to_remove:
        del kwargs_dict[k]

    return repeated_flags, set(found_repeat_keys.keys())


def __non_default_op__(x: Sequence[Any], op: Callable) -> Any:
    """Compute `op` on the sequence x, ignoring missing data"""
    flt = [z for z in x if isinstance(z, (int, float))]
    if len(flt) == 0:
        return DEFAULT_METRIC_VALUE
    return op(flt)


max_non_default = partial(__non_default_op__, op=max)

min_non_default = partial(__non_default_op__, op=min)

mean_non_default = partial(__non_default_op__, op=lambda x: sum(x) / len(x))


def std_non_default(x: Any) -> Any:
    """Std function with potentially undefined values"""
    flt = [z for z in x if z != DEFAULT_METRIC_VALUE]
    if len(flt) == 0:
        return 0.0
    m = sum(flt) / len(flt)
    v = sum((z - m) ** 2 for z in flt) / len(flt)
    return sqrt(v)


def reverse_search_dict(x: Any, d: dict) -> Any:
    """Reverse search x in the dict `d` values and returns the corresponding key if found"""
    try:
        is_in = list(d.values()).index(x)
        return list(d.keys())[is_in]
    except (ValueError, KeyError):
        return KeyError


def hash_experiment_name(name: str, trim: int = 10) -> str:
    """Shortened unique hash for the given experiment name
    Source: https://stackoverflow.com/questions/2510716/short-python-alphanumeric-hash-with-minimal-collisions

    :param name: Name to hash
    :param trim: Trade-off / length of the hash
    """
    return sha1(name.encode("utf8")).hexdigest()[:trim]


def readable_timestamp(t: float | None = None) -> str:
    """Convert a timestamp to a nicely readable format

    :param t: Timestamp. Will default to current time if `None`

    :return: Pretty-printed timestamp as a string
    """
    return (datetime.now() if t is None else datetime.fromtimestamp(t)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def readable_elapsed(t: float) -> str:
    """Print an elapsed time in a pretty readable format

    :param t: Elapsed time in second

    :return: Pretty-formatted string
    """
    sign_string = "-" if t < 0 else ""
    seconds = abs(int(t))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{sign_string}{days}d{hours}h{minutes}m{seconds}s"
    if hours > 0:
        return f"{sign_string}{hours}h{minutes}m{seconds}s"
    if minutes > 0:
        return f"{sign_string}{minutes}m{seconds}s"
    return f"{sign_string}{seconds}s"


def file_or_link_timestamp(path: str) -> float:
    """Returns the timestamp of the given file (or, if symbolic link, of
    the file it ultimately points to)

    :param path: Path or symlink

    :return: Time the file (or symlink target) was last modified. -1 if the file
        cannot be accessed or does not exist.
    """
    try:
        if os.path.islink(path):
            path = str(Path(path).resolve())
        if os.path.exists(path):
            return os.path.getmtime(path)
        return -1.0
    except PermissionError:
        return -1.0


def file_or_link_exists(path: str) -> bool:
    """Check if a file exists (or, in the case of a symbolic link, if its final target exists)

    :param path: Path or symlink

    :return: True iff the file (or symlink target) exists and can be accessed
    """
    try:
        if os.path.islink(path):
            return Path(path).resolve().exists()
        return os.path.exists(path)
    except PermissionError:
        return False


def safe_convert_to_number(s: Any) -> int | float | str:
    """Convert the input string to a float if possible. Otherwise, returns the input string

    :param s: A string

    :return: A float | int if `s` can be convert to a number. Otherwise, return `s` unchanged.
    """
    if isinstance(s, str):
        try:
            try:
                return int(s)
            except ValueError:
                return float(s)
        except ValueError:
            pass
    return s


def resolve_kwargs_nicknames(
    kwargs: dict[str, Any],
    kwargs_nicknames: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve all possible nicknames for keys and values in the given `kwargs` dictionary

    :param kwargs: A dictionary mapping a flag name to its value. If the value is `None`, then
        it's a boolean flag that does not require a value and only needs to be present/absent
    :param kwargs_nickcnames: (Optional) Maps a flag or value name from `kwargs` to another
        string that will be used to replace it in the formatted output.
        This can be used to use shorter aliases in the kwargs dictionary (for both keys and values)
        and expand them to their true meaning when creating the final command
    """
    resolved_flags = {}
    for k, v in kwargs.items():
        expanded_k = k if kwargs_nicknames is None else kwargs_nicknames.get(k, k)
        expanded_v: Any
        # tuple values for repeated flags
        if isinstance(v, tuple):
            expanded_v = tuple(
                x if kwargs_nicknames is None else kwargs_nicknames.get(x, x) for x in v
            )
        # otherwise, single values flags
        else:
            expanded_v = v if kwargs_nicknames is None else kwargs_nicknames.get(v, v)
        resolved_flags[expanded_k] = expanded_v
    return resolved_flags


def format_kwargs(
    kwargs: dict[str, Any],
    parser_format: ParserFormat,
    kwargs_nicknames: dict[str, str] | None = None,
) -> str:
    """Format a dictionary of command flag to a string.

    :param kwargs: A dictionary mapping a flag name to its value. If the value is `None`, then
        it's a boolean flag that does not require a value and only needs to be present/absent
    :param kwargs_nickcnames: (Optional) Maps a flag or value name from `kwargs` to another
        string that will be used to replace it in the formatted output.
        This can be used to use shorter aliases in the kwargs dictionary (for both keys and values)
        and expand them to their true meaning when creating the final command
    :param quote_strings: If True, will add extra quotes around the values of type string
    :param formatting: A format string indicating the keyword format. Defaults to "--flag value"
    :param formatting separator: Defines how to joint the formatted flag/value pairs

    :return: The correctly formatted flag/value pair
    """
    formatted_flags = []
    for k, v in resolve_kwargs_nicknames(kwargs, kwargs_nicknames=kwargs_nicknames).items():
        # None resolves as no-value flag
        if v is None:
            formatted_flags.append(
                parser_format.kwargs_formatting.format(flag=k, value=parser_format.none_value)
            )
        # tuple resolve as repeated kwarg
        elif isinstance(v, tuple):
            accumulated_value = []
            for mult in v:
                if isinstance(mult, str) and parser_format.quote_strings:
                    q = re.escape(parser_format.quote_strings_symbol)
                    mult = re.sub(rf"(?<!\\){q}", rf"\\{parser_format.quote_strings_symbol}", mult)
                    mult = f"{parser_format.quote_strings_symbol}{mult}{parser_format.quote_strings_symbol}"
                if parser_format.repeat_key_in_repeated_args:
                    formatted_flags.append(
                        parser_format.kwargs_formatting.format(flag=k, value=mult)
                    )
                else:
                    accumulated_value.append(mult)
            if not parser_format.repeat_key_in_repeated_args:
                formatted_flags.append(
                    parser_format.kwargs_formatting.format(
                        flag=k, value=" ".join(str(x) for x in accumulated_value)
                    )
                )
        else:
            if isinstance(v, str) and parser_format.quote_strings:
                q = re.escape(parser_format.quote_strings_symbol)
                v = re.sub(rf"(?<!\\){q}", rf"\\{parser_format.quote_strings_symbol}", v)
                v = f"{parser_format.quote_strings_symbol}{v}{parser_format.quote_strings_symbol}"
            formatted_flags.append(parser_format.kwargs_formatting.format(flag=k, value=v))
    return parser_format.kwargs_formatting_separator.join(x.strip() for x in formatted_flags)


def get_nested_metric(d: dict, nested_key: Sequence[str]) -> Any:
    """Index a dictionary by the given nested key

    :param d: Target dictionary
    :param nested_key: Sequence of strings

    :return: The value of the dictionary found under d[nested_key[0]]....[nested_key[-1]]
    Raises a Key Error if the nested_key is empty or not found in the dict
    """
    if len(nested_key) == 0:
        raise KeyError
    if len(nested_key) == 1:
        return d[nested_key[0]]
    return get_nested_metric(d[nested_key[0]], nested_key[1:])


def strip_url_credentials(url: str) -> str:
    """Remove credentials embedded in a http(s) URL (e.g. https://user:token@host/...)"""
    return re.sub(r"^(https?://)[^/@]*@", r"\1", url)


def get_git_revision_hash() -> dict[str, str]:
    """Return current git branch and commit as a dictionary"""
    try:
        git_repo = (
            subprocess.check_output("git config --get remote.origin.url", shell=True)
            .decode("utf-8")
            .replace("\n", "")
            .strip()
        )
        git_branch = (
            subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
            .decode("ascii")
            .strip()
        )
        commit_hash = (
            subprocess.check_output("git rev-parse HEAD", shell=True).decode("ascii").strip()
        )
    except subprocess.CalledProcessError:
        return {"branch": "", "commit": "", "repo": ""}

    return {"branch": git_branch, "commit": commit_hash, "repo": git_repo}
