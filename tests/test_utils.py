from typing import Tuple

import pytest

from launcheon.parser_format import ParserFormat
from launcheon.utils import find_repeated_flags, format_kwargs, parse_range, readable_range


def test_readable_range_interval(min_range: int = 3, max_range: int = 9) -> None:
    indices = list(range(min_range, max_range))
    assert parse_range(readable_range(indices)) == indices
    assert readable_range(indices) == f"{min_range}-{max_range - 1}"


def test_readable_unbounded_range_interval(min_range: int = 3, max_range: int = 9) -> None:
    indices = list(range(min_range, max_range))
    assert parse_range(f"{min_range}-", max_range=max_range) == indices
    with pytest.raises(ValueError):
        parse_range(f"{min_range}-", max_range=None)


def test_readable_range_interval_step(
    min_range: int = 3, max_range: int = 9, step: int = 2
) -> None:
    indices = list(range(min_range, max_range, step))
    assert parse_range(readable_range(indices)) == indices
    assert parse_range(f"{min_range}-{max_range - 1}:{step}") == indices


def test_readable_range_interval_seq(indices: Tuple[int, ...] = (2, 47, 3, 9, 10)) -> None:
    assert sorted(parse_range(readable_range(indices))) == sorted(indices)


def test_format_kwargs_escapes_quote_symbol() -> None:
    fmt = ParserFormat(
        kwargs_formatting="{flag}={value}",
        kwargs_formatting_separator=" ",
        repeat_key_in_repeated_args=False,
        quote_strings=True,
        none_value="null",
        quote_strings_symbol="'",
    )  #
    # Unescaped quotes get a backslash added
    result = format_kwargs({"phrase": "l'apostrophe del'amour"}, fmt)
    assert result == r"phrase='l\'apostrophe del\'amour'"

    # Already-escaped quotes are left alone (no double-escaping)
    result = format_kwargs({"phrase": r"l\'apostrophe del'amour"}, fmt)
    assert result == r"phrase='l\'apostrophe del\'amour'"


def test_no_repeated_flags() -> None:
    repeated_flags, repeated_keys = find_repeated_flags({"gloubiboulga": 42})
    assert len(repeated_flags) == 0
    assert len(repeated_keys) == 0


def test_repeated_flags_override() -> None:
    # error 1: repeat without override
    repeated_flags = {"gloubiboulga": {0: 10}}
    with pytest.raises(SystemExit):
        repeated_flags, repeated_keys = find_repeated_flags(
            {"gloubiboulga:0": 42}, repeated_flags=repeated_flags
        )

    # error 1bis: self repeat
    with pytest.raises(SystemExit):
        repeated_flags, repeated_keys = find_repeated_flags(
            {"gloubiboulga:0": 42, "gloubiboulga": (38, 44)}, repeated_flags=None
        )

    # error 2: allow override, but too many repeates
    repeated_flags = {"gloubiboulga": {0: 10}}
    with pytest.raises(SystemExit):
        repeated_flags, repeated_keys = find_repeated_flags(
            {"gloubiboulga:0": 42, "gloubiboulga": (28,)}, repeated_flags=repeated_flags
        )

    # good 1: one repeat with override
    repeated_flags = {"gloubiboulga": {0: 10}}
    repeated_flags, repeated_keys = find_repeated_flags(
        {"gloubiboulga:0": 42},
        repeated_flags=repeated_flags,
        allow_override_initial_value=True,
    )
    assert repeated_flags["gloubiboulga"][0] == 42
    assert len(repeated_keys) == 1
    assert "gloubiboulga" in repeated_keys

    # good 2: one repeat with override, different format
    repeated_flags = {"gloubiboulga": {0: 10}}
    repeated_flags, repeated_keys = find_repeated_flags(
        {"gloubiboulga": (38,)},
        repeated_flags=repeated_flags,
        allow_override_initial_value=True,
    )
    assert repeated_flags["gloubiboulga"][0] == 38
    assert len(repeated_keys) == 1
    assert "gloubiboulga" in repeated_keys

    # good 3: 1 repeat and 1 addition
    repeated_flags = {"gloubiboulga": {0: 10, 1: 29}}
    repeated_flags, repeated_keys = find_repeated_flags(
        {"gloubiboulga:1-2": (38, 40)},
        repeated_flags=repeated_flags,
        allow_override_initial_value=True,
    )
    assert repeated_flags["gloubiboulga"][0] == 10
    assert repeated_flags["gloubiboulga"][1] == 38
    assert repeated_flags["gloubiboulga"][2] == 40
    assert len(repeated_keys) == 1
    assert "gloubiboulga" in repeated_keys
