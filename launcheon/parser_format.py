"""Handle different parser formatting"""

import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ParserFormat:
    # A string template for formatting a single flag-value pair, where {flag} and {value} are placeholders for the flag name and its corresponding value, respectively.
    kwargs_formatting: str
    # for repeated args we can choose out of two formats:
    # etiher the `append` style from argparse (default)
    # e.g. --foo 2 --foo 3 --foo 4 -> foo = [2,3,4]
    # or the `nargs=*` style from argparse
    # e.g. --foo 2 3 4 -> foo = [2,3,4]
    # we don't allow for a mix though
    repeat_key_in_repeated_args: bool
    # Whether to wrap string values in quotes when formatting command-line arguments. If True, string values will be enclosed in quotes (e.g., "value"), which can be useful for handling values that contain spaces or special characters. If False, string values will be included as-is without quotes.
    quote_strings: bool
    # The string representation to use for None values when formatting command-line arguments. For example, if none_value is set to "null", then any None values will be represented as "null" in the formatted command-line string.
    none_value: str
    # Symbol use to wrap quotes when quote_strings is True. For example, if quote_strings_symbol is set to '"', then string values will be wrapped in double quotes (e.g., "value"). If set to "'", then string values will be wrapped in single quotes (e.g., 'value').
    quote_strings_symbol: str = '"'
    # A string that separates multiple flag-value pairs when formatting command-line arguments. For example, if kwargs_formatting_separator is set to " ", then multiple flag-value pairs would be separated by a space in the final command-line string.
    kwargs_formatting_separator: str = " "

    def dry_run(self, cmd: str) -> str | None:
        del cmd
        raise NotImplementedError("This method should be implemented by subclasses")


@dataclass(frozen=True)
class HydraParserFormat(ParserFormat):
    def __init__(self):
        super().__init__(
            kwargs_formatting="{flag}={value}",
            kwargs_formatting_separator=" ",
            repeat_key_in_repeated_args=False,
            quote_strings=True,
            quote_strings_symbol='"',
            none_value="null",
        )

    def dry_run(self, cmd: str) -> str | None:
        """Validate a Hydra command by composing its config without running the main function.

        First does a static per-override syntax check to pinpoint bad args,
        then falls back to a full --cfg all run for config-level errors.
        Returns None on success, or the error string on failure.
        """

        # Otherwise default to subprocess dry_run
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        result = subprocess.run(
            f"{cmd} --cfg all",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raw = result.stderr or result.stdout
            # Strip torch/distributed boilerplate lines
            cleaned_lines = [
                line
                for line in raw.splitlines()
                if not re.match(r"^[WE]\d{4} \d{2}:\d{2}:\d{2}", line)
                and line.strip() not in ("", "*****************************************")
            ]
            error_output = "\n".join(cleaned_lines).strip()
            return error_output
        return None


@dataclass(frozen=True)
class ArgparseParserFormat(ParserFormat):
    def __init__(self):
        super().__init__(
            kwargs_formatting="--{flag} {value}",
            kwargs_formatting_separator=" ",
            repeat_key_in_repeated_args=True,
            quote_strings=True,
            none_value="",
        )

    def dry_run(self, cmd: str) -> str | None:
        """Returns None on success, or the error string on failure."""
        result = subprocess.run(
            f"{cmd} --help",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return (
                result.stderr
                or result.stdout
                or f"Command '{cmd} --help' failed with exit code {result.returncode}"
            )
        return None


@dataclass(frozen=True)
class SrunArgsParserFormat(ParserFormat):
    def __init__(self):
        super().__init__(
            kwargs_formatting="--{flag} {value}",
            kwargs_formatting_separator=" ",
            quote_strings=False,
            quote_strings_symbol='"',
            none_value="",
            repeat_key_in_repeated_args=False,
        )


class ParserFormatType(Enum):
    HYDRA = "hydra"
    ARGPARSE = "argparse"

    def get_parser_format(self) -> ParserFormat:
        if self == ParserFormatType.HYDRA:
            return HydraParserFormat()
        elif self == ParserFormatType.ARGPARSE:
            return ArgparseParserFormat()
        else:
            raise ValueError(f"Unsupported parser format type: {self}")
