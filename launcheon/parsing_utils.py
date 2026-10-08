"""Utils for parsing metrics and other logs to be displayed in through the `table` command"""

import json
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from typing import Sequence

from launcheon.global_variables import DEFAULT_METRIC_VALUE
from launcheon.utils import get_nested_metric, print_warning, safe_convert_to_number

Metric = str | int | float


def json_parser(
    json_file_path: str = "results.json",
    metrics: Sequence[str] = ("accuracy",),
    load_all: bool = False,
    verbose: bool = False,
) -> dict[str, Metric]:
    """An example command to get a metric for populating the `_exps_metrics` dict in
    Experiment grid. This functions pulls the given `metric` from the file
    `json_file_path` found in the current experiment's directory

    :param json_file_path: The json file to parse
    :param metrics: list of metrics names to retrieve from the json file
    :param load_all: Unused for this metric
    :param verbose: Optional verbosity

    :return: A dictionary mapping a metric name to its int/float value, or "N/A"
        if no value can be found.
    """
    del verbose
    assert not load_all, "json metrics can be parsed as time series"
    if isinstance(metrics, str):
        metrics = (metrics,)
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            out: dict[str, Metric] = {}
            for key in metrics:
                try:
                    aux = safe_convert_to_number(get_nested_metric(data, key.split("/")))
                    out[key] = aux if isinstance(aux, (int, float)) else DEFAULT_METRIC_VALUE
                except KeyError:
                    out[key] = DEFAULT_METRIC_VALUE
            return out
    except (OSError, FileNotFoundError, json.JSONDecodeError, KeyError):
        return {key: DEFAULT_METRIC_VALUE for key in metrics}


def jsonl_parser(
    jsonl_file_path: str = "train_logs.jsonl",
    metrics: str | Sequence[str] = ("loss",),
    load_all: bool = False,
    verbose: bool = False,
) -> dict[str, Metric | list[Metric]]:
    """Parsing a metric from a jsonl file (defaults to only parsing the
    last line for the given metric)


    :param jsonl_file_path: The json file to parse
    :param metrics: list of metrics names to retrieve from the json file
    :param load_all: If True, loads all rows (timesteps) for this metric.
        Otherwise, only loads the latest.
    :param verbose: Optional verbosity

    :return: A dictionary mapping a metric name to its int/float value(s),
        or "N/A" if no value can be found.
    """
    if isinstance(metrics, str):
        metrics = (metrics,)

    collected: dict[str, Metric | list[Metric]] = defaultdict(lambda: []) if load_all else {}
    try:
        with open(jsonl_file_path, "r") as jsonl_file:
            for line in jsonl_file.readlines()[(0 if load_all else -1) :]:
                data = json.loads(line)

                for key in metrics:
                    try:
                        aux = safe_convert_to_number(get_nested_metric(data, key.split("/")))
                        val = aux if isinstance(aux, (int, float)) else DEFAULT_METRIC_VALUE
                    except KeyError:
                        val = DEFAULT_METRIC_VALUE

                    if load_all:
                        collected[key].append(val)  # type: ignore
                    else:
                        collected[key] = val

            if "line" in locals():
                return collected
            raise UnboundLocalError  # empty file: we never went through the loop

    except (OSError, FileNotFoundError):
        if verbose:
            print_warning(f"Could not open metrics file: {jsonl_file_path}")
    except UnboundLocalError:
        if verbose:
            print_warning(f"Metrics file exist but is empty: {jsonl_file_path}")
    except json.JSONDecodeError:
        if verbose:
            print_warning(f"Could not parse metrics file as JSON: {jsonl_file_path}")
    except (
        KeyError,
        IndexError,
    ):
        if verbose:
            print_warning(f"Could not find metrics {metrics} in file: {jsonl_file_path}")
    if load_all:
        return {key: [DEFAULT_METRIC_VALUE] for key in metrics}
    return {key: DEFAULT_METRIC_VALUE for key in metrics}


def db_parser(
    db_file_path: str = "eval_results.db",
    table: str = "eval",
    metrics: Sequence[str] = ("accuracy",),
    load_all: bool = False,
    verbose: bool = False,
) -> dict[str, Metric | list[Metric]]:
    """Parsing a metric from a SQL database


    :param db_file_path: The database to parse
    :param table: Table to query inside the database
    :param metrics: list of metrics names (columns) to retrieve
    :param load_all: If True, loads all rows (timesteps) for this metric.
        Otherwise, only loads the latest.
    :param verbose: Optional verbosity

    :return: A dictionary mapping a metric name to its int/float value(s),
        or "N/A" if no value can be found.
    """
    try:
        # Note: sqlite3.connect would create an empty database if the file does not exist
        if not os.path.isfile(db_file_path):
            raise FileNotFoundError(db_file_path)
        with closing(sqlite3.connect(db_file_path)) as db:
            quoted_table = '"' + table.replace('"', '""') + '"'
            data = db.cursor().execute(f"SELECT * FROM {quoted_table}")
            column_names = [x[0] for x in data.description]
            values = sorted(data.fetchall(), key=lambda x: x[-1])
        if len(values) == 0:
            raise ValueError(f"Empty table {table}")
        if load_all:
            output: dict[str, Metric | list[Metric]] = {}
            for i, key in enumerate(column_names):
                if key not in metrics:
                    continue
                output[key] = [v[i] for v in values]
            return output

        return {key: v for key, v in zip(column_names, values[-1]) if key in metrics}
    except (sqlite3.Error, TypeError, ValueError, FileNotFoundError):
        if verbose:
            print_warning(
                f"Could not find metrics {metrics} in table {table} of database {db_file_path}"
            )
        if load_all:
            return {key: [DEFAULT_METRIC_VALUE] for key in metrics}
        return {key: DEFAULT_METRIC_VALUE for key in metrics}
