# pyright: reportInvalidStringEscapeSequence=false
"""Utils to plot the different table modes in launcheon"""

import bisect
import operator
import re
import sys
from collections import defaultdict
from functools import reduce
from itertools import product
from random import sample
from typing import Any, Callable, Literal

import rich
import rich.markdown
import rich.markup
from rich.table import Table

from launcheon.global_variables import (
    DEFAULT_METRIC_VALUE,
    HIGHLIGHT_COLORS,
    PRETTY_TABLE_COLORS,
    RICH_CUSTOM_BOX,
)
from launcheon.utils import (
    max_non_default,
    mean_non_default,
    min_non_default,
    parse_as_tuple,
    print_error,
    print_warning,
    readable_timestamp,
    std_non_default,
)


def export_table(
    table: Table,
    export_to: Literal["markdown", "latex"],
    ascending: bool = False,
    keep_first_col_as_comment: bool = False,
):
    data = [
        [re.sub(re.compile(r"\[[^(\[\])]*\]"), "", str(x)) for x in col._cells]
        for col in table.columns
    ]
    # max_cell size per column
    max_cell_size = [max(len(x) for x in col_cells) for col_cells in data]
    # let's find the best per row and best overall
    max_row_per_col = []
    max_row_overall, max_val_overall = [], None
    for col_idx, col_cells in enumerate(data):
        max_val = None
        max_row = []
        for row, c in enumerate(col_cells):
            try:
                v = float(c)
                if ascending:
                    v = -v
                if max_val is None or v > max_val:
                    max_row = [row]
                    max_val = v
                elif v == max_val:
                    max_row.append(row)
            except ValueError:
                pass
        max_row_per_col.append(max_row)
        if max_val is not None:
            if max_val_overall is None or (
                max_val < max_val_overall if ascending else max_val > max_val_overall
            ):
                max_row_overall = [col_idx]
                max_val_overall = max_val
            elif max_val == max_val_overall:
                max_row_overall.append(col_idx)

    if export_to == "markdown":
        md_strings = [[] for _ in range(len(table.columns[0]._cells) + 2)]
        for col_idx, col in enumerate(table.columns):
            md_strings[0].append(f"  {str(col.header).replace('_acc', '')}  ")
            md_strings[1].append("-" * len(md_strings[0][-1]))
            for i, ct in enumerate(data[col_idx]):
                md_strings[2 + i].append(f"{ct}{' ' * max(0, max_cell_size[col_idx] - len(ct))}")
        rich.print(
            "[yellow]Markdown start ----------------------------------------------[/yellow]\n"
        )
        print(
            "\n".join(
                "|  "
                + "  |  ".join(
                    (
                        (f"**{z.strip()}**" if col in max_row_overall else f"*{z.strip()}*")
                        if row in max_row_per_col[col]
                        else str(z)
                    )
                    for col, z in enumerate(x)
                )
                + "  |"
                for row, x in enumerate(md_strings)
            )
        )
        rich.print("\n[yellow]---------------------------------------------- Markdown end[/yellow]")
    elif export_to == "latex":
        md_strings = [[] for _ in range(len(table.columns[0]._cells) + 1)]
        comments = []
        for col_idx, col in enumerate(table.columns):
            if keep_first_col_as_comment and col_idx == 0:
                comments = [col.header]
            else:
                md_strings[0].append(str(col.header).replace("_acc", "").replace("_", "\\_"))
            for i, ct in enumerate(data[col_idx]):
                ct = ct.replace("_", "\\_")
                if keep_first_col_as_comment and col_idx == 0:
                    comments.append(ct)
                else:
                    md_strings[1 + i].append(
                        f"{ct}{' ' * max(0, max_cell_size[col_idx] - len(ct))}"
                    )
        rich.print("[yellow]LaTeX start ----------------------------------------------[/yellow]\n")
        print(
            """\\begin{table}[h!]
\\centering
\\resizebox{\\textwidth}{!}{""",
            end="",
        )
        fmt = "l" + "c" * (len(table.columns) - 1 - int(keep_first_col_as_comment))
        print(f"""
\\begin{{tabular}}{{{fmt}}}
\\toprule""")

        print("\t" + "  &  ".join(f"\\sc{{{x}}}" for x in md_strings[0]) + "  \\\\")
        print("\t\\midrule")
        print(
            "\t"
            + "\n\t".join(
                "  &  ".join(
                    (
                        f"\\sc{{{z}}}"
                        if col == 0
                        else (
                            f"\\textbf{{\\underline{{{z}}}}}"
                            if col in max_row_overall
                            else f"\\underline{{{z}}}"
                        )
                        if row in max_row_per_col[col + int(keep_first_col_as_comment)]
                        else str(z)
                    )
                    for col, z in enumerate(x)
                )
                + "  \\\\"
                + (
                    f"  % {comments[row]}"
                    if keep_first_col_as_comment and row < len(comments)
                    else ""
                )
                for row, x in enumerate(md_strings[1:])
            )
        )
        cap = re.sub(re.compile(r"\[[^(\[\])]*\]"), "", str(table.caption))
        print(f"""
\\bottomrule
\\end{{tabular}}
}}
\\caption{{{cap}}}
\\end{{table}}""")
        rich.print("\n[yellow]---------------------------------------------- LaTeX end[/yellow]")


def table_per_exp(
    grid: "ExperimentGrid",  # type: ignore  # noqa: F821
    columns: str | tuple[str, ...] | None = None,
    exclude: str | tuple[str, ...] | None = None,
    metrics: str | tuple[str, ...] | None = None,
    sort: bool | str = False,
    ascending: bool = False,
    muted: bool = False,
    short: bool = False,
    float_fmt: str = "{:.3f}",
    return_table: bool = False,
    previous_color_maps: dict[str, dict[str, str]] | None = None,
    include_job_info: bool = False,
    show_fixed: bool = False,
    export_to: Literal["markdown", "latex"] | None = None,
) -> tuple[Table, dict[str, dict[str, str]]] | None:
    """Pretty print a list of the experiments in table form where each row represents an experiment.

    :param columns: Which kwargs to display as columns. Defaults to all columns / rows
    :param exclude: "exclusive" counterpart of `--columns`
    :param metrics: If given, additionally display the given metrics as extra columns
    :param sort: Sort table by the given metric.
    :param ascending: Controls the sorting order
    :param muted: If given, do not display the table in colors
    :param short: If given, do not expand the keyword arguments inside groups
    :param float_fmt: Format for float numbers
    :param include_job_info: Whether to include the Job info and status in the table
    """
    color_maps: dict[str, dict[str, str]] = (
        defaultdict(lambda: {}) if previous_color_maps is None else previous_color_maps
    )
    table = Table(
        caption=(
            f"Experiment Grid for `{grid.base_name}`"
            f" ([bold]{len(grid)}[/bold] xps) "
            f"- last update: {readable_timestamp(t=None)}\n"
            f"({grid.log_dir_root})"
        ),
        box=RICH_CUSTOM_BOX,
        show_lines=True,
    )
    # selects which column to display
    keep_column = defaultdict(lambda: columns is None)
    if columns is not None:
        for c in parse_as_tuple(columns):
            keep_column[c] = True
    if exclude is not None:
        for c in parse_as_tuple(exclude):
            keep_column[c] = False

    # Reverse kwargs_nicknames to map the underlying flags to their nicknames
    reverse_nicknames = (
        {} if grid.kwargs_nicknames is None else {v: k for k, v in grid.kwargs_nicknames.items()}
    )

    rows_aggregate: dict[int, dict[str, Any]] = defaultdict(lambda: {})
    cols_aggregate: dict[str, dict[Any, bool]] = defaultdict(lambda: {})
    is_flag_group = defaultdict(lambda: False)
    for exp in grid:
        # Iterate over fully expanded arguments
        expanded_kwargs = exp.get_full_kwargs_dict()
        for flag, value in expanded_kwargs.items():
            # If --short is True, we do not display which are inside a group
            if short and (
                flag in exp.kwarg_to_group
                or reverse_nicknames.get(flag, flag) in exp.kwarg_to_group
            ):
                continue
            # Otherwise, store the value for this flag
            if keep_column[flag] or keep_column[reverse_nicknames.get(flag, flag)]:
                if isinstance(value, tuple):
                    for tidx, tv in enumerate(value):
                        cols_aggregate[f"{flag}:{tidx}"][tv] = True
                        rows_aggregate[exp.idx][f"{flag}:{tidx}"] = tv
                else:
                    cols_aggregate[flag][value] = True
                    rows_aggregate[exp.idx][flag] = value

        # Store kwargs group name if any
        if exp.kwargs_groups is not None:
            for flag, (value, _) in exp.kwargs_groups.items():
                if keep_column[flag]:  # group names cannot be nicknames so no need to check
                    cols_aggregate[flag][value] = True
                    rows_aggregate[exp.idx][flag] = value
                    is_flag_group[flag] = True

    # Remove flags which have identical values over all columns
    # except if they are explicity required via `columns`
    # And diplay the resulting "fixed config"
    columns_num_unique_values = {
        k: len(v)
        for k, v in cols_aggregate.items()
        if ((columns is not None and k in columns) or len(v) > 1)
    }
    if show_fixed:
        removed_keys = set(cols_aggregate) - set(columns_num_unique_values)
        if len(removed_keys) > 0:
            config_table = Table(
                box=RICH_CUSTOM_BOX,
                show_lines=True,
            )
            config_table.add_column("Flag (Fixed config)")
            config_table.add_column("Value")
            for k in sorted(removed_keys):
                v = next(iter(cols_aggregate[k].keys()))
                if grid.kwargs_nicknames is not None:
                    v = grid.kwargs_nicknames.get(v, v)
                config_table.add_row(f"[cyan]{k}[/cyan]", str(v))
            rich.print(config_table)

    # If columns are not specified by the user, we sort them from the
    # shortest to longest flag value for pretty-printing
    columns_keys = list(columns_num_unique_values.keys())
    if columns is None:
        columns_keys = sorted(
            columns_keys,
            key=lambda x: sum(len(str(row.get(x, ""))) for row in rows_aggregate.values()),
        )

    # Add colors because we like colors
    if not muted:
        for exp_idx in rows_aggregate:
            for flag, value in rows_aggregate[exp_idx].items():
                # do not add color if all experiments have a different values
                # because we don't like colors *that* much
                if columns_num_unique_values.get(flag, float("inf")) == len(rows_aggregate):
                    continue

                # if the kwarg belongs to a group, use the same color across the group
                if flag in grid._exps_dict[exp_idx].kwarg_to_group:
                    group_name, group_value = grid._exps_dict[exp_idx].kwarg_to_group[flag]
                    c = color_maps[group_name].setdefault(
                        group_value, sample(PRETTY_TABLE_COLORS, 1)[0]
                    )

                # if the kwargs is a nickname of a kwargs that belongs to a group
                elif (
                    flag in reverse_nicknames
                    and reverse_nicknames[flag] in grid._exps_dict[exp_idx].kwarg_to_group
                ):
                    group_name, group_value = grid._exps_dict[exp_idx].kwarg_to_group[
                        reverse_nicknames[flag]
                    ]
                    c = color_maps[group_name].setdefault(
                        group_value, sample(PRETTY_TABLE_COLORS, 1)[0]
                    )

                # base case: unique new color
                else:
                    c = color_maps[flag].setdefault(value, sample(PRETTY_TABLE_COLORS, 1)[0])
                rows_aggregate[exp_idx][flag] = f"[{c}]{rich.markup.escape(str(value))}[/{c}]"

    # Select metrics to display
    metrics_keys = []
    if metrics is not None:
        metrics_keys = list(parse_as_tuple(metrics))
    for k in grid._exps_metrics.keys():
        if k not in metrics_keys:
            metrics_keys.append(k)

    # Add Headers to the table
    id_headers = ["ID"]
    if grid.use_hash_in_dirnames:
        id_headers += ["Sig"]
    if include_job_info:
        id_headers += ["JobID", "Status", "Runtime"]
    cols_names = id_headers + [""] * (len(columns_keys) + len(metrics_keys))
    cols_names[len(id_headers) + (len(columns_keys) - 1) // 2] = "Config"
    if len(metrics_keys):
        cols_names[len(id_headers) + len(columns_keys) + (len(metrics_keys) - 1) // 2] = "Metrics"
    for c in cols_names:
        table.add_column(c, no_wrap=False, max_width=20, overflow="fold")
    if len(metrics_keys) > 1:
        table.add_column("")

    table.add_row(
        *([""] * len(id_headers)),
        *[f"🔗 {flag}" if is_flag_group[flag] else flag for flag in columns_keys],
        *[f"📊 {metric_name}" for metric_name in metrics_keys],
        *([] if len(metrics_keys) <= 1 else ["📊 avg."]),
        style="bold grey100 on grey19",
    )

    # Add rows for each experiment
    # sort by ID
    sort_fn: Callable[[int], int | float] = lambda exp_idx: -exp_idx  # type: ignore

    # sort by metric
    if sort and len(metrics_keys) > 0:
        sort_on_metric = (
            metrics_keys[0] if (isinstance(sort, bool) or sort not in metrics_keys) else sort
        )

        def sort_fn(
            exp_idx: int,
        ) -> int | float:
            x = grid._exps_metrics[sort_on_metric].get(exp_idx, DEFAULT_METRIC_VALUE)
            if x == DEFAULT_METRIC_VALUE or isinstance(x, str):
                return float("inf") if ascending else -float("inf")
            return x

    # Formatting
    def __formatter__(x: Any, bolded: bool = False) -> str:
        if isinstance(x, int):
            if bolded:
                return f"[bold]{x}[/bold]"
            return str(x)

        if isinstance(x, float):
            if bolded:
                return f"[bold]{float_fmt.format(x)}[/bold]"
            return float_fmt.format(x)

        return str(x)

    # Time to fill this table !
    for exp_idx in sorted(rows_aggregate, key=sort_fn, reverse=not ascending):
        exp = grid._exps_dict[exp_idx]
        avg_metric = mean_non_default(
            [
                grid._exps_metrics[metric].get(exp_idx, DEFAULT_METRIC_VALUE)
                for metric in metrics_keys
            ]
        )
        job = exp.job
        table.add_row(
            *(
                [f"{exp_idx:02d}"]
                + ([exp.hashed_name] if grid.use_hash_in_dirnames else [])
                + (
                    [
                        job.id,
                        job.status.formatted_str,
                        job.get_elapsed_time_str(formatted=False, extra_verbose=False),
                    ]
                    if include_job_info
                    else []
                )
            ),
            *[
                __formatter__(
                    rows_aggregate[exp_idx].get(flag, "[bright_black]not found[/bright_black]")
                )
                for flag in columns_keys
            ],
            *[
                __formatter__(
                    grid._exps_metrics[metric].get(exp_idx, DEFAULT_METRIC_VALUE),
                    bolded=True,
                )
                for metric in metrics_keys
            ],
            *([] if len(metrics_keys) <= 1 else [__formatter__(avg_metric)]),
        )
    if return_table:
        return table, color_maps

    rich.print(table)
    if export_to is not None:
        # if no metrics, just export directly the config table
        if len(metrics_keys) == 0:
            export_table(
                table, export_to=export_to, ascending=ascending, keep_first_col_as_comment=False
            )
        # if metrics, we convert all the config in one big cell
        else:
            table = Table(
                caption=(
                    f"Experiment Grid for `{grid.base_name}`"
                    f" ([bold]{len(grid)}[/bold] xps) "
                    f"- last update: {readable_timestamp(t=None)}\n"
                    f"({grid.log_dir_root})"
                ),
                box=RICH_CUSTOM_BOX,
                show_lines=True,
            )

            id_headers = ["Hash", "Config"]
            for c in id_headers + list(metrics_keys):
                table.add_column(c)
            if len(metrics_keys) > 1:
                table.add_column("avg.")

            for exp_idx in sorted(rows_aggregate, key=sort_fn, reverse=not ascending):
                exp = grid._exps_dict[exp_idx]
                avg_metric = mean_non_default(
                    [
                        grid._exps_metrics[metric].get(exp_idx, DEFAULT_METRIC_VALUE)
                        for metric in metrics_keys
                    ]
                )
                name = f"{exp_idx:02d} - {exp.hashed_name}"
                config = "_".join(
                    f"{flag}={rows_aggregate[exp_idx].get(flag, 'n/a')}" for flag in columns_keys
                )
                table.add_row(
                    name,
                    config,
                    *[
                        __formatter__(
                            grid._exps_metrics[metric].get(exp_idx, DEFAULT_METRIC_VALUE),
                            bolded=True,
                        )
                        for metric in metrics_keys
                    ],
                    *([] if len(metrics_keys) <= 1 else [__formatter__(avg_metric)]),
                )
            export_table(
                table, export_to=export_to, ascending=ascending, keep_first_col_as_comment=True
            )
    return None


def add_rich_table_multi_header(
    rich_table: Table,
    *iterables: list,
    names: tuple[str, ...] | None = None,
    style: str = "bold white",
    rows_names: tuple[str, ...] | None = None,
) -> int:
    """Add multi row/columns header to a rich table

    :param rich_table: Target rich table
    :param iterables: list of possible values for each flag we want to display as columns
    :param names: (optional) Names of each column. If given, there should be as
        many names as `iterables`
    :param style: Rich style for the heads
    :param row_names: Names of the rows indexing (for multi-rows).
        if None, assumes we have a single row index.
    """
    if names is not None:
        assert len(names) == len(iterables)
    if rows_names is None or len(rows_names) == 0:
        rows_names = ("",)
    assert rows_names is not None

    # total number of columns after doing the product of all iterables
    macro_repeats = 1
    total_length = reduce(operator.mul, [len(x) for x in iterables])

    for idx, iterable in enumerate(iterables):
        if len(iterable) == 0:
            continue

        # prefix = the first leading columns
        # they will later corresponds to values
        # for the different rows
        prefix = [""] * len(rows_names)
        # add name of the column in last position
        if names is not None:
            prefix[-1] = names[idx]
        # for the last row in the header, we should also
        # add the row names
        if idx == len(iterables) - 1:
            row_name: str = ""
            for x, row_name in enumerate(rows_names):
                prefix[x] = row_name
            if names is not None:
                prefix[-1] = f"{row_name} / {names[idx]}"

        # Add column header for idx
        micro_repeats = total_length // (macro_repeats * len(iterable))
        row = prefix + [
            f"{value}"
            for _ in range(macro_repeats)
            for value in iterable
            for _ in range(micro_repeats)
        ]
        if idx == 0:
            for iv, v in enumerate(row):
                rich_table.add_column(
                    v,
                    header_style=style,
                    justify="right" if iv <= len(prefix) else "center",
                )
        else:
            rich_table.add_row(
                *row,
                end_section=True,
                style=style,
            )
        macro_repeats *= len(iterable)
    return total_length + len(iterables)


def get_syntax_highlighting_boundaries(table: dict[Any, Any], agg_op: Callable) -> list[float]:
    """Returns interval boundaries for syntax highlighting colors

    :param table: A metric dictionary, typically mapping an experiment index to its metric value
    :param agg_op: The operation we want to compute boundaries for
    """
    vals = [y for x in table.values() for y in [agg_op(x)] if isinstance(y, (float, int))]
    # no numerical values found
    if len(vals) == 0:
        return [-float("inf"), float("inf")]

    # otherwise, build the boundaries for higlighting
    mi = min(vals)
    ma = max(vals)
    intervals = [mi + (ma - mi) * k / len(HIGHLIGHT_COLORS) for k in range(len(HIGHLIGHT_COLORS))]
    return intervals


def table_agglomerate(
    grid: "ExperimentGrid",  # type: ignore   # noqa: F821
    rows: str | tuple[str, ...],
    columns: str | tuple[str, ...] | None = None,
    exclude: str | tuple[str, ...] | None = None,
    metrics: str | tuple[str, ...] | None = None,
    ascending: bool = False,
    muted: bool = False,
    float_fmt: str = "{:.3f}",
    agg: Literal["mean", "max", "min"] = "mean",
    werr: bool = False,
    show_len: bool = False,
    err_fmt: str = "{:.2e}",
    return_table: bool = False,
    show_fixed: bool = False,
    show_agglo: bool = True,
    export_to: Literal["markdown", "latex"] | None = None,
) -> Table | None:
    """Pretty print a list of the experiments in table form where rows and columns are a
    subset of hyperparameters and the content of cells is a metric

    :param rows: Which kwargs to display as rows.
    :param columns: Which kwargs to display as columns. Defaults to all columns - rows
    :param exclude: "exclusive" counterpart of `--columns`
    :param metric: Which metric to agglomerate / display in cells. If not given, will default
        to the first recorded metric
    :param ascending: Inverse the sorting order for the metric highlighting
    :param muted: If given, do not display the highlighting by metric
    :param float_fmt: Format for float numbers
    :param agg: Agglomerating function (average, maximum or minimum)
    :param werr: If True, also displays the std of agglomerated values
    :param show_len: If True, also displays the number of *valid* data points
        we are agglomerating on as a separate table
    :param err_fmt: Formatting for added errors
    """
    assert len(grid._exps_metrics), "The given experiment grid has no recorded metrics"
    if metrics is None:
        metrics_keys = tuple(grid._exps_metrics.keys())
    else:
        metrics_keys = parse_as_tuple(metrics)
        for k in metrics_keys:
            if k not in grid._exps_metrics:
                print_warning(f"Metric `{k}` not found in the given experiment grid")
    assert len(metrics_keys) >= 1, "No valid metrics selection given to `table`"

    if return_table:
        if len(metrics_keys) > 1:
            print_error("Live view of the table is only enabled with a single metric")
            sys.exit(1)

    # Determine which rows/columns to display
    keep_rows = parse_as_tuple(rows)
    parsed_columns: tuple[str, ...] | None = None

    all_columns = set(flag for exp in grid for flag in exp.kwargs)
    if columns is None:
        flt_columns = all_columns
    else:
        parsed_columns = parse_as_tuple(columns)
        flt_columns = set(parsed_columns)

    for row in keep_rows:
        flt_columns.discard(row)
    if exclude is not None:
        flt_columns = flt_columns.difference(parse_as_tuple(exclude))

    keep_columns = tuple(flt_columns)

    if parsed_columns is not None:
        keep_columns = tuple(sorted(keep_columns, key=parsed_columns.index))

    # Aggregate: metric -> (row flag, col flag) -> accumulated value
    aggregate: dict[str, dict[Any, Any]] = defaultdict(lambda: defaultdict(lambda: []))

    seen_values = defaultdict(lambda: set())
    for exp in grid:
        row_index = []
        row_index = tuple(
            exp.get_value_from_anyquery(row)
            if ":" not in row
            else exp.get_value_from_anyquery(row.split(":")[0])[int(row.split(":")[1])]
            for row in keep_rows
        )
        for flag, value in zip(keep_rows, row_index):
            seen_values[flag].add(value)

        col_index = tuple(
            exp.get_value_from_anyquery(col)
            if ":" not in col
            else exp.get_value_from_anyquery(col.split(":")[0])[int(col.split(":")[1])]
            for col in keep_columns
        )
        for flag, value in zip(keep_columns, col_index):
            seen_values[flag].add(value)

        if all(x is None for x in col_index):
            print_error(f"Empty indexing found for selected columns: {columns}")
            sys.exit(1)

        for mkey in metrics_keys:
            try:
                aggregate[mkey][row_index + col_index].append(grid._exps_metrics[mkey][exp.idx])
            except (KeyError, IndexError):
                aggregate[mkey][row_index + col_index].append(DEFAULT_METRIC_VALUE)

    # If columns was not user-specified, we can omit all the fixed config
    non_unique_keys: dict[str, bool] | None = None
    if columns is None:
        non_unique_keys = {
            k: True
            for k, v in seen_values.items()
            if ((columns is not None and k in columns) or len(v) > 1)
        }
        if show_fixed:
            removed_keys = set(seen_values) - set(non_unique_keys)
            if len(removed_keys) > 0:
                config_table = Table(
                    box=RICH_CUSTOM_BOX,
                    show_lines=True,
                )
                config_table.add_column("Flag (Fixed config)")
                config_table.add_column("Value")
                for k in sorted(removed_keys):
                    v = next(iter(seen_values[k]))
                    if grid.kwargs_nicknames is not None:
                        v = grid.kwargs_nicknames.get(v, v)
                    config_table.add_row(f"[cyan]{k}[/cyan]", str(v))
                rich.print(config_table)
    # If columns were speficied we need to go through all remaining kwargs
    # to determine which ones we are aggregating vs which ones are singletons
    elif show_fixed or show_agglo:
        leftover = all_columns.difference(keep_columns + keep_rows).union(grid.common_keys)
        if exclude is not None:
            leftover = leftover.difference(parse_as_tuple(exclude))

        # remove kwargs that are in a group
        delete = []
        first_exp = next(iter(grid._exps_dict.values()))
        for k in keep_columns + keep_rows:
            if first_exp.kwargs_groups is not None and k in first_exp.kwargs_groups:
                delete.extend(first_exp.kwargs_groups[k][1])
        leftover = leftover.difference(delete)

        # replace nicknamed arguments by their full names
        delete, add = [], []
        if grid.kwargs_nicknames is not None:
            for k in leftover:
                if k in grid.kwargs_nicknames:
                    delete.append(k)
                    add.append(grid.kwargs_nicknames[k])
            leftover = leftover.difference(delete).union(add)

        fixed_config, agglo_config = {}, {}
        for flag in leftover:
            vs = grid.unique_values(flag)
            if len(vs) == 1:
                fixed_config[flag] = vs[0]
            elif len(vs) > 1:
                agglo_config[flag] = vs

        if len(fixed_config) > 0 and show_fixed:
            config_table = Table(
                box=RICH_CUSTOM_BOX,
                show_lines=True,
            )
            config_table.add_column("Flag (Fixed config)")
            config_table.add_column("Value")
            for k in sorted(fixed_config):
                v = fixed_config[k]
                if grid.kwargs_nicknames is not None:
                    v = grid.kwargs_nicknames.get(v, v)
                config_table.add_row(f"[cyan]{k}[/cyan]", str(v))
            rich.print(config_table)

        if len(agglo_config) > 0 and show_agglo:
            config_table = Table(
                box=RICH_CUSTOM_BOX,
                show_lines=True,
            )
            config_table.add_column(f"Flag (aggregating over, {agg})")
            config_table.add_column("Values")
            for k in sorted(agglo_config):
                vs = agglo_config[k]
                if grid.kwargs_nicknames is not None:
                    vs = [grid.kwargs_nicknames.get(v, v) for v in vs]
                config_table.add_row(f"[cyan]{k}[/cyan]", ", ".join(str(v) for v in vs))
            rich.print(config_table)

    # Get rows indices
    rows_headers = [(x, grid.unique_values(x)) for x in keep_rows]
    cols_headers = [(x, grid.unique_values(x)) for x in keep_columns]

    # sanity check
    print()
    for lst, header in [(rows_headers, "row"), (cols_headers, "colum")]:
        for i, (name, vals) in enumerate(lst):
            if len(vals) == 0:
                print_warning(
                    f"No values found for {header} [magenta]{name}[/magenta]."
                    " Did you maybe mistype it ?"
                )
                lst[i] = (name, [None])

    def __formatter__(x: Any) -> str:
        if isinstance(x, float):
            return float_fmt.format(x)
        return str(x)

    agg_op = (
        mean_non_default if agg == "mean" else max_non_default if agg == "max" else min_non_default
    )
    for mkey in metrics_keys:
        table = Table(
            caption=(
                f"[bold cyan]{mkey}[/bold cyan] metric for experiment grid `{grid.base_name}`"
                f" ({len(grid)} experiments)\n"
                f"[bright_black]last update: {readable_timestamp(t=None)}[/bright_black]"
            ),
            caption_style="white",
            box=RICH_CUSTOM_BOX,
            show_lines=True,
        )

        # Add headers
        if non_unique_keys is not None:
            add_rich_table_multi_header(
                table,
                *tuple(zip(*[x for x in cols_headers if x[0] in non_unique_keys]))[1],
                names=tuple(x for x in keep_columns if x in non_unique_keys),
                rows_names=tuple(x for x in keep_rows if x in non_unique_keys),
            )
        else:
            add_rich_table_multi_header(
                table,
                *tuple(zip(*cols_headers))[1],
                names=tuple(x for x in keep_columns),
                rows_names=tuple(x for x in keep_rows),
            )

        # prepare syntax highlighting
        intervals = get_syntax_highlighting_boundaries(aggregate[mkey], agg_op=agg_op)

        # Time to fill that table
        for row_indexing in product(*tuple(zip(*rows_headers))[1]):
            # beginning of the row = flags' value for row indexing
            row_content = [f"[bold white]{val}[/]" for val in row_indexing]
            # Then fill in the value of the metric for each colum
            for col_indexing in product(*tuple(zip(*cols_headers))[1]):
                try:
                    vals = aggregate[mkey][(*row_indexing, *col_indexing)]
                    x = agg_op(vals)
                    if isinstance(x, str):
                        color = "bright_black"
                        cell = x
                    else:
                        cell = __formatter__(x)
                        if werr and agg == "mean":
                            cell += " ± " + err_fmt.format(std_non_default(vals))
                        if show_len:
                            cell += f" ({sum(1 for x in vals if x != DEFAULT_METRIC_VALUE)})"
                        if not muted:
                            color = HIGHLIGHT_COLORS[
                                (-1 if ascending else 1)
                                * bisect.bisect_left(intervals, x, hi=len(intervals) - 1)
                                - (1 if ascending else 0)
                            ]
                            cell = f"[{color}]{cell}[/{color}]"
                    row_content.append(__formatter__(cell))
                except (KeyError, ValueError):
                    row_content.append(f"[bright_black]{DEFAULT_METRIC_VALUE}[/]")
            # Add section line every group of the first row flag
            table.add_row(*row_content)

        if return_table:
            return table

        rich.print(table)
        if export_to is not None:
            export_table(table, export_to=export_to, ascending=ascending)
