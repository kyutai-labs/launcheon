# pyright: reportInvalidStringEscapeSequence=false
"""Implements the base class for a group of experiments"""

import ast
import json
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from functools import reduce
from itertools import product
from math import ceil
from pathlib import Path
from random import randint, sample, seed
from typing import Any, Callable, Iterator, Literal, Sequence, Type

import fire
import rich
import rich.markup
from rich.live import Live
from rich.table import Table

from launcheon.experiment import Experiment, get_name_from_kwargs, is_expandable_arg
from launcheon.global_variables import (
    PRETTY_TABLE_COLORS,
    STRING_TO_FILTER_OP,
    DistLauncher,
)
from launcheon.job import JobStatus
from launcheon.parser_format import ParserFormat, ParserFormatType
from launcheon.parsing_utils import db_parser, json_parser, jsonl_parser
from launcheon.table_utils import table_agglomerate, table_per_exp
from launcheon.utils import (
    get_git_revision_hash,
    is_singleton_kwarg,
    parse_as_tuple,
    parse_range,
    print_error,
    print_file_content,
    print_warning,
    readable_range,
    readable_timestamp,
    resolve_duplicate_key_conflict,
)


class ExperimentGrid:
    """Base experiment manager class"""

    def __init__(
        self,
        _experiment_constructor: Type = Experiment,
        base_name: str = "",
        base_cmd: str = "python",
        log_dir: str = "",
        use_hash_in_dirnames: bool = False,
        # Resources and environments
        num_nodes: int = 1,
        num_gpus: int = 1,
        num_cpus_per_gpu: int = 8,
        # formatting options
        parser_format: str | ParserFormatType | ParserFormat = ParserFormatType.ARGPARSE,
        verify_at_submit: bool = False,
        # Other kwargs to build the experiment's command
        kwargs_nicknames: dict[str, str] | None = None,
        # If those ranges are given, ech experiment will be assigned a
        # random seed and a random port that can then be accessed by the Python or YAML API
        # A global random seed is also used to make this process reproducible
        global_seed: int = 42,
        set_exp_random_seed: bool = True,
        exp_seed_min: int = 0,
        exp_seed_max: int = 10000,
        set_exp_random_port: bool = True,
        exp_port_min: int = 8200,
        exp_port_max: int = 30000,
        # multinode
        dist_launcher: Literal[
            "auto", "deepspeed", "accelerate", "torchrun", "jax", "none"
        ] = "auto",
        # extra setup command to wrap the launchin command under
        # micromamba env
        micromamba_env: str | None = None,
        # track git code
        # if false: nothing happens
        # if true, copy the code from the given `git_branch` and `git_commit`
        # into the experiments directory and run the code from there
        # Note that this will supersede any `work_dir` option given to SlurmExperimentGrid
        git_sync: bool = False,
        git_repo: str | None = None,
        git_branch: str | None = None,
        git_commit: str | None = None,
        git_clone_depth: int = 3,
        git_setup: str | None = None,
        git_dirs_exclude: tuple[str, ...] | None = None,
        # in case you're very confident in your experiment grid and want to
        # save a tiny bit of time
        skip_sanity_checks: bool = False,
        locked: bool = False,
        expand_kwargs_in_name: bool = True,
        expand_groups_in_name: bool = True,
        include_common_args_in_name: bool = True,
        override_log_file: str | None = None,
        # Any extra arguments will be treated as parts of the sweep to build the experiments
        **kwargs: Any,
    ) -> None:
        """Basic experiment manager

        :param _experiment_constructor: Base constructor for experiments in the grid
        :param base_name: Prefix for all experiments names
        :param base_cmd: Base/Shared command for all experiments. Note that this base command
            can include launchone secret keys to be expanded by each experiment, therefore
            it might not be identical for all experiments
        :param log_dir: Base log dir root for all experiments. Each experiment will create
            its own subdirectory.
        :param use_hash_in_dirname: If True, use the hashed name of the experiment (instead
            of the long name) to name experiment directories
        :param num_nodes: Number of nodes to run on for each job
        :param num_gpus: Number of GPUs to allocate (per node) for each job
        :param num_cpus_per_gpu: Number of CPUs per GPU to schedule per job
        :param quote_strings: If True, will add extra quotes around the values of type string
        :param kwargs_formatting: Optional. How to format flag/values for the command line
        :param kwargs_formatting_separator: Optional. How to separate formatted flag/value pairs
        :param kwargs_nicknames: Dictionary mapping strings (nicknames for flags' names or
            values) to other strings (true names of the flag name/value expanded when creating
            the experiment's command)
        :param global_seed: Global seed mainly useful when setting random port and seed per experiment
        :param set_exp_random_seed: If True, allocates a random int seed to each experiment, which
            can be accessed via launcheon secret key `{{launcheon.exp.seed}}`
        :param exp_seed_min: Min range for sampling the experiment random seed
        :param exp_seed_max: Max range for sampling the experiment random seed
        :param set_exp_random_port: If True, allocates a random int port to each experiment, which
            can be accessed via launcheon secret key `{{launcheon.exp.port}}`
        :param exp_port_min: Min range for sampling the experiment random port
        :param exp_port_max: Max range for sampling the experiment random port
        :param micromamba_env: Optional micromamba environment name that the command will
            run run through
        :param git_sync: If True, the script will first clone the given `git_repo` (or uses
            the current repo if None) at the given `git_branch` (or current branch) and
            given `git_commit` in the epxeriment's work directory and will run the code
            from there. Note that this should overwrite any `work_dir` flag, e.g. in
            slurm_experiment_grid.
        :param git_repo: Git repo to clone; if None use the repo of the current directory
        :param git_branch: Git branch to clone; if None use the repo of the current directory
        :param git_commit: Git commit to clone; if None use the repo of the current directory
        :param git_clone_depth: By default, we only make a shallow clone of depth 3; otherwise
            if you want to pull from an older commit, you can manually specifier an older clone depth
        :param git_setup: Some extra optional commands which are run after the git clone step
        :param git_dirs_exclude: This will ignore the given dirs when running git pull
        :param skip_sanity_check: Advanced users only
        :param locked: Locked experiments cannot be removed from the command line
        """
        creation_time: int | float = time.perf_counter_ns()
        # Base experiment grid setup
        self.base_name = base_name
        self.base_cmd = base_cmd
        self._log_dir_root = log_dir
        self.locked = locked
        self.verify_at_submit = verify_at_submit
        if isinstance(parser_format, str):
            self.parser_format = ParserFormatType(parser_format).get_parser_format()
        elif isinstance(parser_format, ParserFormatType):
            self.parser_format = parser_format.get_parser_format()
        elif isinstance(parser_format, ParserFormat):
            self.parser_format = parser_format
        else:
            raise ValueError("Found parser format of unknown type", type(parser_format))

        self.git_sync = git_sync
        self.git_repo = git_repo
        self.git_branch = git_branch
        self.git_commit = git_commit
        self.git_setup = git_setup
        self.git_clone_depth = git_clone_depth
        self.git_dirs_exclude = git_dirs_exclude
        self.use_latest_git_commit = False
        if self.git_sync:
            git_info = get_git_revision_hash()
            if self.git_repo is None:
                self.git_repo = git_info["repo"]
            if self.git_branch is None:
                self.git_branch = git_info["branch"]
            if self.git_commit is None:
                self.use_latest_git_commit = True
                self.git_commit = git_info["commit"]

        # Resources
        self.num_gpus = num_gpus
        self.num_cpus_per_gpu = num_cpus_per_gpu
        self.num_nodes = num_nodes
        try:
            self.dist_launcher = DistLauncher(dist_launcher.lower())
        except ValueError as e:
            print("Unknown distributed launcher option", dist_launcher)
            raise e
        if self.dist_launcher == DistLauncher.AUTO:
            if "deepspeed" in self.base_cmd:
                self.dist_launcher = DistLauncher.DEEPSPEED
            elif "accelerate" in self.base_cmd:
                self.dist_launcher = DistLauncher.ACCELERATE
            elif "torchrun" in self.base_cmd:
                self.dist_launcher = DistLauncher.TORCHRUN
            else:
                self.dist_launcher = DistLauncher.JAX

        self.kwargs_nicknames = kwargs_nicknames
        self.use_hash_in_dirnames = use_hash_in_dirnames
        self.micromamba_env = micromamba_env

        # Each experiment will store its own idx anyway, however having a dict
        # structure is still useful to select experiments more efficiently
        # after we have filtered them
        self._exps_dict: dict[int, Experiment] = OrderedDict()

        # With the `table`` command, we can also display some metrics. `metrics`
        # are accumulated by chaining commands before calling `table`
        # New/custom metrics commands can be added via the Python API
        self._exps_metrics: dict[
            str, dict[int, int | float | str | Sequence[int | float | str]]
        ] = defaultdict(lambda: {})
        self._update_metrics: list[Callable] = []

        # Sanity checks
        if not skip_sanity_checks:
            # A.
            if self.kwargs_nicknames is not None:
                # No cyclic structure in nicknames
                for k, v in self.kwargs_nicknames.items():
                    if v in self.kwargs_nicknames:
                        print_error(
                            "kwargs_nicknames does not accept "
                            f"cyclic structures: Found {v} as both a key and a value"
                        )
                        sys.exit(1)

                # no double nicknames because it would be a headache
                # nicknames is expected to be bijective
                nicknames_targets = list(self.kwargs_nicknames.values())
                if len(set(nicknames_targets)) < len(nicknames_targets):
                    print_error("A flag name can only have at most single nickname")
                    sys.exit(1)

            # B. Remove duplicate values + check that we have no more
            #  than one level of groups
            for key, vals in kwargs.items():
                if isinstance(vals, dict):
                    expanded_dict: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
                        lambda: []
                    )
                    for subkey, subsweep in vals.items():
                        if not isinstance(subsweep, dict):
                            print_error(
                                f"Error for group {subkey}: Values inside "
                                "kwarg group should be a dictionary"
                            )
                            sys.exit(1)

                        # We do not allow for nested groups
                        if any(isinstance(x, dict) for x in subsweep.values()):
                            print_error(
                                "Bad specification for sub-config "
                                f"[cyan]{subkey}[/cyan] in kwargs group [magenta]{key}[/magenta]:"
                                f" Nested sweeps are not allowed"
                            )
                            sys.exit(1)

                        # [experimental] in case we have a sweep inside the group,
                        # we first expand it here to simplify the next sanity
                        # checks and preprocessing
                        cands = {k: v for k, v in subsweep.items() if isinstance(v, list)}
                        if len(cands) == 0:
                            continue

                        swept_keys, swept_values = zip(*cands.items())
                        # Instantiate all experiments
                        for bundle in product(*swept_values):
                            # When we create the experiment name at this level, we are using
                            # groups names, which makes for nicer/shorter name. Alternatively,
                            # we could use the default naming of the `Experiment` object but
                            # this would expand all the kwargs groups and can be wuite length
                            expanded_kwargs = dict(zip(swept_keys, bundle))
                            expanded_key = f"{subkey}_{get_name_from_kwargs(expanded_kwargs)}"
                            expanded_kwargs.update(
                                {k: v for k, v in subsweep.items() if k not in expanded_kwargs}
                            )
                            expanded_dict[subkey].append((expanded_key, expanded_kwargs))

                    # replace any existing expanded_dict
                    for subkey in expanded_dict:
                        del vals[subkey]
                        for added_group_keys, added_group_vals in expanded_dict[subkey]:
                            vals[added_group_keys] = added_group_vals

                # Remove duplicate values from sweeps
                if isinstance(vals, list):
                    kwargs[key] = list(sorted(set(vals), key=vals.index))

            # C. Sanity check - duplicates
            # There are two cases that can lead to duplicate keys/flags:
            remove_keys = []
            # a) Case A: A key is present as itself and as a kwargs nicknames
            # the tie-breaking rule is that we give precedence to a key
            # being swept over; if ties cannot be broken, we throw an error
            # b) A key is present at the top level and inside a group.
            # Here we directly throw an error because trying to resolve the
            # tie would be super annoying
            seen_keys: dict[str, tuple[int, str, bool, Any]] = {}
            for key_a, val_a in kwargs.items():
                # Case A
                if (
                    self.kwargs_nicknames is not None
                    and key_a in self.kwargs_nicknames
                    and (key_b := self.kwargs_nicknames[key_a]) in kwargs
                ):
                    remove_keys.append(
                        resolve_duplicate_key_conflict(key_a, key_b, val_a, kwargs[key_b])
                    )

                # Mark flags key to resolve case B
                # Note that we are also marking group names here. Technically they do
                # not conflict with anything as they do not appear as flags, but this
                # can lead to confusions and misleading naming. so we forbid it.
                seen_keys[self.get_real_key_name(key_a)] = (
                    1,
                    key_a,
                    is_singleton_kwarg(val_a),
                    val_a,
                )

            # Case B: Check kwargs inside group that may be conflicting
            # with kwargs or something else
            overriden_defaults = defaultdict(lambda: {})
            for key_a, val_a in kwargs.items():
                if isinstance(val_a, dict):
                    for group_name, subsweep in val_a.items():
                        for subk, subv in subsweep.items():
                            if (
                                retrieved := seen_keys.get(self.get_real_key_name(subk), None)
                            ) is not None:
                                is_from_kwargs, src_name, is_singleton, src_val = retrieved
                                # if not swept over, the group value takes precedence HOWEVER there is
                                # still an edge case when it is defined in a group but not the others, so
                                # in that case we assign the signleton value as default to the other groups
                                if is_from_kwargs and is_singleton:
                                    remove_keys.append(src_name)
                                    overriden_defaults[key_a][src_name] = src_val
                                    print_warning(
                                        f"Key [cyan]{subk}[/cyan]"
                                        f" from group [magenta]{group_name}[/magenta] ({subv}) "
                                        "takes precedence over singleton duplicate "
                                        f"[cyan]{src_name}[/cyan] from kwargs ({src_val})"
                                    )
                                else:
                                    print_error(
                                        f"Key [cyan]{subk}[/cyan]"
                                        f" from group [magenta]{group_name}[/magenta] "
                                        f"({subv}) "
                                        f"is duplicate: conflict with [cyan]{src_name}[/cyan]"
                                        f" ({src_val})"
                                    )
                                    sys.exit(1)
                    seen_keys.update(
                        {
                            self.get_real_key_name(subkey): (
                                0,
                                f"[magenta]{key_a}[/magenta]/[cyan]{subkey}[/cyan]",
                                False,
                                subsweep[subkey],
                            )
                            for subsweep in val_a.values()
                            for subkey in subsweep
                        }
                    )

            # remove keys
            if len(remove_keys) > 0:
                kwargs = {k: v for k, v in kwargs.items() if k not in remove_keys}

                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        # if dict doesn't define one of the ovrriden default  while
                        # its sibling group do, give it the default
                        for tgt_key in overriden_defaults[k].keys():
                            for group_name, subconfig in v.items():
                                if all(k3 != tgt_key for k3 in subconfig.keys()):
                                    subconfig[tgt_key] = overriden_defaults[k][tgt_key]

        # 1. Fixed arguments not part of the experiment names
        # Identify arguments which are common to all experiments (no sweep)
        # These are arguments of the shape `sweep[k] = v` such that v is
        # not iterable
        # Note that sweeps of length 1, e.g. `sweep[k] = [v]` are still considered
        # a sweep and will be used to build the unique experiment name
        self.common_args, dropped_args = {}, []
        for key, vals in kwargs.items():
            if isinstance(vals, (list, dict)):
                if len(vals) == 0:
                    dropped_args.append(key)
            else:
                dropped_args.append(key)
                self.common_args[key] = vals
        for key in dropped_args:
            del kwargs[key]
        # This includes all common keys, including the ones that might become
        # unique among experiment due to replacement of values by launcheon secret keys
        self.common_keys = set(self.common_args.keys())

        # Check for duplicate kwargs found in base_command
        pattern = self.parser_format.kwargs_formatting.format(
            flag=r"([a-zA-Z_0-9]+)",
            value=r"([^-]*)" + self.parser_format.kwargs_formatting_separator,
        )
        for flag, value in re.findall(pattern, self.base_cmd):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
            true_flag = self.get_real_key_name(flag)
            if true_flag in kwargs:
                print_warning(
                    f"Found duplicate key/nickname"
                    f" [cyan]{flag} | {true_flag}[/cyan] in base command and experiment sweep:\n"
                    f"  > {flag} ({kwargs[flag]}) will take precedence"
                )

        # 2. Launcheon secret keys
        # Forbid secret key in kwargs nicknames: sounds like a headache for little gains
        if self.kwargs_nicknames is not None:
            for key, value in self.kwargs_nicknames.items():
                if is_expandable_arg(value) != "no":
                    print_error("Launcheon secret keys are not supported inside `kwargs_nicknames`")
                    sys.exit(1)

        # expanded_kwargs: If there is any launcheon secret key to expand in
        # one of the common args, it becomes experiment-specific and will be
        # filled on-the-fly when creating the experiment
        expanded_kwargs: dict[str, Any] = {}
        assert expanded_kwargs is not None
        dropped_args = []

        for k, v in self.common_args.items():
            if is_expandable_arg(v) != "no":
                expanded_kwargs[k] = v
                dropped_args.append(k)

        for key in dropped_args:
            del self.common_args[key]

        # 3. Kwargs groups
        # Identify nested kwargs groups that will need to be expanded
        # when creating the experiments
        kwargs_groups_names = []
        for key, vals in kwargs.items():
            if isinstance(vals, dict):
                kwargs_groups_names.append(key)

        # Create the experiment grid
        seed(global_seed)
        if len(kwargs):
            keys, values = zip(
                *[
                    (k, kwargs[k])
                    for k in sorted(
                        kwargs.keys(),
                        key=lambda x: x if kwargs_nicknames is None else kwargs_nicknames.get(x, x),
                    )
                ]
            )
            num_experiments = reduce(operator.mul, [len(v) for v in values], 1)

            # Generate unique random seed/port for each experiemnt
            # can be used in the sweep via secret keys
            exp_random_seeds: Sequence[int] | None = None
            if set_exp_random_seed:
                exp_random_seeds = sample(range(exp_seed_min, exp_seed_max), num_experiments)

            exp_random_ports: Sequence[int] | None = None
            if set_exp_random_port:
                exp_random_ports = sample(range(exp_port_min, exp_port_max), num_experiments)

            # Instantiate all experiments
            for bundle in product(*values):
                # When we create the experiment name at this level, we are using
                # groups names, which makes for nicer/shorter name. Alternatively,
                # we could use the default naming of the `Experiment` object but
                # this would expand all the kwargs groups and can be wuite length
                exp_kwargs = dict(zip(keys, bundle))
                exp_name = None
                if not expand_groups_in_name:  # Legacy naming format (groups are not expanded)
                    exp_name = get_name_from_kwargs(exp_kwargs)

                # expand/flatten kwargs groups into proper keyword arguments
                exp_kwargs_groups = None
                if len(kwargs_groups_names) > 0:
                    exp_kwargs_groups = {
                        k: (exp_kwargs[k], list(kwargs[k][exp_kwargs[k]].keys()))
                        for k in kwargs_groups_names
                        if k in exp_kwargs
                    }

                    add_flags = {}
                    remove_flags = []
                    for k, v in exp_kwargs.items():
                        if k in kwargs_groups_names:
                            add_flags.update(kwargs[k][v])
                            remove_flags.append(k)
                    for k in remove_flags:
                        del exp_kwargs[k]
                    exp_kwargs.update(add_flags)

                # Add the experiment to the grid
                exp_idx = len(self._exps_dict)
                self._exps_dict[exp_idx] = _experiment_constructor(
                    # exp idx
                    idx=exp_idx,
                    exp_name=exp_name,
                    # exp specific
                    kwargs={**exp_kwargs, **expanded_kwargs},
                    expand_kwargs_in_name=expand_kwargs_in_name,
                    include_common_args_in_name=include_common_args_in_name,
                    # store group names for this exp if any, only used to filter
                    # based on kwargs group names
                    kwargs_groups=exp_kwargs_groups,
                    # common to all experiments
                    base_name=self.base_name,
                    command=self.base_cmd,
                    common_args=deepcopy(self.common_args),
                    log_dir_root=self.log_dir_root,
                    override_log_file=override_log_file,
                    use_hashed_dirname=use_hash_in_dirnames,
                    parser_format=self.parser_format,
                    random_seed=(
                        exp_random_seeds[exp_idx] if exp_random_seeds is not None else None
                    ),
                    random_port=(
                        exp_random_ports[exp_idx] if exp_random_ports is not None else None
                    ),
                    _kwargs_nicknames=deepcopy(self.kwargs_nicknames),
                )
        # Case where there is no sweep, only a single experiment in the grid
        else:
            self._exps_dict[0] = _experiment_constructor(
                idx=0,
                base_name=self.base_name,
                command=base_cmd,
                common_args=self.common_args,
                random_seed=randint(exp_seed_min, exp_seed_max),
                random_port=randint(exp_port_min, exp_port_max),
                kwargs=expanded_kwargs,
                expand_kwargs_in_name=expand_kwargs_in_name,
                include_common_args_in_name=include_common_args_in_name,
                log_dir_root=self.log_dir_root,
                override_log_file=override_log_file,
                use_hashed_dirname=use_hash_in_dirnames,
                parser_format=self.parser_format,
                _kwargs_nicknames=deepcopy(self.kwargs_nicknames),
            )
        creation_time = (time.perf_counter_ns() - creation_time) / 1e6

        num_forced_hash = sum(exp.forced_hashed_dirname for exp in self._exps_dict.values())
        if num_forced_hash > 0:
            print_warning(
                f"{num_forced_hash} experiment name(s) contain a path separator or are too long"
                " to be used as directory names. Using hashed directory names for these"
                " experiments instead (set `use_hash_in_dirnames=True` to silence this warning)"
            )

        if not skip_sanity_checks:
            rich.print(
                f"[cyan]\\[launcheon] Succesfully instantiated experiment grid "
                f"{self.base_name} in {creation_time:.2e}ms 🐬✨[/cyan]"
            )

    def check(self) -> None:
        """For debugging - Check that initialization went well"""
        rich.print(f"[green]Succesfully instantiated experiment grid {self.base_name} 🐬✨[/green]")
        self.count_res()

    def fire(self) -> None:
        """Starts a Fire cli for this grid, which makes it accessible through the command line"""
        fire.Fire(self)

    def __update_jobs__(self, notset_only: bool = False) -> None:
        """Update experiment's job statuses (in particular, status and start time).
        This can be overriden for performane reason (e.g. it is faster to do
        a batch command to retrieve multiple jobs' statuses at once)

        :param notset_only: Whether to refresh all jobs (False) or only the ones which
            have not been set through previous commands (True).
            Setting this to True is generally faster. Typically, we set it to False
            for live update commands (`table`, `monitor`) or commands where we want to
            have the most up-to-date info about job status (`submit`)
        """
        for exp in self:
            if not notset_only or exp._job is None:
                exp.update_job()

    def update_jobs(self, notset_only: bool = False, verbose: bool = True) -> None:
        """Wrapper around __update_jobs__ with extra perf info printing"""
        creation_time: float | int = time.perf_counter_ns()
        self.__update_jobs__(notset_only=notset_only)
        if verbose:
            creation_time = (time.perf_counter_ns() - creation_time) / 1e6
            rich.print(
                "[bright_black]\\[perf] updated jobs' status for grid "
                f"{self.base_name} in {creation_time:.2e}ms[/bright_black]",
                flush=True,
            )

    def register_exp_info(self, exp: Experiment, title: str) -> None:
        """Register exp info and override git info if git_sync is active"""
        exp.register_exp_info(
            title,
            git_sync=self.git_sync,
            git_repo=(self.git_repo if self.git_sync else None) or "",
            git_branch=(self.git_branch if self.git_sync else None) or "",
            git_commit=(self.git_commit if self.git_sync else None) or "",
        )

    def verify(self):
        """Verify command of experiments"""
        rich.print("[yellow]Verifying experiments' commands before submission...[/yellow]")
        for idx, exp in enumerate(self):
            cmd = self.get_full_exp_cmd(exp, local=True)
            out = self.parser_format.dry_run(cmd)
            if out is not None:
                print()
                print(out)
                print()
                print_error(
                    f"\nInvalid command for experiment [yellow]{exp.idx} - {exp.hashed_name}[/yellow]:"
                )
                rich.print(f"[bright_black][{cmd}[/bright_black]")
                rich.print("  > See error trace above")
                sys.exit(1)
            else:
                print(f"\r[{idx:03d}/{len(self):03d}]", end="")
        rich.print("\n[green]All experiments' commands are valid![/green]")

    @abstractmethod
    def __submit_exps__(
        self,
        *_args: Any,
        **_mismatched_flags: Any,
    ) -> int | None:
        """[Need override] Bare-bone skeleton for a submission script.
        See `slurm_experiment_grid` for a proper example implementation.

        In particular, you should not forget `exp.register_exp_info(str(jobid))`
        """
        # Strict check to avoid submitting experiment by mistake !
        if len(_args) > 0 or len(_mismatched_flags):
            print_error(
                "Found unexpected flags to the `submit` command: , ".join(_args) + ","
                if len(_args) > 0
                else ", ".join(_mismatched_flags.keys())
            )
        raise NotImplementedError

    def submit(
        self,
        *args: Any,
        dry_run: bool = False,
        watch: bool = False,
        no_verify: bool = False,
        **kwargs: Any,
    ) -> None:
        """Submit command wrapper such that the output from
        __submit_exps__ is not printed on stdout"""
        if dry_run:
            rich.print(f"[bold cyan]{'-' * 50} DRY RUN - Start {'-' * 50}[/bold cyan]")
        else:
            self.check_git()
        if self.verify_at_submit and not no_verify:
            self.verify()
        num_launched = self.__submit_exps__(*args, dry_run=dry_run, **kwargs)
        if not dry_run:
            rich.print(f"Submitted [bold green]{num_launched or 0}[bold green] experiments")
            print()
        else:
            rich.print(f"[bold cyan]{'-' * 50} DRY RUN - End {'-' * 50}[/bold cyan]")
        if watch:
            self.monitor()

    @property
    def log_dir_root(self) -> str:
        """Root log directory"""
        return self._log_dir_root

    @log_dir_root.setter
    def log_dir_root(self, value: str) -> None:
        """Set root log directory and propagate to all experiments"""
        self._log_dir_root = value
        for exp in self:
            exp.log_dir_root = value

    def __iter__(self) -> Iterator[Experiment]:
        """Make the class iterable"""
        yield from self._exps_dict.values()

    def __len__(self) -> int:
        """Number of experiments in the iterable"""
        return len(self._exps_dict)

    def __getitem__(self, idx: int) -> Experiment:
        """Only overwriting this to get proper type hints for `SlurmExperiemnt`"""
        return self._exps_dict[idx]  # type: ignore

    def __readable_range__(self) -> str:
        """Print out the indices in the current exps dict as a readable range"""
        return readable_range(list(self._exps_dict.keys()))

    def get_real_key_name(self, key: str) -> str:
        """Return the true value of the given key/flag, taking into account
        any potential nicknames"""
        if self.kwargs_nicknames is not None:
            return self.kwargs_nicknames.get(key, key)
        return key

    def get_git_code_dir(self) -> str:
        return os.path.abspath(os.path.join(self.log_dir_root, "code"))

    def check_git(self) -> None:
        """Check whether the current repo contains modified files"""
        if self.git_sync and self.use_latest_git_commit:
            # Also check if there are any modified file
            changed = (
                subprocess.check_output("git diff-index --name-only HEAD", shell=True)
                .decode("utf-8")
                .splitlines()
            )
            if len(changed) > 0:
                print()
                print_warning(
                    "You have [red]uncommited[/red] modified files in your current directory:"
                )
                print("\n".join(f"  - {c}" for c in changed))
                print()
                rich.print("[yellow]Continue ?[/yellow] \\[[bold]Y[/bold]/n]")
                rich.print(
                    "[bright_black]The code will run from commit[/bright_black]"
                    f" [cyan]{self.git_commit}[/cyan]"
                    "[bright_black] and ignore your uncommited changes[/bright_black]"
                )
                resp = input()
                if resp in {"n", "N"}:
                    rich.print("[cyan]Safe choice 👍. Exiting[/cyan]")
                    sys.exit()

    def setup_git_clone(self):
        """Clone the git repo in the experiment's directory.
        Note that this step should be ran in the submit_exps function,
        on the pod to avoid shenanigans with the"""
        create_repo = True
        if self.git_sync:
            assert (
                self.git_repo is not None
                and self.git_branch is not None
                and self.git_commit is not None
            )
            rich.print("Preparing git-synced environment...")
            # if code dir already exists, we check the current branch, commit and repo match
            code_dir = self.get_git_code_dir()
            if os.path.exists(code_dir):
                try:
                    old_git_repo, old_git_commit, old_git_branch = (
                        subprocess.check_output(
                            f"cd {shlex.quote(code_dir)}; git config --get remote.origin.url; "
                            f"git rev-parse HEAD; git log -n 1 --pretty=%d HEAD",
                            shell=True,
                            stderr=subprocess.DEVNULL,
                        )
                        .decode("utf-8")
                        .splitlines()
                    )
                    old_git_repo = old_git_repo.strip()
                    # if either the old or new git repo are local path, we will skip the repo check
                    # TODO: Not sure if this the strictest check tbh
                    check_repo_equality = True
                    if not ("git@" in old_git_repo and "git@" in self.git_repo):
                        check_repo_equality = False
                    old_git_commit = old_git_commit.strip()
                    old_git_branch = (
                        old_git_branch.rsplit(",", 1)[-1]
                        .replace(")", "")
                        .strip()
                        .replace("origin/", "")
                    )
                    if "HEAD" in old_git_branch:
                        old_git_branch = subprocess.check_output(
                            f"cd {shlex.quote(code_dir)}; git show-ref",
                            shell=True,
                            stderr=subprocess.DEVNULL,
                        ).decode("utf-8")
                        old_git_branch = old_git_branch.split("origin/")[-1].strip()
                    bools = [
                        (
                            check_repo_equality and old_git_repo != self.git_repo,
                            "repository",
                            old_git_repo,
                            self.git_repo,
                        ),
                        (
                            old_git_branch != self.git_branch,
                            "branch",
                            old_git_branch,
                            self.git_branch,
                        ),
                        (
                            old_git_commit != self.git_commit,
                            "commit",
                            old_git_commit,
                            self.git_commit,
                        ),
                    ]
                    # If repo already match our requested one, we can exit already
                    if all(not x[0] for x in bools):
                        create_repo = False
                    # otherwise, ask for user input if we shuld keep or update the repo
                    else:
                        print()
                        print_warning(
                            "A different git-synced code directory already exists for this experiment grid:"
                        )
                        for x, name, old, new in bools:
                            emo = "❌" if x else "✅"
                            sep = "[red3]!=[/red3]" if x else "[turquoise2]==[/turquoise2]"
                            suffix = ""
                            if name == "commit":
                                suffix = (
                                    subprocess.check_output(
                                        f"cd {shlex.quote(code_dir)};"
                                        f" git show --no-patch --format=%ci {shlex.quote(old)}",
                                        shell=True,
                                    )
                                    .decode("utf-8")
                                    .replace("\n", "")
                                )
                                suffix = f" at {suffix})"
                            rich.print(
                                f"  {emo} Requested {name} [bright_black]{new}[/bright_black]"
                                f" {sep} [bright_black]{old}{suffix}[/bright_black]"
                            )
                        rich.print(
                            "\n[bold cyan]Choose:[/bold cyan] [yellow]Overwrite old dir (y), continue with old code dir (c) or abort (n) ?[/yellow]"
                            " \\[y/c/[bold]N[/bold]]"
                        )
                        resp = input()
                        if resp in {"y", "Y", "yes", "YES"}:
                            # we need to remove the directory: as we did a shallow clone, it's not
                            # so easy to switch branch
                            rich.print("[gray]Removing code dir...[/gray]")
                            shutil.rmtree(code_dir)
                        elif resp in {"c", "C", "continue", "CONTINUE"}:
                            rich.print(
                                f"[gray]Will run from previous git-synced dir {code_dir} at:[/gray]"
                            )
                            rich.print(
                                "\n".join(
                                    f"[cyan]git {name}[/cyan]: {old}" for _, name, old, _ in bools
                                )
                            )
                            create_repo = False
                        else:
                            rich.print("[cyan]Safe choice 👍. Exiting[/cyan]")
                            sys.exit()
                except subprocess.CalledProcessError:
                    rich.print(f"\n[gray]{code_dir}[/gray] is not a valid git repository")
                    rich.print("[yellow]Continue ?[/yellow] \\[y/[bold]N[/bold]]")
                    rich.print(
                        "[bright_black]This will remove the code directory under[/bright_black]"
                        f" [gray]{code_dir}[/gray]"
                    )
                    resp = input()
                    if resp not in {"y", "Y", "yes", "YES"}:
                        rich.print("[cyan]Safe choice 👍. Exiting[/cyan]")
                        sys.exit()
                    else:
                        rich.print("[gray]Removing code dir...[/gray]")
                        _code_dir = Path(code_dir)
                        _staging = Path(tempfile.mkdtemp(dir=_code_dir.parent)) / _code_dir.name
                        _code_dir.rename(_staging)
                        threading.Thread(
                            target=shutil.rmtree, args=(_staging.parent,), daemon=True
                        ).start()

            # Clone repo
            if create_repo:
                # Check if git branch exists on remote repo and contains the commit.
                # Note: `git ls-remote` only matches ref names, not commit SHAs, so we
                # look up the remote branch head then check the commit is one of its ancestors
                ref_exists = False
                try:
                    remote_head = subprocess.check_output(
                        [
                            "git",
                            "ls-remote",
                            "--exit-code",
                            "--heads",
                            self.git_repo,
                            self.git_branch,
                        ],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).split()[0]
                    subprocess.check_call(
                        ["git", "merge-base", "--is-ancestor", self.git_commit, remote_head],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    ref_exists = True
                except (subprocess.CalledProcessError, IndexError):
                    pass

                # If ref does'mt exist, we wil clone from local repo
                if not ref_exists:
                    rich.print(
                        f"[yellow]WARN[/yellow]: Commit not found on remote repo.\n"
                        f"Using local repo instead [cyan]{(cwd := Path.cwd())}[/cyan]"
                    )
                    self.git_repo = cwd.as_posix()

                q_code_dir = shlex.quote(code_dir)
                q_repo = shlex.quote(self.git_repo)
                q_branch = shlex.quote(self.git_branch)
                q_commit = shlex.quote(self.git_commit)
                clone_msg = shlex.quote(
                    f"Cloning repo {self.git_repo} at {self.git_branch}:{self.git_commit}..."
                )
                # Checkout the requested commit; if it is not part of the shallow clone,
                # fetch the full history. Any failure aborts the git sync
                checkout_cmd = (
                    f"{{ git -c advice.detachedHead=false checkout --quiet {q_commit} 2>/dev/null"
                    " || { git fetch --quiet --unshallow origin"
                    f" && git -c advice.detachedHead=false checkout --quiet {q_commit}; }}"
                    ' || { echo "Error: could not checkout the requested git commit" >&2; exit 1; }; }; '
                )
                if self.git_dirs_exclude is None:
                    cmd = (
                        f"echo {clone_msg}; "
                        f"git clone --quiet --depth={self.git_clone_depth} --single-branch"
                        f" --branch {q_branch} {q_repo} {q_code_dir}"
                        ' || { echo "Warning: Reusing git clone found in experiment dir" >&2; }; '
                        f'cd {q_code_dir} || {{ echo "Error: git clone command failed:" >&2; exit 1; }}; '
                        + checkout_cmd
                    )
                else:
                    sparse_git_config = 'echo "/*" >> .git/info/sparse-checkout; '
                    sparse_git_config += "; ".join(
                        f"echo {shlex.quote('!' + d)} >> .git/info/sparse-checkout"
                        for d in self.git_dirs_exclude
                    )
                    fetch_refspec = shlex.quote(
                        f"+refs/heads/{self.git_branch}:refs/remotes/origin/{self.git_branch}"
                    )
                    cmd = (
                        f"echo {clone_msg}; "
                        f'mkdir -p {q_code_dir} || {{ echo "Error: mkdir command failed:" >&2; exit 1; }};'
                        f' cd {q_code_dir} || {{ echo "Error: cd command failed:" >&2; exit 1; }};'
                        " git init --quiet;"
                        f"git remote add origin {q_repo}; "
                        f"git config core.sparseCheckout true; {sparse_git_config}; "
                        f"git config remote.origin.fetch {fetch_refspec}; "
                        f"git fetch --quiet --depth={self.git_clone_depth} origin {q_branch}"
                        ' || { echo "Error: git fetch command failed" >&2; exit 1; }; '
                        + checkout_cmd
                    )
            else:
                cmd = f"cd {shlex.quote(code_dir)}; "

            if self.git_setup is not None:
                cmd += self.git_setup.strip()
                if not cmd.endswith(";"):
                    cmd += "; "
            out = subprocess.call(cmd, shell=True)
            if out != 0:
                print_error(f"Git synchronization exited with an error (Exit code {out})")
                sys.exit(int(out))

    def setup_git_env(self) -> str:
        """Assumes setup_git_clone(exp) has already been run"""
        if self.git_sync:
            code_dir = self.get_git_code_dir()
            # could prepend setup_git_clone(exp) here if we wanted to run the git clone
            # steps directly on the GPU machines
            return f"cd {shlex.quote(code_dir)}; "
        return ""

    def get_full_exp_cmd(self, exp: Experiment, local: bool = False) -> str:
        """Shortcut to get the full experiment command including micromamba env"""
        return self.wrap_exp_cmd(exp.get_cmd(), local=local)

    def wrap_exp_cmd(self, s: str, local: bool = False) -> str:
        """Wrap the given experiment command with the micromamba env and git sync setup"""
        if self.micromamba_env is not None:
            s = f"micromamba run -n {shlex.quote(self.micromamba_env)} bash -c {shlex.quote(s)}"

        if self.git_sync and not local:
            s = f"{self.setup_git_env()}{s}"

        if is_expandable_arg(s) != "no":
            matches = re.findall(r"\{\{launcheon\.exp\.([\w.]+)\}\}", s)
            keys = ", ".join(f"[bold]{rich.markup.escape(str(m))}[/bold]" for m in matches)
            print_error(f"The generated command contains non-expanded secret keys: {keys}\n ")
        return s

    def unique_values(self, kwarg: str) -> list[Any]:
        """Get the list of unique values across experiments for the given kwarg

        :param kwarg: Name of the `kwarg` | nickname | group name etc.

        :return: A sorted value of all unique values taken by this kwarg
            through the current experiment grid
        """
        values = set()
        for exp in self:
            if ":" in kwarg:
                name, pos = kwarg.rsplit(":", 1)
                v = exp.get_value_from_anyquery(name)
                try:
                    v = v[int(pos)]
                except (ValueError, TypeError):
                    v = None
            else:
                v = exp.get_value_from_anyquery(kwarg)
            if v is not None:
                values.add(v)

        try:
            max_int = max(x for x in values if isinstance(x, (int, float)))
        except ValueError:
            max_int = 0

        def __sort__(x: Any) -> tuple[int | float, ...]:
            if isinstance(x, (int, float)):
                return (x,)
            return tuple(max_int + ord(c) for c in str(x))

        return sorted(values, key=__sort__)

    def sweep_kwargs(self) -> dict[str, list[Any]]:
        """Returns the list of kwargs that actually change across experiments"""
        candidates = set(x for exp in self for x in exp.kwargs)
        uniques = {}
        for c in candidates:
            vs = self.unique_values(c)
            if len(vs) > 1:
                uniques[c] = vs
        return uniques

    def filter(
        self,
        kwarg_name: str,
        kwarg_op: Literal["=", "!=", "<", "<=", ">", ">=", "in", "notin"] | None = None,
        kwarg_value: Any = None,
    ) -> "ExperimentGrid":
        """(Chainable) Filter experiments based on a kwarg value"""
        if kwarg_op is None:
            # try longer operators first, so that e.g. `!=` is not parsed as `=`
            for op in sorted(STRING_TO_FILTER_OP, key=len, reverse=True):
                aux = kwarg_name.split(op)
                if len(aux) == 2:
                    kwarg_name = aux[0]
                    kwarg_op = op  # type: ignore
                    kwarg_value = aux[1]
                    break

        if kwarg_op not in STRING_TO_FILTER_OP and kwarg_op not in {"in", "notin"}:
            print_error(f"Found unexpected operation in filter: {kwarg_op}")
            sys.exit(1)
        if kwarg_value is None:
            print_error(f"Found no target value in filtering op: {kwarg_name} {kwarg_op} ???")
            sys.exit(1)

        self._exps_dict = {
            k: exp
            for k, exp in self._exps_dict.items()
            if exp.filter(kwarg_name, kwarg_op, kwarg_value)
        }

        if len(self._exps_dict) == 0:
            print_error(f"Selection '{kwarg_name} {kwarg_op} {kwarg_value}' is empty !")
            sys.exit(1)
        return self

    def select(
        self,
        selection: (
            tuple[int, ...]
            | Literal[
                "done",
                "failed",
                "running",
                "pending",
                "unscheduled",
                "cancelled",
                "active",
                "error",
                "todo",
            ]
        ),
    ) -> "ExperimentGrid":
        """(Chainable) Selects a subset of experiments in the grid; Different formants
        are available, e.g.:

        > select 42 - Select the 42-th experiment
        > select 0, 2, 5  - Select experiment 0, 2 and 5
        > select 0,5-10:2 - Select experiments 0, 5, 7 and 9 (note that all
            bounds are inclusive)
        > select 10-:2 - Select experiments 10, 12, 14 ... etc until the end
            of the list of experiments
        """
        # special keywords
        if isinstance(selection, str) and (
            selection in [x.name.lower() for x in JobStatus]
            or selection in [x.name for x in JobStatus]
        ):
            s = getattr(JobStatus, selection.upper())
            self.update_jobs(notset_only=True)
            self._exps_dict = {k: exp for k, exp in self._exps_dict.items() if exp.job.status == s}
            if len(self._exps_dict) == 0:
                print_error(f"Selection '{selection}' is empty !")
                sys.exit(1)
            return self

        if selection == "active":  # all experiment that have some logs to display
            self.update_jobs(notset_only=True)
            self._exps_dict = {
                k: exp for k, exp in self._exps_dict.items() if exp.job.status.has_results
            }
            if len(self._exps_dict) == 0:
                print_error(f"Selection '{selection}' is empty !")
                sys.exit(1)
            return self

        if selection == "error":  # failed or cancelled jobs
            self.update_jobs(notset_only=True)
            self._exps_dict = {
                k: exp for k, exp in self._exps_dict.items() if exp.job.status.errored
            }
            if len(self._exps_dict) == 0:
                print_error(f"Selection '{selection}' is empty !")
                sys.exit(1)
            return self

        if selection == "todo":  # failed  + not run experiments
            self.update_jobs(notset_only=True)
            self._exps_dict = {
                k: exp for k, exp in self._exps_dict.items() if exp.job.status.to_run
            }
            if len(self._exps_dict) == 0:
                print_error(f"Selection '{selection}' is empty !")
                sys.exit(1)
            return self

        # custom format using , - and : identifiers
        if isinstance(selection, str):
            index_selection = parse_range(selection=selection, max_range=len(self))
        # single integer e.g. select 0
        elif isinstance(selection, int):
            index_selection = [selection]
        # tuple of integers e.g. select 0,5,6
        else:
            index_selection = list(selection)
        self._exps_dict = {
            idx: self._exps_dict[idx] for idx in sorted(index_selection) if idx in self._exps_dict
        }
        if len(self._exps_dict) == 0:
            print_error(f"Selection '{selection}' is empty !")
            sys.exit(1)
        return self

    def flatten(self, group_by: int | None = None, keep_idx: bool = False) -> "ExperimentGrid":
        """Flatten the current experiment grid into one single experiment wwhich will
        run every experiment in the grid sequentially.

        :param group_by: If group by is a int, instead the experiment is flatenned
            into `len(self) / group_by` experiments
        :param keep_idx: By default, we flatten the experiments in a "contiguous" manner.
            But if `keep_idx` is True, we preserve the actual id of the experiment in
            the original experiment grid. This can be useful when chaining flatten
            after select or filter operations

        :return: The experiment grid of the resulting experiments
        """
        if len(self) == 0:
            return self
        new_exps_dict: dict[int, Experiment] = {}
        exp_keys = sorted(list(self._exps_dict))
        if keep_idx:
            group_by = group_by or exp_keys[-1] + 1
            num_groups = exp_keys[-1] // group_by + 1
        else:
            group_by = group_by or len(self)
            num_groups = int(ceil(len(exp_keys) / group_by))

        for grp_idx in range(0, num_groups):
            if keep_idx:
                keys_subset = sorted(
                    set(range(grp_idx * group_by, (grp_idx + 1) * group_by)).intersection(exp_keys)
                )
            else:
                keys_subset = exp_keys[grp_idx * group_by : (grp_idx + 1) * group_by]

            if len(keys_subset) == 0:
                continue

            base_exp = deepcopy(self._exps_dict[keys_subset[0]])
            # Note: we use the raw commands here, as the flattened experiment's command will
            # itself be wrapped (micromamba, git, srun...) when generating its full command
            base_exp.command = ";\n".join(self._exps_dict[key].get_cmd() for key in keys_subset)
            base_exp.idx = grp_idx
            base_exp.kwargs = {}
            base_exp.kwargs_groups = None
            if keep_idx:
                base_exp.exp_name = base_exp.name + (
                    f"_flatten={group_by * grp_idx}-{group_by * (grp_idx + 1)}"
                )
            else:
                base_exp.exp_name = base_exp.name + f"_flatten={readable_range(keys_subset)}"
            base_exp.common_args = None
            base_exp.random_port = None
            base_exp.random_seed = None
            new_exps_dict[grp_idx] = base_exp
        self._exps_dict = new_exps_dict
        return self

    def remove(self, *_args: Any, **_mismatched_flags: Any) -> None:
        """Cancel + Remove selected experiments' directory"""

        # Strict check to avoid removing experiment results by mistake !
        if len(_args) > 0 or len(_mismatched_flags):
            rich.print(
                "[red]Error[/]: Found unexpected flags following the `remove` command: ",
                ", ".join(_args) + "," if len(_args) > 0 else "",
                ", ".join(_mismatched_flags.keys()),
            )
            return

        self.__update_jobs__()
        for exp in self:
            if exp.has_log_dir:
                if self.locked and not (js := exp.job.status).failed:
                    print_error(
                        f"Experiment grid is locked: Cannot remove exp {exp.idx} with status {js}"
                    )
                    continue
                exp.print_as_header()
                print(f"Deleting directory {exp.log_dir}")
                exp.remove()

    def cancel(self, *_args: Any, **_mismatched_flags: Any) -> None:
        """Cancel job associated to the experiment"""
        # Strict check to avoid cancelling experiments by mistake !
        if len(_args) > 0 or len(_mismatched_flags):
            rich.print(
                "[red]Error[/]: Found unexpected flags following the `cancel` command: ",
                ", ".join(_args) + "," if len(_args) > 0 else "",
                ", ".join(_mismatched_flags.keys()),
            )
            return

        for exp in self:
            if exp.status.scheduled:
                exp.print_as_header()
                exp.cancel()

    def tensorboard(
        self, *_args: Any, port: int = 8897, hash_names: bool = False, bind_all: bool = False
    ) -> None:
        """Launch tensorboard on the given port for the base log dir of this experiment

        :param port: Port to serve tensorboard on
        :param hash_names: If True, use the hashed experiment names as run names
        :param bind_all: If True, serve tensorboard on all network interfaces rather than
            only localhost. Note that this exposes your logs to anyone who can reach the machine
        """
        del _args
        rich.print(f"Starting tensorboard instance on [cyan]{len(self)}[/cyan] experiments")
        # V2: only display selected experiments; using legacy logdir_spec
        specs = [
            (f"{exp.idx}_{exp.hashed_name if hash_names else exp.name}", exp.log_dir)
            for exp in self
        ]
        logdir_spec = ",".join(f"{name.replace(':', '-')}:{log_dir}" for name, log_dir in specs)
        subprocess.run(
            ["tensorboard", f"--port={port}", f"--logdir_spec={logdir_spec}"]
            + (["--bind_all"] if bind_all else []),
            check=False,
        )

    def count(self) -> None:
        """Print the number of experiments in the grid"""
        rich.print(f"[magenta]{len(self)}[/] experiments in the grid")

    def count_res(self) -> None:
        """Print the number of resources that would be used by all experiments in the grid"""
        gpus = self.num_nodes * self.num_gpus * len(self._exps_dict)
        cpus = max(1, gpus) * self.num_cpus_per_gpu
        rich.print(
            f"Running all [magenta]{len(self)}[/] experiments in the grid requires "
            f"a total of [cyan]{gpus}[/cyan] GPUs and [cyan]{cpus}[/cyan] CPUS"
        )

    def print(
        self,
        tgt: Literal[
            "cmd", "log", "logfile", "logdir", "name", "root", "status", "info", "git", "jobid"
        ] = "status",
        num_lines: int | None = None,
    ) -> None:
        """Print some information about the selected experiments

        :parm tgt: Subcommand of print
        :param num_lines: Optional for `print log`: Determines the number of lines to print;
            If positive (resp. negative), this corresponds to the `head`  (resp. `tail`) command
        """
        assert tgt in [
            "cmd",
            "log",
            "logdir",
            "logfile",
            "name",
            "root",
            "status",
            "jobid",
            "info",
            "git",
        ], f"Unknown print command {tgt}"
        if tgt == "status":
            self.update_jobs(notset_only=True)

        if tgt == "root":
            rich.print(f"[green]Root log dir:[/] {self.log_dir_root}")
            return

        max_len_name = 0
        if tgt == "status":
            max_len_name = max(len(exp.name) for exp in self)

        for exp in self:
            color = (
                "cyan"
                if (tgt == "cmd" or (tgt != "name" and exp.has_launched))
                else "bright_black"
                if tgt != "name"
                else "white"
            )
            exp.print_as_header(
                color=color,
                linejump=tgt not in ["name", "status"],
                suffix=(
                    (f"{' ' * (max_len_name - len(exp.name) + 3)}{exp.job.get_full_status_str()}")
                    if tgt == "status"
                    else ""
                ),
                max_name_length=100000 if tgt == "name" else 60,
            )

            if tgt == "cmd":
                print(self.get_full_exp_cmd(exp, local=True))

            if tgt == "info":
                try:
                    with open(exp.info_file, "r") as json_file:
                        rich.print(json.load(json_file))
                except FileNotFoundError:
                    print(f"Info file {exp.info_file} does not exist")

            elif tgt == "git":
                try:
                    with open(exp.info_file, "r") as json_file:
                        git_info = json.load(json_file).get("git", None)
                    if git_info is None:
                        print(f"No git info found under {exp.info_file}")
                    rich.print(
                        f"Git repo: [cyan]{git_info.get('repo', 'N/A')}[/cyan] "
                        f"branch: [cyan]{git_info.get('branch', 'N/A')}[/cyan] "
                        f"commit: [cyan]{git_info.get('commit', 'N/A')}[/cyan]"
                    )
                except FileNotFoundError:
                    print(f"Info file {exp.info_file} does not exist")

            elif tgt == "jobid":
                jobid = exp.jobid
                if jobid is not None:
                    print(jobid)

            elif tgt == "logdir":
                if exp.has_log_dir:
                    t = exp.last_updated_time
                    if t > 0:
                        rich.print(
                            f"[bright_black]Last modified {readable_timestamp(t)}[/bright_black]"
                        )
                    print(exp.log_dir)
                    print(" |_ ", end="", flush=True)
                    subprocess.run(["ls", exp.log_dir], check=False)
                else:
                    rich.print("[bright_black]Log dir does not exist yet[/bright_black]")
                    print(exp.log_dir)

            elif tgt == "logfile":
                if exp.has_launched:
                    t = exp.last_updated_time
                    if t > 0:
                        rich.print(
                            f"[bright_black]Last modified {readable_timestamp(t)}[/bright_black]"
                        )
                    print(exp.log_file)
                else:
                    rich.print("[bright_black]Log file does not exist yet[/bright_black]")
                    print(exp.log_file)

            elif tgt == "log":
                t = exp.last_updated_time
                if t > 0:
                    rich.print(
                        f"[bright_black]Last modified {readable_timestamp(t)}[/bright_black]"
                    )
                print_file_content(exp.log_file, num_lines=num_lines)

    def get(self, flag: str) -> None:
        """Display"""
        self.__update_jobs__()
        table = Table(
            caption=(f"Experiment Grid for `{self.base_name}` ([bold]{len(self)}[/bold] xps) "),
            show_lines=True,
        )
        for c in ["ID", "Hash", "Job", "Status", flag]:
            table.add_column(
                c,
                no_wrap=False,
                max_width=20,
                overflow="fold",
                style="cyan" if c == "Hash" else "green" if c == flag else None,
            )
        for exp in self:
            j = exp.job
            v = exp.get_value_from_anyquery(flag)
            table.add_row(
                str(exp.idx),
                str(exp.hashed_name),
                str(j.id),
                str(j.status),
                "N/A" if v is None else str(v),
            )
        rich.print(table)

    def status(self):
        """Alias for print status"""
        self.print("status")

    def log(self, num_lines: int | None = None):
        """Alias for print log"""
        self.print("log", num_lines=num_lines)

    def logdir(self):
        """Alias for print logdir"""
        self.print("logdir")

    def logfile(self):
        """Alias for print logfile"""
        self.print("logfile")

    def tailf(self, num_lines: int | None = 100):
        """tail -f on the log file of the first found running experiment"""
        self.update_jobs(notset_only=True)
        for exp in self:
            if exp.job.status.running:
                exp.print_as_header(color="white")
                t = exp.last_updated_time
                if t > 0:
                    rich.print(
                        f"[bright_black]Last modified {readable_timestamp(t)}[/bright_black]"
                    )
                subprocess.run(["tail", "-f", "-n", str(num_lines), exp.log_file], check=False)
                break

    def jobid(self):
        """Alias for print jobid"""
        self.print("jobid")

    def cmd(self):
        """Alias for print cmd"""
        self.print("cmd")

    def git(self):
        """Alias for print git"""
        self.print("git")

    def monitor(
        self,
        n: int = -1,
        refresh: int = 5,
        refresh_job_status: int = 3,
        display: str = "log_file",
    ) -> None:
        """Live monitor the log of experiments and refresh every `refresh` seconds

        :param n: Number of lines to display. A negative number means we're displaying the *last*
            `n` lines of the experiment's log files
        :param refresh: How often (in seconds) to refresh the live display
        :param refresh_job_status: Refresh job status every `refresh_job_status` refreshes.
        """

        def __generate_table__() -> Table:
            tstamp = readable_timestamp(t=None)
            table = Table(
                caption=(
                    f"{self.base_name} - last update: {tstamp} - refresh every {refresh} seconds"
                )
            )
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Job ID")
            for exp in self:
                if n != 0:
                    status_lines = exp.job.get_full_status_str().split("\n", abs(n))
                    time_lines = exp.job.get_elapsed_time_str().split("\n", abs(n))
                else:
                    status_lines = [exp.job.get_full_status_str()]
                    time_lines = [exp.job.get_elapsed_time_str()]

                table.add_row(
                    f"[green]{exp.idx}[/green]",
                    f"[green]{exp.name}[/green]",
                    status_lines.pop(0),
                    time_lines.pop(0),
                )
                if n != 0:
                    try:
                        output_lines = (
                            subprocess.check_output(
                                [
                                    "tail" if n < 0 else "head",
                                    "-n",
                                    str(abs(n)),
                                    getattr(exp, display),
                                ],
                                stderr=subprocess.DEVNULL,
                            )
                            .decode()
                            .splitlines()
                        )
                        output_lines = output_lines[n:] if n < 0 else output_lines[:n]
                        for line in output_lines:
                            table.add_row(
                                "",
                                line,
                                status_lines.pop(0) if len(status_lines) > 0 else "",
                                time_lines.pop(0) if len(time_lines) > 0 else "",
                            )
                    except subprocess.CalledProcessError:
                        table.add_row("", "", "\n".join(status_lines), "\n".join(time_lines))
            return table

        n_rows = os.get_terminal_size()[1]
        vertical_overflow: Literal["crop", "ellipsis", "visible"] = "ellipsis"
        if len(self) * (abs(n) + 1) > n_rows:
            vertical_overflow = "visible"
        try:
            with Live(
                __generate_table__(),
                auto_refresh=False,
                vertical_overflow=vertical_overflow,
            ) as live:
                refresh_counter = 0
                while 1:
                    live.update(__generate_table__(), refresh=True)
                    time.sleep(refresh)
                    # refresh job id infrequently
                    refresh_counter += 1
                    if refresh_counter % refresh_job_status == 0:
                        # here we refresh all jobs
                        self.update_jobs(notset_only=False, verbose=False)

        except KeyboardInterrupt:
            print("Interrupted !")

    def table(
        self,
        rows: str | None = None,
        columns: str | None = None,
        metrics: str | None = None,
        exclude: str | None = None,
        sort: bool | str = False,
        ascending: bool = False,
        muted: bool = False,
        short: bool = False,
        agg: Literal["mean", "min", "max"] = "mean",
        werr: bool = False,
        show_len: bool = False,
        err_fmt: str = "{:.2e}",
        float_fmt: str = "{:.3f}",
        include_job_info: bool = True,
        show_fixed: bool = False,
        live: int = 0,
        export_to: Literal["markdown", "latex"] | None = None,
    ) -> None:
        """Pretty print a list of the experiments in table form.

          * If `rows` is None, each row correspond to an experiemnt ID, the columns
            are a concatenations of stuff in columns and in metrics (all, if those are `None`)
            the content of the cell is simply the value of each column per experiment ID.
          * If `rows` is not None, then rows are subset of experiments kwargs
            the columns are either the columns given by `columns` or all columns except for `rows
            when `columns` is None
          * Finally, the content of the cell is the (averaged) metric(s) given in `metrics`.
            If `metrics` is None, it is the first metric that has been parsed by `add_metric`

        :param rows: Which hparams to display as rows. Defautls to the experiemnts' ID
            if `rows` is None
        :param columns: Which kwargs to display as columns. Defaults to all columns / rows
        :param metrics: Which metrics to display inside the table (each metric will be a sep)
        :param muted: If given, do not display the table in colors
        :param short: If given, do not expand the keyword arguments inside groups
        :param float_fmt: Format string for float numbers
        :param live: If > 0, update the table live every `live` seconds
        """

        if rows is None:
            del rows, werr, err_fmt, show_len, agg

            def __generate_table__(
                return_table: bool = True, color_maps: dict | None = None
            ) -> Table | None | tuple[Table | None, dict]:
                self.update_jobs(notset_only=False, verbose=False)
                return table_per_exp(
                    self,
                    columns=columns,
                    exclude=exclude,
                    metrics=metrics,
                    sort=sort,
                    ascending=ascending,
                    muted=muted,
                    short=short,
                    float_fmt=float_fmt,
                    return_table=return_table,
                    previous_color_maps=color_maps,
                    include_job_info=include_job_info,
                    show_fixed=show_fixed and color_maps is None,
                    export_to=export_to,
                )

        else:
            del short, sort

            def __generate_table__(
                return_table: bool = True, color_maps: dict | None = None
            ) -> Table | None | tuple[Table | None, dict]:
                self.update_jobs(notset_only=False, verbose=False)
                return (
                    table_agglomerate(
                        self,
                        rows=rows,
                        columns=columns,
                        exclude=exclude,
                        metrics=metrics,
                        ascending=ascending,
                        muted=muted,
                        agg=agg,
                        werr=werr,
                        show_len=show_len,
                        float_fmt=float_fmt,
                        err_fmt=err_fmt,
                        return_table=return_table,
                        show_fixed=show_fixed,
                        show_agglo=color_maps is None,
                        export_to=export_to,
                    ),
                    {},
                )

        if live == 0:
            __generate_table__(return_table=False)
        else:
            n_rows = os.get_terminal_size()[1]
            vertical_overflow: Literal["crop", "ellipsis", "visible"] = "ellipsis"
            if len(self) * 2 > n_rows - 5:
                vertical_overflow = "visible"
            try:
                table, color_maps = __generate_table__()  # type: ignore
                with Live(
                    table, auto_refresh=False, vertical_overflow=vertical_overflow
                ) as live_widget:
                    while 1:
                        time.sleep(live)
                        for up in self._update_metrics:
                            up(self)
                        live_widget.update(
                            __generate_table__(color_maps=color_maps)[0],  # type: ignore
                            refresh=True,
                        )
            except KeyboardInterrupt:
                print("Interrupted !")

    def run(self, print_stdout: bool = True) -> None:
        """Launch the experiments in the grid *locally* and *sequentially*"""
        for exp in self:
            if print_stdout:
                exp.print_as_header(total=len(self) - 1)
                print(self.get_full_exp_cmd(exp, local=True))
                print()
            with subprocess.Popen(
                self.get_full_exp_cmd(exp, local=True),
                shell=True,
                stdout=sys.stdout if print_stdout else subprocess.PIPE,
                stderr=sys.stderr if print_stdout else subprocess.PIPE,
            ) as p:
                p.communicate()

    def __filter_exps_to_submit__(
        self,
        overwrite: Any = None,
        resume: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> bool:
        """Filter experiments to be submitted based on whether they already have run or not

        :param overwrite: if True, will erase any existing experiments log dir and relaunch them
        :param resume: If True, will relaunch existing experiments without erasing their log dir
        :param dry_run: If True, will only display the Slurm bash script but without submitting
            any experiment
        :param quiet: If True, slightly decrease the verbosity of this function

        :return: False if the given options are invalid and nothing should be submitted
        """
        # check_overwrite = overwrite is True or "brute"
        check_overwrite = overwrite is not None and (
            (isinstance(overwrite, bool) and overwrite) or overwrite == "brute"
        )
        if check_overwrite and resume:
            print_error("`overwrite` and `resume` flags cannot both be set at the same time")
            return False

        # Then, we filter out experiments which have already ran
        remove_idx = []
        self.update_jobs(notset_only=False)
        for exp in self:
            if exp.maybe_should_not_submit:
                if check_overwrite:
                    if not dry_run:
                        remove = False
                        if overwrite != "brute":
                            if not quiet:
                                rich.print(
                                    f"\nExperiment {exp.idx:02d} exists\n"
                                    f"[yellow]{exp.log_dir}[/yellow]\n"
                                    f"[bold deep_pink4]Remove[/]"
                                    f" logdir or skip ? (y/[bold]N[/bold])"
                                )
                            remove = input() == "y"
                        else:
                            remove = True

                        if remove:
                            if self.locked and not (jsp := exp.job.status).failed:
                                print_error(
                                    f"Cannot remove experiment {exp.idx:02d} with status {jsp}: ExperimentGrid is locked "
                                )
                                remove_idx.append(exp.idx)
                            else:
                                if not quiet:
                                    rich.print(
                                        f"Experiment {exp.idx:02d} - [deep_pink4]Removed![/]"
                                    )
                                exp.remove()
                        else:
                            if not quiet:
                                rich.print(
                                    f"Experiment {exp.idx:02d} exists "
                                    f"({exp.job.status.formatted_str})"
                                    "- [green]Skipping[/green]"
                                )
                            remove_idx.append(exp.idx)
                    else:
                        rich.print(
                            f"Experiment {exp.idx:02d} exists - Would [red]overwrite[/red]"
                            f" logdir [yellow]{exp.log_dir}[/yellow]"
                        )
                # skip already running or pending experiments
                elif exp.job.status.scheduled:
                    if not quiet:
                        rich.print(
                            f"Experiment {exp.idx:02d} is scheduled "
                            f"(Job {exp.job.id} - {exp.job.status.formatted_str})"
                            f" - [green]Skipping[/green]"
                        )
                    remove_idx.append(exp.idx)
                # if experiments is still to be shceduled but the log dir is not empty,
                # if resume -> we jut submit it
                # if not resume -> we skip it
                elif not resume:
                    if not quiet:
                        rich.print(
                            f"Experiment {exp.idx:02d} exists ({exp.job.status.formatted_str})"
                            "- [green]Skipping[/]"
                        )
                    remove_idx.append(exp.idx)
                else:
                    if dry_run or not quiet:
                        rich.print(
                            f"Experiment {exp.idx:02d} exists ({exp.job.status.formatted_str})"
                            " - Will [green]resume[/green] "
                            f"from existing logdir [yellow]{exp.log_dir}[/yellow]"
                        )
                    # If resume, we will just move the slurm log file as backup but otherwise nothing
                    # prevents the experiment from running
                    if not dry_run:
                        p = Path(exp.log_file)
                        if p.exists():
                            shutil.move(p.resolve(), f"{exp.log_file}.previous")
                        if p.exists() and p.is_symlink():
                            p.unlink()

        for idx in remove_idx:
            del self._exps_dict[idx]
        return True

    def add_metric(
        self,
        which: str,
        name: str | None = None,
        file_name: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "ExperimentGrid":
        """(Chainable) Accumulate metrics to be displayed by the table command using
        chainable commands.
        Note that since different metrics may have different number of arguments,
        to properly handle the chainable structure, we need to separate add_metric
        commands with "-". For instance:

        python launcheon_scripts.py add_metric json --json_file_name results.json \
        --metrics accuracy,train_loss - table
        """
        del kwargs

        metrics_to_load = []
        if which == "json":
            json_file_name = file_name or "results.json"
            for metric in parse_as_tuple(name or "accuracy"):
                if metric in self._exps_metrics:
                    rich.print(
                        f"[purple]WARNING:[/purple]: Metric '{metric}' already exists. Skipping"
                    )
                else:
                    metrics_to_load.append(metric)

            def __load_metrics__(grid: "ExperimentGrid", load_all: bool = False) -> None:
                nonlocal json_file_name, metrics_to_load
                creation_time: int | float = time.perf_counter_ns()
                for exp in grid:
                    out = json_parser(
                        json_file_path=os.path.join(exp.log_dir, json_file_name),
                        metrics=metrics_to_load,
                        load_all=load_all,
                        verbose=verbose,
                    )
                    for key, val in out.items():
                        grid._exps_metrics[key][exp.idx] = val
                creation_time = time.perf_counter_ns() - creation_time
                rich.print(
                    f"[bright_black]\\[perf] Loaded {which} metric(s) {metrics_to_load}"
                    f" in {creation_time:.2e}ms[/bright_black]",
                    flush=True,
                )

            __load_metrics__(self)
            self._update_metrics.append(__load_metrics__)
            return self

        if which == "jsonl":
            jsonl_file_name = file_name or "train_logs.jsonl"
            name = name or "loss"
            for metric in parse_as_tuple(name):
                if metric in self._exps_metrics:
                    rich.print(
                        f"[purple]WARNING:[/purple]: Metric '{metric}' already exists. Skipping"
                    )
                else:
                    metrics_to_load.append(metric)

            def __load_metrics__(grid: "ExperimentGrid", load_all: bool = False) -> None:
                nonlocal jsonl_file_name, metrics_to_load
                creation_time: int | float = time.perf_counter_ns()
                for exp in grid:
                    out = jsonl_parser(
                        jsonl_file_path=os.path.join(exp.log_dir, jsonl_file_name),
                        metrics=metrics_to_load,
                        load_all=load_all,
                        verbose=verbose,
                    )
                    for key, val in out.items():
                        grid._exps_metrics[key][exp.idx] = val
                creation_time = time.perf_counter_ns() - creation_time
                rich.print(
                    f"[bright_black]\\[perf] Loaded {which} metric(s) {metrics_to_load}"
                    f" in {creation_time:.2e}ms[/bright_black]",
                    flush=True,
                )

            __load_metrics__(self)
            self._update_metrics.append(__load_metrics__)
            return self

        if which == "db":
            file_name = file_name or "eval_results.db:eval"
            file_name, table_name = file_name.rsplit(":", 1)
            name = name or "loss"
            for metric in parse_as_tuple(name):
                if metric in self._exps_metrics:
                    rich.print(
                        f"[purple]WARNING:[/purple]: Metric '{metric}' already exists. Skipping"
                    )
                else:
                    metrics_to_load.append(metric)

            def __load_metrics__(grid: "ExperimentGrid", load_all: bool = False) -> None:
                nonlocal file_name, metrics_to_load
                assert file_name is not None
                creation_time: int | float = time.perf_counter_ns()
                for exp in grid:
                    out = db_parser(
                        db_file_path=os.path.join(exp.log_dir, file_name),
                        table=table_name,
                        metrics=metrics_to_load,
                        load_all=load_all,
                        verbose=verbose,
                    )
                    for key, val in out.items():
                        grid._exps_metrics[key][exp.idx] = val
                creation_time = time.perf_counter_ns() - creation_time
                rich.print(
                    f"[bright_black]\\[perf] Loaded {which} metric(s) {metrics_to_load}"
                    f" in {creation_time:.2e}ms[/bright_black]",
                    flush=True,
                )

            __load_metrics__(self)
            self._update_metrics.append(__load_metrics__)
            return self

        raise NotImplementedError(f"No parse found for metric of type '{which}'")

    def plot(
        self,
        x: str,
        y: str | None = None,
        group_by: str | None | tuple[str, ...] = None,
    ) -> None:
        """In terminal Plotting utils"""
        import plotext as plt  # type: ignore

        group_by_kwargs = None if group_by is None else parse_as_tuple(group_by)
        plt.plot_size(plt.tw() / 3, plt.th() / 3)
        plt.theme("pro")
        if y is None:
            colors = {}
            for up in self._update_metrics:
                up(self, load_all=True)
            for exp in self:
                key = (
                    None
                    if group_by_kwargs is None
                    else tuple(exp.get_value_from_anyquery(kw) for kw in group_by_kwargs)
                )
                try:
                    data = self._exps_metrics[x][exp.idx]
                    if isinstance(data, str) or not isinstance(data, Iterable):
                        data = [data]
                    pts = [v for v in data if isinstance(v, (int, float))]
                    if len(pts) == 0:
                        continue
                    if key not in colors:
                        colors[key] = sample(PRETTY_TABLE_COLORS, 1)[0]
                        plt.plot(pts, label=key, color=colors[key], marker="dot")
                    else:
                        plt.plot(pts, color=colors[key], marker="dot")
                except KeyError:
                    continue
            if len(colors) > 0:
                plt.title(f"{x} - group by {group_by_kwargs}")
                plt.show()
            else:
                print_error(f"No data points found for metric {x}")
                sys.exit(1)
        else:
            xs, ys = defaultdict(lambda: []), defaultdict(lambda: [])
            for exp in self:
                key = (
                    None
                    if group_by_kwargs is None
                    else tuple(exp.get_value_from_anyquery(kw) for kw in group_by_kwargs)
                )
                try:
                    valx = self._exps_metrics[x][exp.idx]
                    if isinstance(valx, (int, float)):
                        xs[key].append(valx)
                    else:
                        raise KeyError
                except KeyError:
                    continue

                try:
                    valy = self._exps_metrics[y][exp.idx]
                    if isinstance(valy, (int, float)):
                        ys[key].append(valy)
                    else:
                        raise KeyError
                except KeyError:
                    print_error(f"Found valid metric {x} but not {y} for exp {exp.idx}")
                    continue

            if sum(len(x) for x in xs.values()) == 0:
                print_error(f"No data points found for metric {x}")
                sys.exit(1)
            if sum(len(y) for y in ys.values()) == 0:
                print_error(f"No data points found for metric {y}")
                sys.exit(1)

            print()
            if len(xs) == 1:
                key = next(iter(xs.keys()))
                plt.scatter(xs[key], ys[key], marker="dot")
            else:
                for key in xs:
                    plt.scatter(xs[key], ys[key], label=key, marker="dot")

            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{x} vs {y} - group by {group_by_kwargs}")
            plt.show()
