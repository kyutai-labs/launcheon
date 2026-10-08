"""Implements the base class for a single experiment in the grid.
Note that some methods are abstract as they will depend on the underlying cluster used.
See `base/slurm_experiment_grid.py` for an example instantiation"""

import ast
import json
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import rich
import rich.markup

from launcheon.global_variables import EXP_INFO_FILE, STRING_TO_FILTER_OP
from launcheon.job import Job, JobStatus
from launcheon.parser_format import ArgparseParserFormat, ParserFormat
from launcheon.utils import (
    file_or_link_exists,
    file_or_link_timestamp,
    find_repeated_flags,
    format_kwargs,
    get_git_revision_hash,
    hash_experiment_name,
    parse_as_tuple,
    print_error,
    print_warning,
    resolve_kwargs_nicknames,
    safe_convert_to_number,
    sort_mixed_int_str,
    strip_url_credentials,
)

NAME_DEPENDENT_SECRET_KEYS = ["name", "hashed_name", "log_dir"]


def is_expandable_arg(value: str) -> Literal["yes", "no", "delay"]:
    """Returns true iff a flag's value can be expanded with a launcheon secret key"""
    if (
        isinstance(value, str)
        and len(re.findall(r"\{\{launcheon.exp.(?P<flag>[\w.]+)\}\}", value)) > 0
    ):
        if any(("{{launcheon.exp." + key + "}}") in value for key in NAME_DEPENDENT_SECRET_KEYS):
            return "delay"
        return "yes"
    return "no"


def is_safe_dirname(name: str) -> bool:
    """Returns True iff `name` can be used as a single directory name inside the log root"""
    return (
        len(name) > 0
        and name not in {".", ".."}
        and os.sep not in name
        and (os.altsep is None or os.altsep not in name)
        and "\0" not in name
        and len(name.encode("utf-8")) <= 255
    )


def get_name_from_kwargs(exp_kwargs: dict[str, Any]) -> str:
    exp_name = ""
    for k in sorted(exp_kwargs.keys()):
        v = (
            "-".join(str(x) for x in exp_kwargs[k])
            if isinstance(exp_kwargs[k], tuple)
            else str(exp_kwargs[k])
        )
        # these two secrete keys are treated dependently to because they depend on the name creation
        if any(("{{launcheon.exp." + key + "}}") in v for key in NAME_DEPENDENT_SECRET_KEYS):
            continue
        k = k.split(":", 1)[0]
        exp_name += f"_{k}={v}"
    return exp_name[1:]


@dataclass
class Experiment(ABC):
    """A single experiment in the grid

    :param idx: Index of the experiment in the grid
    :param kwargs: Command flag arguments unique to this experiment
    :param kwargs_groups: Name of keyword arguments groups in this experiments, as well as
        name of kwargs linked to it. This is mainly used for filtering and pretty printing a
        table of the experiment grid
    :param exp_name: Optional. Specific experiment name. If not given, will be automatically
        created from `kwargs`
    :param base_name: Optional. Base name prefix for all experiments
    :param log_dir_root: Optional. Log dir root common to all experiments in the grid
    :param command: Optional. The base command to run
    :param common_args: Optional. the command flag arguments which are shared across experiments
        (these won't be used for naming)
    :param use_hashed_dirname: If True, will use a shorted hashed name for
        this experiment's log directory
    :param random_port: Defines a random port for this experiment at initialization
        It can be accessed via {{launcheon.exp.port}}
    :param random_seed: Defines a random seed for this experiment at initialization
        It can be accessed via {{launcheon.exp.seed}}
    :param strict_warn: For advanced users; Disable warnings against unresolved
        launcheon secret keys. Only use it if you have keys you want to expand manually
        after intiialization
    """

    # experiment specific
    idx: int
    # kwargs dict as given in the sweep (i.e. may still contain nicknames)
    # but groups already flattened/extended
    kwargs: dict[str, Any]
    # Maps the name of a kwargs groups to its value for this experiment and
    # the list of kwargs defined for this group
    kwargs_groups: dict[str, tuple[str, list[str]]] | None = None
    # Normally, exp_name is defined on-the-fly for each experiment
    # However it is often safer to define it at the experiment_grid to
    # avoid cyclic structures in laucheon secret keys replaement
    exp_name: str | None = None
    # Common to all experiments
    base_name: str = ""
    log_dir_root: str = ""
    override_log_file: str | None = None
    command: str = ""
    common_args: dict[str, Any] | None = None
    # if True, will use a shortened hash of the unique experiment name as a log directory
    use_hashed_dirname: bool = False
    # Set to True if the hashed dirname was forced because the name is not a valid dirname
    forced_hashed_dirname: bool = field(default=False, init=False)
    expand_kwargs_in_name: bool = True
    include_common_args_in_name: bool = True
    parser_format: ParserFormat = ArgparseParserFormat()
    # Random port or seed assigned at experiment's generation
    random_port: int | None = None
    random_seed: int | None = None
    # can be used to disable some warning when doing weird experimental stuff
    strict_warn: bool = True
    # Kwargs nicknames
    _kwargs_nicknames: dict[str, str] | None = None
    _reverse_kwargs_nicknames: dict[str, str] | None = None
    # Job associated to this experiment
    _job: Job | None = None

    def __post_init__(self) -> None:
        """Post init function"""
        # A. Handle repeated flags (e.g. `append` action in argparse, multiple=True in click, etc.)
        # note that we also handle possible duplicates here for simplicity
        # in contrast, for "normal" arguments, duplicates are handled at the experiment grid
        # level, which yields more information (e.g. where does the conflict coems from)
        repeated_flags = None
        found_repeat_in_common: set[str] = set()
        if self.common_args is not None:
            repeated_flags, found_repeat_in_common = find_repeated_flags(
                self.common_args,
                repeated_flags=repeated_flags,
                allow_override_initial_value=False,
            )
        repeated_flags, found_repeat_in_self = find_repeated_flags(
            self.kwargs,
            repeated_flags=repeated_flags,
            allow_override_initial_value=True,
        )
        # of course there could be some repeated flags hidden in groups
        # Delete repeated flags from their respectful source
        if self.common_args is not None:
            for key in found_repeat_in_common - found_repeat_in_self:
                self.common_args[key] = tuple(
                    repeated_flags[key][idx] for idx in sort_mixed_int_str(repeated_flags[key])
                )

        for key in found_repeat_in_self:
            self.kwargs[key] = tuple(
                repeated_flags[key][idx] for idx in sort_mixed_int_str(repeated_flags[key])
            )

        # 3. Finally, if exp name was defined when initializing the experiment grid,
        # also expand any secret keys
        # Define name of the experiments
        self.exp_name = (
            f"{self.base_name}{'_' if (len(self.base_name) > 0 and len(self.kwargs) > 0) else ''}"
            + (
                self.exp_name
                or get_name_from_kwargs(
                    self.kwargs
                    if not self.expand_kwargs_in_name
                    else self.get_full_kwargs_dict(
                        include_common_args=self.include_common_args_in_name
                    )
                )
            )
        )

        # B. Expand launcheon secret keys
        # 1. Expand launcheon keys in the base command
        self.command = self.__expand_launcheon_secret_keys__(self.command)

        # 2. Expand launcheon keys in the experiment's kwargs
        delayed_replace = []
        for key, value in self.kwargs.items():
            # string field
            if expand := is_expandable_arg(value):
                if expand == "delay":
                    delayed_replace.append(key)
                    continue
                self.kwargs[key] = self.__expand_launcheon_secret_keys__(value)
            elif isinstance(value, tuple):
                if any(is_expandable_arg(x) == "delay" for x in value):
                    delayed_replace.append(key)
                    continue
                self.kwargs[key] = tuple(
                    (
                        x
                        if is_expandable_arg(x) != "yes"
                        else self.__expand_launcheon_secret_keys__(x)
                    )
                    for x in value
                )

        self.exp_name = self.__expand_launcheon_secret_keys__(self.exp_name)

        # Safeguard: names containing a path separator (e.g. a path in a sweep value)
        # or that are too long cannot be safely used as a directory name. They would
        # create nested (or out-of-root) log directories, which then get deleted by
        # `remove` / `submit --overwrite`. Fall back to the hashed name instead.
        if not self.use_hashed_dirname and not is_safe_dirname(self.exp_name or ""):
            self.use_hashed_dirname = True
            self.forced_hashed_dirname = True

        # Finally, expand special keys that depend on exp_name
        for key in delayed_replace:
            value = self.kwargs[key]
            if isinstance(value, str):
                self.kwargs[key] = self.__expand_launcheon_secret_keys__(value)
            elif isinstance(value, tuple):
                self.kwargs[key] = tuple(self.__expand_launcheon_secret_keys__(x) for x in value)

        # C. Defining a mapping from kwarg name to the group they belong
        # This is mainly used for unified colors in table pretty printing
        self.kwarg_to_group = {}
        if self.kwargs_groups is not None:
            for group_name, (group_value, linked_kwargs) in self.kwargs_groups.items():
                for kwarg in linked_kwargs:
                    self.kwarg_to_group[kwarg] = (group_name, group_value)

        # reverse nickname for faster searching
        if self._kwargs_nicknames is not None:
            self._reverse_kwargs_nicknames = {v: k for k, v in self._kwargs_nicknames.items()}

    @abstractmethod
    def cancel(self) -> None:
        """[Need override] Cancels the job corresponding to this experiment"""
        raise NotImplementedError

    @abstractmethod
    def update_job(self) -> None:
        """[Need override] set the Job corresponding to this experiment in self._job."""
        # use given jobid or read it from the exp's info file
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def __base_log_file__(cls) -> str:
        """[Need override] Log file for this experiment, relative to the base log directory"""
        raise NotImplementedError

    def __expand_launcheon_secret_keys__(self, value: Any) -> Any:
        """Expand Launcheon's secret keys in all kwargs of the given experiment"""
        if isinstance(value, str):
            # Special secret keys that expand an experiment *attribute*
            if "{{launcheon.exp.log_dir}}" in value:
                value = value.replace("{{launcheon.exp.log_dir}}", self.log_dir)

            if "{{launcheon.exp.name}}" in value:
                value = value.replace("{{launcheon.exp.name}}", self.name)

            if "{{launcheon.exp.hashed_name}}" in value:
                value = value.replace("{{launcheon.exp.hashed_name}}", self.hashed_name)

            if "{{launcheon.exp.port}}" in value:
                value = value.replace("{{launcheon.exp.port}}", str(self.random_port))

            if "{{launcheon.exp.seed}}" in value:
                value = value.replace("{{launcheon.exp.seed}}", str(self.random_seed))

            # Otherwise the default case is to expand any of the experiment *kwargs*
            for pattern in re.findall(r"\{\{launcheon.exp.(?P<flag>[\w.]+)\}\}", value):
                replace = self.get_value_from_anyquery(pattern)
                if replace is not None:
                    aux = f"launcheon.exp.{pattern}"
                    value = value.replace("{{" + aux + "}}", str(replace))

            # If there are any secret keys left it's probably because there is a
            # nested structure happening, which we do not suport because handling
            # the right ordering would be a hassle
            # arg1: {{launcheon.arg2}}
            # arg2: {{launcheon.arg3}}
            extra_patterns = re.findall(r"\{\{launcheon.exp.(?P<flag>\w+)\}\}", value)
            if len(extra_patterns) > 0 and self.strict_warn:
                print_warning(
                    "Could not expand launcheon secret keys: "
                    f"[yellow]{','.join(x for x in extra_patterns)}[/yellow].\n"
                    "This is likely due to chained secret keys, for which the "
                    "resolving order is not guaranteed. If this is not expected"
                    " behavior, please modify your launcheon script accordingly."
                )
                sys.exit(1)
        elif isinstance(value, tuple):
            return tuple(self.__expand_launcheon_secret_keys__(v) for v in value)
        return value

    @property
    def name(self) -> str:
        """Returns a unique name for the experiment

        :param expand_group: If True, expand the kwargs found under each group in the name.
        Otherwise (defaults) use groups in experiment naming for shorter names
        """
        assert self.exp_name is not None
        return self.exp_name

    @property
    def hashed_name(self) -> str:
        """Returns a shortened hash version of the experiment name. Can be used for
        unique indexing of the experiment"""
        return hash_experiment_name(self.name)

    @property
    def log_dir(self) -> str:
        """Log directory for this experiment"""
        return os.path.join(
            self.log_dir_root,
            self.hashed_name if self.use_hashed_dirname else self.name,
        )

    @property
    def has_log_dir(self) -> bool:
        """Returns true iff the log dir for this experiment already exists"""
        return os.path.exists(self.log_dir)

    @property
    def maybe_should_not_submit(self) -> bool:
        """if True, this will trigger the check for whether to submit
        this experiment in the `submit_exps` command. The check is resolved
        according to the flags to `submit` (e.g. overwrite, resume etc)"""
        if os.path.exists(self.log_dir):
            # if any files were generated
            for f in Path(self.log_dir).iterdir():
                if f.name not in {EXP_INFO_FILE, self.__base_log_file__()}:
                    return True
            # if log_file exists and is not empty
            log_file = self.log_file
            if (
                os.path.exists(log_file)
                and os.path.isfile(log_file)
                and os.path.getsize(log_file) != 0
            ):
                return True
        return self.job.status.scheduled

    def remove(self) -> None:
        """Remove this experiment's logs + cancel job if running"""
        if self.job.status.scheduled:
            self.cancel()
        if self.has_log_dir:
            shutil.rmtree(self.log_dir)

    def get_full_kwargs_dict(
        self,
        include_base_command: bool = False,
        include_common_args: bool = True,
    ) -> dict[str, Any]:
        """Return a dictionary containing all expanded flags fed to
        this experiment, including those included in the base command

        :param include_base_command: If True, also attempts to parse kwargs from
            the base command; This should generally be False, but can be useful
            when one want to obtain *all flags* fed to the command line, outside
            of the general launcheon loop

        :return: A dictionary mapping a string to its kwargs values
        """
        src = {}
        if self.common_args is not None and include_common_args:
            src.update(self.common_args)

        if include_base_command:
            # Parse any flags given to the base command into common flags
            # because it will make life easier
            pattern = self.parser_format.kwargs_formatting.format(
                flag=r"([a-zA-Z_0-9]+)",
                value=r"([^-]*)" + self.parser_format.kwargs_formatting_separator,
            )
            move_to_kwargs: dict[str, Any] = {}
            if not self.command.endswith(self.parser_format.kwargs_formatting_separator):
                cmd = self.command + self.parser_format.kwargs_formatting_separator
            else:
                cmd = self.command
            for flag, value in re.findall(pattern, cmd):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                # note that any duplicate from repeated flags is handled
                # in the experiment's __post_init__
                if flag in move_to_kwargs:
                    # if duplicate, we assume the user is smart enough and that they meant
                    # to write a repeat flag
                    if isinstance(move_to_kwargs[flag], list):
                        move_to_kwargs[flag].append(value)
                    else:
                        move_to_kwargs[flag] = [move_to_kwargs[flag], value]
                else:
                    move_to_kwargs[flag] = value
            src.update(
                {k: (tuple(x) if isinstance(x, list) else x) for k, x in move_to_kwargs.items()}
            )

        src.update(self.kwargs)

        return resolve_kwargs_nicknames(src, kwargs_nicknames=self._kwargs_nicknames)

    def get_formatted_cmd_kwargs(self, include_common_args: bool = True) -> str:
        """Returns all command flags for this experiment, correctly formatted and expanded
        to form the command line as a string

        :param include_common_args: Whether to include args common to all experiments,
            or only the ones specific to the current experiment
        :param quote_strings: If True, will add extra quotes around the values of type string
        :param formatting: A format string indicating the keyword format. Defaults to "--flag value"
        :param formatting separator: Defines how to joint the formatted flag/value pairs

        :return: The list of command flags, formatted as a single string
        """
        src = {**self.kwargs}
        if include_common_args and self.common_args is not None:
            src.update(self.common_args)

        return format_kwargs(
            kwargs=src, kwargs_nicknames=self._kwargs_nicknames, parser_format=self.parser_format
        )

    def get_cmd(self) -> str:
        """Generate the full command to be run by this experiment

        :param quote_strings: If True, will add extra quotes around the values of type string
        :param kwargs_formatting: A format string indicating the keyword format. Defaults
            to "--flag value"
        :param kwargs_formatting separator: Defines how to joint the formatted flag/value pairs

        :return: The command to be run as a string
        """
        formatted_kwargs = self.get_formatted_cmd_kwargs(include_common_args=True)
        # Special launcheon keys to select where to insert the kwargs
        if "{{launcheon.kwargs_placeholder}}" in self.command:
            return self.command.replace("{{launcheon.kwargs_placeholder}}", formatted_kwargs)
        # Default: just append the formatted kwargs to the command
        return f"{self.command} {formatted_kwargs}"

    def __get_value_from_anyquery__(self, flag: str) -> Any:
        """Get the value for a given flag | nicknames. The resolution order is as follows:
        * kwargs
        * nicknamed kwarg (when `kwargs_nicknames` is not None)
        * kwargs group name
        * common kwargs
        * expanded kwarg (i.e. common kwarg with launcheon secret key value)

        Not that this does not expand potential nicknames in the output values, unlike
        the method `get_value_from_anyquery`
        """
        pos: int | None = None
        if ":" in flag:
            flag, aux = flag.rsplit(":", 1)
            try:
                pos = int(aux)
            except ValueError as e:
                print_error(f"Wrong format for {flag}:{aux}; Expected an integer after `:`")
                raise e

        # 1. this is a normal flag
        if flag in self.kwargs:
            return self.kwargs[flag] if pos is None else self.kwargs[flag][pos]

        # 2. This is the target of a nickname
        if self._reverse_kwargs_nicknames is not None and flag in self._reverse_kwargs_nicknames:
            alias_flag = self._reverse_kwargs_nicknames[flag]
            if alias_flag in self.kwargs:
                return self.kwargs[alias_flag] if pos is None else self.kwargs[alias_flag][pos]
            if self.common_args is not None and alias_flag in self.common_args:
                return (
                    self.common_args[alias_flag]
                    if pos is None
                    else self.common_args[alias_flag][pos]
                )

        # 3. this is a group name
        if self.kwargs_groups is not None and flag in self.kwargs_groups:
            if pos is not None:  # this would just be weird tbg
                raise ValueError(
                    f"Found identical name {flag} between kwargs group and repeated kwarg"
                )
            return self.kwargs_groups[flag][0]

        # 4. this is one of the common args
        if self.common_args is not None and flag in self.common_args:
            return self.common_args[flag] if pos is None else self.common_args[flag][pos]

        # 2bis. This is a nicknmae (low priority because it is a bif of an edge case)
        # this would happen if we are trying to query for a nickname which is not
        # actually being used (not found in kwargs)
        if self._kwargs_nicknames is not None and flag in self._kwargs_nicknames:
            true_flag = self._kwargs_nicknames[flag]
            if true_flag in self.kwargs:
                return self.kwargs[true_flag] if pos is None else self.kwargs[true_flag][pos]
            if self.common_args is not None and true_flag in self.common_args:
                return (
                    self.common_args[true_flag] if pos is None else self.common_args[true_flag][pos]
                )

        # 5. last resort: it is a base attribute of the experiment object itself
        if hasattr(self, flag) and pos is None:
            return getattr(self, flag)
        # not found

        return None

    def get_value_from_anyquery(self, flag: str, expand: bool = True) -> Any:
        """Same as get_value_from_anyquery, but additionally expand alias/nicknames"""
        v = self.__get_value_from_anyquery__(flag)
        if v is None or not expand or self._kwargs_nicknames is None:
            return v
        return self._kwargs_nicknames.get(v, v)

    def __set_value_from_anyquery__(self, flag: str, value: Any) -> None:
        """Get the value for a given flag | nicknames. The resolution order is as follows:
        * kwargs
        * nicknamed kwarg (when `kwargs_nicknames` is not None)
        * kwargs group name
        * common kwargs
        * expanded kwarg (i.e. common kwarg with launcheon secret key value)

        Not that this does not expand potential nicknames in the output values, unlike
        the method `get_value_from_anyquery`
        """
        pos: int | None = None
        if ":" in flag:
            flag, aux = flag.rsplit(":", 1)
            try:
                pos = int(aux)
            except ValueError as e:
                print_error(f"Wrong format for {flag}:{aux}; Expected an integer after `:`")
                raise e

        # 1. this is a normal flag
        if flag in self.kwargs:
            if pos is None:
                self.kwargs[flag] = value
            else:
                self.kwargs[flag][pos] = value

        # 2. This is the target of a nickname
        elif self._reverse_kwargs_nicknames is not None and flag in self._reverse_kwargs_nicknames:
            alias_flag = self._reverse_kwargs_nicknames[flag]
            if alias_flag in self.kwargs:
                if pos is None:
                    self.kwargs[alias_flag] = value
                else:
                    self.kwargs[alias_flag][pos] = value
            if self.common_args is not None and alias_flag in self.common_args:
                if pos is None:
                    self.common_args[alias_flag] = value
                else:
                    self.common_args[alias_flag][pos] = value

        # 3. this is a group name
        elif self.kwargs_groups is not None and flag in self.kwargs_groups:
            raise NotImplementedError(f"Replacing group names is not allowed (`{flag}`, `{value}`)")

        # 4. this is one of the common args
        elif self.common_args is not None and flag in self.common_args:
            if pos is None:
                self.common_args[flag] = value
            else:
                self.common_args[flag][pos] = value

        # 2bis. This is a nicknmae (low priority because it is a bif of an edge case)
        # this would happen if we are trying to query for a nickname which is not
        # actually being used (not found in kwargs)
        elif self._kwargs_nicknames is not None and flag in self._kwargs_nicknames:
            true_flag = self._kwargs_nicknames[flag]
            if true_flag in self.kwargs:
                if pos is None:
                    self.kwargs[true_flag] = value
                else:
                    self.kwargs[true_flag][pos] = value
            if self.common_args is not None and true_flag in self.common_args:
                if pos is None:
                    self.common_args[true_flag] = value
                else:
                    self.common_args[true_flag][pos] = value

        # 5. last resort: it is a base attribute of the experiment object itself
        elif hasattr(self, flag) and pos is None:
            setattr(self, flag, value)

    def set_value_from_anyquery(self, flag: str, value: Any) -> None:
        """Set given flag to the given value"""
        self.__set_value_from_anyquery__(flag, value)

    def filter(self, kwarg_name: str, kwarg_op: str, kwarg_value: str) -> bool:
        """Returns True iff the current experiment matches the specified filter

        Note that this also takes into account nicknames and grouped kwargs,
        resolved in the following order:
        no expansion -> expand group -> expand nicknames on keys -> expand nicknames on values

        :param kwarg_name: The flag we are filtering
        :param kwarg_op: The comparison operatoin
        """
        # not in and in are a bit special operation and are based on regexp
        if kwarg_op in {"in", "notin"}:
            val = self.get_value_from_anyquery(kwarg_name)
            m = re.search(re.compile(str(kwarg_value)), str(val))
            return (m is None) ^ (kwarg_op == "in")

        # for simpler binary operation
        target_values = [kwarg_value]
        if self._kwargs_nicknames is not None and kwarg_value in self._kwargs_nicknames:
            target_values.append(self._kwargs_nicknames[kwarg_value])

        val = self.get_value_from_anyquery(kwarg_name)
        for tgt in target_values:
            if STRING_TO_FILTER_OP[kwarg_op](
                safe_convert_to_number(val), safe_convert_to_number(tgt)
            ):
                return True
            if isinstance(val, tuple) and isinstance(tgt, str):
                if STRING_TO_FILTER_OP[kwarg_op](sorted(val), sorted(parse_as_tuple(tgt))):
                    return True
        return False

    @property
    def log_file(self) -> str:
        """Log file for this experiment.
        Note that log file is always expected to be inside log dir"""
        return os.path.join(
            self.log_dir,
            self.override_log_file
            if self.override_log_file is not None
            else self.__base_log_file__(),
        )

    @property
    def last_updated_time(self) -> float:
        """Returns last time the log file for this experiment was updated"""
        return file_or_link_timestamp(self.log_file)

    @property
    def has_launched(self) -> bool:
        """Returns True iff this experiment has already been launched"""
        return file_or_link_exists(self.log_file)

    @property
    def job(self, update: bool = False) -> Job:
        """Return the cluster JOB associated to the current experiment. By
        default, this returns the currently saved job without looking up
        its potentially new status on the cluster. This behavior can be
        forced by setting ``update``

        :return: A Job object that should (at the minimum encapsulate) a job ID,
            the corresponding job status, the job start time if applicable
        """
        if self._job is None or update:
            self.update_job()
            assert self._job is not None
        return self._job

    @property
    def status(self) -> JobStatus:
        """Return status of the job associated with the current experiment"""
        return self.job.status

    @property
    def info_file(self) -> str:
        """File storing the slurm job id of the corresponding experiment"""
        return os.path.join(self.log_dir, EXP_INFO_FILE)

    def register_exp_info(
        self,
        jobid: str,
        git_sync: bool = True,
        git_repo: str = "",
        git_branch: str = "",
        git_commit: str = "",
    ) -> None:
        """Register the given job ID and exp info in the experiment's info file"""
        d = {
            "jobid": jobid,
            "name": {"full": self.name, "hashed": self.hashed_name},
            "cli_flags": {
                "kwargs": self.kwargs,
                "common_args": self.common_args,
            },
            "launcheon": {
                "kwargs_groups": self.kwargs_groups,
                "kwargs_nicknames": self._kwargs_nicknames,
                "random_port": self.random_port,
                "random_seed": self.random_seed,
                "base_grid_name": self.base_name,
                "base_grid_cmd": self.command,
            },
            "git": get_git_revision_hash(),
        }
        if git_sync:
            d["git"]["repo"] = git_repo
            d["git"]["branch"] = git_branch
            d["git"]["commit"] = git_commit
        # Never record credentials embedded in the remote URL in the (shareable) info file
        if isinstance(d["git"].get("repo"), str):
            d["git"]["repo"] = strip_url_credentials(d["git"]["repo"])
        # write atomically so that concurrent readers never see a partial file
        tmp_info_file = f"{self.info_file}.tmp"
        with open(tmp_info_file, "w") as tgt_file:
            json.dump(d, tgt_file, indent=4, sort_keys=True)
        os.replace(tmp_info_file, self.info_file)

    @property
    def jobid(self) -> str | None:
        """Return experiment's job ID after reading the exp's info file

        :return: The Job ID (as a string) if it has been scheduled, otherwise None
        """
        if self._job is not None:
            return self._job.id

        if os.path.exists(self.info_file):
            try:
                with open(self.info_file, "r") as tgt_file:
                    out = json.load(tgt_file).get("jobid", None)
                    # Job IDs end up in shell commands (sacct, scancel...): only accept
                    # well-formed IDs (e.g. "123" or "123_4" for job arrays)
                    if not isinstance(out, str) or re.fullmatch(r"\d+(_\d+)?", out) is None:
                        out = None
                    return out
            except json.JSONDecodeError:
                return None
        return None

    def print_as_header(
        self,
        color: str = "green",
        linejump: bool = True,
        suffix: str = "",
        end: str = "\n",
        total: int | None = None,
        max_name_length: int = 60,
    ) -> None:
        """Pretty print a header with the name of this experiment

        :param color: Base color
        :param linejump: Whether to line jump (*before*) the title
        :param suffix: Any extra suffix
        :param end: End (added after suffix)
        :param total: Total number of experiments. If given, will add
        a "/ total " after the exp's ID
        """
        hash_add = f"[yellow]({self.hashed_name})[/yellow] " if self.use_hashed_dirname else ""
        idx = f"{self.idx:03d}" if total is None else f"{self.idx:03d} / {total:03d}"
        escaped_name = rich.markup.escape(self.name)
        rich.print(
            ("\n" if (self.idx > 0 and linejump) else "")
            + f"[bold {color}]{idx}[/bold {color}] - "
            + f"{hash_add}[bold {color}]{escaped_name[:max_name_length]}{'...' if len(escaped_name) > max_name_length else ''}[/bold {color}]{suffix}",
            end=end,
        )
