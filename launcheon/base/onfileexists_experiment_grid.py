"""A subclass of `OnConstraintSlurmExperiment` where both the
constraints are based onfile existence
"""

import os
import shutil
from typing import Any, Callable, Iterator, Literal

import rich

from launcheon.base.onconstraint_slurm_experiment_grid import (
    OnConstraintSlurmExperiment,
    OnConstraintSlurmExperimentGrid,
)
from launcheon.utils import print_error, print_file_content


class OnFileExistsExperiment(OnConstraintSlurmExperiment):
    """A subclass of Launcheon experiment that starts/ends on file existence.
    It defines two (optional) attributes:
        * `start_file_name`: The experiment can only be submitted when this file exists
        (e.g. an eval job waiting for a specific checkpoint to exist)
        * `end_file_name`: The expeirment is considered done when this file exists
        (this means we can remove the associated slurm log file/launcheon.json and
        still have a way to check whether the experiment was already run)
    """

    def __init__(
        self,
        *args: Any,
        start_file_name: str | None = None,
        end_file_name: str | None = None,
        postprocess_start_file_name: Callable | None = None,
        postprocess_end_file_name: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """Launcheon experiment that only runs when start_file exists but end_file does not exist

        :param start_file_name: Start file name condition. Note that launcheon's secret
            keys found in this name will be expandend at initialization.
        :param end_file_name: End file name condition. Note that launcheon's secret
            keys found in this name will be expandend at initialization.
        :param postprocess_start_file_name: Optional callable applied to `start_file_name`
            at initialization. Note that postprocess can return `None` (no start constraint).
        :param postprocess_end_file_name: Optional callable applied to `end_file_name`
            at initialization. Note that postprocess can return `None` (no end constraint)`.
        """
        # disable strict warn except if the user requests it
        kwargs["strict_warn"] = kwargs.get("strict_warn", False)
        super().__init__(*args, **kwargs)

        self.start_file_name = start_file_name
        self.end_file_name = end_file_name

        # Postprocess start file name
        if self.start_file_name is not None:
            self.start_file_name = os.path.abspath(
                super().__expand_launcheon_secret_keys__(start_file_name)
            )
            if postprocess_start_file_name is not None:
                self.start_file_name = postprocess_start_file_name(self.start_file_name)

        # Postprocess end file name
        if self.end_file_name is not None:
            self.end_file_name = os.path.abspath(
                super().__expand_launcheon_secret_keys__(self.end_file_name)
            )
            if postprocess_end_file_name is not None:
                self.end_file_name = postprocess_end_file_name(self.end_file_name)

        # Post init launcheon key: to be safe, we manually replace start and
        # end file name otherwise it might create a loop in the key expansion process
        if (
            "{{launcheon.exp.end_file_name}}" in self.command  # type: ignore[has-type]
            and self.end_file_name is not None
        ):
            self.command = self.command.replace(  # type: ignore[has-type]
                "{{launcheon.exp.end_file_name}}", self.end_file_name
            )
        if "{{launcheon.exp.start_file_name}}" in self.command and self.start_file_name is not None:
            self.command = self.command.replace(
                "{{launcheon.exp.start_file_name}}", self.start_file_name
            )

    @property
    def can_start(self) -> bool:
        """Return True iff this experiment can be submitted"""
        return self.start_file_name is None or os.path.exists(self.start_file_name)

    @property
    def has_ended(self) -> bool:
        """Returns True iff this experiment has already been launched"""
        if self.end_file_name is not None:
            return os.path.exists(self.end_file_name)
        return False

    def remove_endfile(self) -> None:
        """Remove experiment logs *and* end file"""
        if self.end_file_name is not None and os.path.exists(self.end_file_name):
            if os.path.isdir(self.end_file_name):
                shutil.rmtree(self.end_file_name)
            else:
                os.remove(self.end_file_name)

    def remove(self, keep_endfile: bool = True) -> None:
        """Remove slurm logs but be careful the end file is not in the
        directory we are tying to remove...
        If you do want to remove the end file (hardcore cleanup),
        the set `keep_endfile` to False"""
        if self.has_ended and keep_endfile:
            assert self.end_file_name is not None
            ef = os.path.abspath(self.end_file_name)
            ld = os.path.abspath(self.log_dir)
            if ef[: len(ld)] == ld:
                rich.print(
                    "[bold red]End file[/bold red] is in the"
                    "experiment's log directory; Not removing logs"
                )
                return
        # Otherwise, remove the slurm logs only
        super().remove()
        if not keep_endfile:
            self.remove_endfile()


class OnFileExistsExperimentGrid(OnConstraintSlurmExperimentGrid):
    """Experiment grid wrapper around `OnFileExistsExperiment`"""

    def __init__(
        self,
        *args: Any,
        start_file_name: str | None = None,
        end_file_name: str | None = None,
        postprocess_start_file_name: Callable | None = None,
        postprocess_end_file_name: Callable | None = None,
        **kwargs: Any,
    ):
        """Experiment grid wrapper around `OnFileExistsExperiment`

        :param start_file_name: Start file name condition. In most cases, this should contain
            launcheon's secret keys that can then be expanded by each unique experiment.
        :param end_file_name: End file name condition. In most cases, this should contain
            launcheon's secret keys that can then be expanded by each unique experiment.
        :param postprocess_start_file_name: Optional callable applied to `start_file_name`
            at initialization
        :param postprocess_end_file_name: Optional callable applied to `end_file_name`
            at initialization
        """
        super().__init__(
            *args,
            _experiment_constructor=OnFileExistsExperiment,
            _experiment_constructor_kwargs=dict(
                start_file_name=start_file_name,
                end_file_name=end_file_name,
                postprocess_start_file_name=postprocess_start_file_name,
                postprocess_end_file_name=postprocess_end_file_name,
            ),
            **kwargs,
        )

    def __iter__(self) -> Iterator[OnFileExistsExperiment]:
        """Only overwriting this to get proper type hints for `SlurmExperiemnt`"""
        yield from self._exps_dict.values()  # type: ignore

    def remove_endfile(self) -> None:
        """Remove end file for all experiments that have already ended"""
        for exp in self:
            exp.remove_endfile()

    def remove(self, *_args: Any, keep_endfile: bool = True, **_mismatched_flags: Any) -> None:
        """Remove end file for all experiments that have already ended"""
        if len(_args) > 0 or len(_mismatched_flags):
            print_error(
                (
                    "Found unexpected flags following the `remove` command: "
                    + ", ".join(_args)
                    + ","
                    if len(_args) > 0
                    else "" + ", ".join(_mismatched_flags.keys())
                ),
            )
            return
        for exp in self:
            exp.remove(keep_endfile=keep_endfile)

    def print(
        self,
        tgt: Literal[
            "cmd",
            "log",
            "logfile",
            "logdir",
            "name",
            "root",
            "slurm_cmd",
            "status",
            "info",
            "git",
            "jobid",
            "constraint",
            "endfile_name",
            "endfile",
        ] = "status",
        num_lines: int | None = None,
    ) -> None:
        if tgt in ["endfile", "endfile_name"]:
            for exp in self:
                color = "cyan" if exp.has_launched else "bright_black"
                exp.print_as_header(
                    color=color,
                    linejump=True,
                )
                if tgt == "endfile_name":
                    print(exp.end_file_name)
                elif tgt == "endfile":
                    if exp.end_file_name is None:
                        rich.print("[bright_black]No end file specified[/bright_black]")
                    else:
                        print_file_content(exp.end_file_name, num_lines=num_lines)
        else:
            super().print(tgt=tgt, num_lines=num_lines)  # type: ignore
