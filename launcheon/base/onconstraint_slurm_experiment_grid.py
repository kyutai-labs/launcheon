"""A special type of experiment that can only start when a certain constraint is met
(e.g. an input file exists), and is considered done when another constraint is met
(e.g. an output file has been created), in additional to the usual SLURM job status

This is typically used for evaluation jobs, e.g.:
  * start constraint: When a certain checkpoint exists
  * end constraints: When the eval outputs have been written somewhere,
  in which case the SLURM output logs can be removed, but the experiment
  will still be considered completed.

An example instantiation of this type of experiment can be found in
`onfileexists_experiment_grid.py` (based on file existence).
"""

from abc import abstractmethod
from typing import Any, Callable, Iterator

from launcheon.base.slurm_experiment_grid import (
    SlurmExperiment,
    SlurmExperimentGrid,
    SlurmJob,
)
from launcheon.job import JobStatus


class OnConstraintSlurmExperiment(SlurmExperiment):
    """A subclass of SlurmExperiment that starts/ends when a certain constraint is met"""

    @property
    @abstractmethod
    def can_start(self) -> bool:
        """[Need override] Return True iff the starting constraint is met"""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_ended(self) -> bool:
        """Returns True iff the ending constraint is met"""
        raise NotImplementedError

    def update_job(self) -> None:
        """Update the corresponding SLURM job status.

        OnConstraint experiment are considered finished when their corresponding
        end_constraint is met, *even if the corresponding SLURM job* is not done.

        A typical use case is an experiment grid containing many small eval jobs:
        Slurm job logs have been removed, but the job output is already
        safely written somewhere
        """
        self._job = None
        jobid = self.jobid
        if jobid is None:
            if self.has_ended:
                self._job = SlurmJob(jobid=jobid, status=JobStatus.DONE)
            else:
                super().update_job()
        # if the experiment has not met its start constraing, we define a
        # special new status
        elif not self.can_start:
            self._job = SlurmJob(jobid=jobid, status=JobStatus.CANNOTSCHEDULE)
        else:
            super().update_job()

    @property
    def maybe_should_not_submit(self) -> bool:
        """Return True iff this experiment should be considered for
        overwriting when we are tying to submit it"""
        return self.has_log_dir or self.has_ended


class OnConstraintSlurmExperimentGrid(SlurmExperimentGrid):
    """Experiment grid wrapper around `OnConstraintExperiment`"""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        """Launcheon experiment that only runs when start_file exists but end_file does not exist"""
        _experiment_constructor: Callable = OnConstraintSlurmExperiment
        if "_experiment_constructor" in kwargs:
            # 😠 Danger zone 😠 : using a custom experiment constructor
            # instead of default `SlurmExperiment`
            assert issubclass(kwargs["_experiment_constructor"], OnConstraintSlurmExperiment)
            _experiment_constructor = kwargs.pop("_experiment_constructor")
        super().__init__(
            *args,
            _experiment_constructor=_experiment_constructor,
            **kwargs,
        )

    def __iter__(self) -> Iterator[OnConstraintSlurmExperiment]:
        """Only overwriting this to get proper type hints for `SlurmExperiemnt`"""
        yield from self._exps_dict.values()  # type: ignore

    def __filter_exps_to_submit__(
        self,
        overwrite: Any = None,
        resume: bool = False,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> bool:
        """Filter out experiment that can start or have ended before submitting"""
        # since submit is not a chainable command anyway,
        # we can modify the exps_dict as much as we want
        # Filter out experiments which can not start anyway
        self._exps_dict = {k: v for k, v in self._exps_dict.items() if v.can_start}  # type: ignore
        # Default filter, it will also take into account experiment which have ended
        # because we have updated maybe_should_not_submit
        return super().__filter_exps_to_submit__(
            overwrite=overwrite, resume=resume, dry_run=dry_run, quiet=quiet
        )

    def __update_jobs__(self, notset_only: bool = False) -> None:
        """For performance reason, it's faster to update jobs in batch ( single call to sacct)"""
        # update jobids
        super().__update_jobs__(notset_only=notset_only)
        # manually set jobs of ended experiments
        for exp in self:
            if exp.has_ended:
                exp._job = SlurmJob(jobid=exp.jobid, status=JobStatus.DONE)
            elif not exp.can_start:
                exp._job = SlurmJob(jobid=exp.jobid, status=JobStatus.CANNOTSCHEDULE)
