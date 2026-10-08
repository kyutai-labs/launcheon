"""Wrapper encapsulating job info for a given experiment"""

import time
from abc import abstractmethod
from enum import Enum, unique

from launcheon.utils import readable_elapsed, readable_timestamp


@unique
class JobStatus(Enum):
    """Status of a job corresponding to an experiment"""

    SACCTERROR = -1  # error in querying job status
    DONE = 0  # succesfully completed
    FAILED = 1  # unsuccesfully completed
    RUNNING = 2  # currently running
    PENDING = 3  # scheduled but has yet to run
    UNSCHEDULED = 4  # never run but not submitted
    CANCELLED = 5  # submitted but cancelled
    CANNOTSCHEDULE = 6  # special status for experiment that can only
    # be submitted when they meet a certain contraint
    # e.g. OnFileExistsExperiment

    @property
    def completed(self) -> bool:
        """Return True iff this job has run and ended (succsefully or not)"""
        return self in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]

    @property
    def done(self) -> bool:
        """Return True iff the has succesfully completed"""
        return self == JobStatus.DONE

    @property
    def failed(self) -> bool:
        """Return True iff the job has crashed"""
        return self == JobStatus.FAILED

    @property
    def scheduled(self) -> bool:
        """Return True iff this job has been scheduled (running or pending)"""
        return self in [JobStatus.RUNNING, JobStatus.PENDING]

    @property
    def cancelled(self) -> bool:
        """Return True iff the current status is CANCELLED"""
        return self == JobStatus.CANCELLED

    @property
    def pending(self) -> bool:
        """Return True iff the current job is in pending status"""
        return self in [JobStatus.PENDING, JobStatus.CANNOTSCHEDULE]

    @property
    def running(self) -> bool:
        """Return True iff the current status is RUNNING"""
        return self == JobStatus.RUNNING

    @property
    def errored(self) -> bool:
        """Return True iff the job has stopped but did not complete successfully"""
        return self in [JobStatus.CANCELLED, JobStatus.FAILED]

    @property
    def has_results(self) -> bool:
        """Return True iff this job status qualifies jobs that have produced some results"""
        return self in [JobStatus.DONE, JobStatus.RUNNING]

    @property
    def to_run(self) -> bool:
        """Return True iff this job status qualifies jobs that still needs to be run"""
        return self in [
            JobStatus.FAILED,
            JobStatus.UNSCHEDULED,
            JobStatus.CANCELLED,
            JobStatus.CANNOTSCHEDULE,
        ]

    @property
    def color(self) -> str:
        """Color of the job status for pretty printing"""
        if self == JobStatus.FAILED:
            return "red"
        if self == JobStatus.CANCELLED:
            return "orange1"
        if self == JobStatus.DONE:
            return "dark_green"
        if self == JobStatus.RUNNING:
            return "dodger_blue1"
        if self == JobStatus.CANNOTSCHEDULE:
            return "dark_violet"
        if self == JobStatus.SACCTERROR:
            return "grey42"
        return "white"

    @property
    def formatted_str(self) -> str:
        """Returns a pretty printed string for the current job"""
        c = self.color
        return f"[{c}]{self.name}[/{c}]"


class Job:
    """A Job"""

    def __init__(
        self,
        jobid: str | None = None,
        status: JobStatus = JobStatus.UNSCHEDULED,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        """A Job

        :param jobid: Unique identifier for the job
        :param status: Initial status for the job
        :param start_time: Start time for the job if applicable
        :param end_time: End time for the job if applicable
        """
        self.id = jobid
        self.status = status
        self._start_time = start_time
        self._end_time = end_time

    def get_full_status_str(self, extra_verbose: bool = False) -> str:
        """Returns a pretty-printed rich string for the job status"""
        elapsed = self.get_elapsed_time_str(extra_verbose=extra_verbose)
        if len(elapsed) > 0 and self.status in {
            JobStatus.RUNNING,
            JobStatus.DONE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            end_time_str = ""
            if (
                self.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
                and self._end_time is not None
                and self._end_time > 0
            ):
                end_time_str = (
                    f" [bright_black][Ended: {readable_timestamp(self._end_time)}][/bright_black]"
                )
            return f"{self.id} - {self.status.formatted_str} [bright_black][{elapsed}][/bright_black]{end_time_str}"
        return f"{self.id} - {self.status.formatted_str}"

    @property
    def start_time(self) -> float:
        """Shortcut access to start time"""
        return -1.0 if self._start_time is None else self._start_time

    @property
    @abstractmethod
    def end_time(self) -> float:
        """If the job is completed (succesfully or not) return endtime
        as a timestamp. Otherwise returns -1"""
        if self._end_time is not None:
            return self._end_time
        if self.status.completed:
            # self.end_time = ....
            raise NotImplementedError
        return -1

    def get_elapsed_time_str(
        self,
        end_time: float | None = None,
        extra_verbose: bool = False,
        formatted: bool = True,
    ) -> str:
        """Returns the pretty-printed elapsed time between `end_time` and the start
        time of the current job.

        :param end_time: Compute elapsed time between `end_time` and this job's start time

        :return: A string capturing the elapsed time in readable format
        """
        metadata = ""
        has_end_time: bool = True
        if end_time is None or end_time < 0:
            end_time = self.end_time
            if end_time is None or end_time < 0:
                end_time = time.time()
                has_end_time = False
        assert end_time is not None
        if end_time > 0:
            if extra_verbose and has_end_time and formatted:
                metadata += (
                    f"[bright_black]Last modified: {readable_timestamp(end_time)}[/bright_black]\n"
                )
            if self.start_time > 0:
                if formatted:
                    metadata += (
                        "[bright_black]Elapsed: "
                        f"{readable_elapsed(end_time - self.start_time)}[/bright_black]"
                    )
                else:
                    metadata += readable_elapsed(end_time - self.start_time)
        return metadata
