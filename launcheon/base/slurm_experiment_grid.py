"""Instantiates `Job`, `Experiment` and `ExperimentGrid` for a SLURM cluster"""

import os
import shlex
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import rich

from launcheon.base.slurm_utils import (
    get_jobid_from_slurm_output,
    parse_job_status,
    parse_slurm_timestamp,
    read_relative_symlink,
    wrap_in_multinode_for_slurm,
)
from launcheon.experiment import Experiment
from launcheon.experiment_grid import ExperimentGrid
from launcheon.global_variables import (
    SLURM_LOGDIR_FOR_JOB_ARRAY,
    SLURM_LOGFILE,
    SLURM_LOGFILE_FOR_JOB_ARRAY,
)
from launcheon.job import Job, JobStatus
from launcheon.parser_format import SrunArgsParserFormat
from launcheon.utils import format_kwargs, print_error, readable_timestamp


class SlurmJob(Job):
    """A Slurm Job. Compared to the base Job class, this contains two
    new fields (`nodelist` and `exit_code`), mainly  used to display
    extra information for failed jobs"""

    def __init__(
        self,
        jobid: str | None = None,
        status: JobStatus = JobStatus.UNSCHEDULED,
        start_time: float | None = None,
        end_time: float | None = None,
        nodelist: str | None = None,
        exit_code: str | None = None,
    ):
        super().__init__(jobid=jobid, status=status, start_time=start_time, end_time=end_time)
        self.nodelist = nodelist
        self.exit_code = exit_code

    @property
    def end_time(self) -> float:
        """If the job is completed (succesfully or not) return endtime
        as a timestamp. Otherwise returns -1"""
        if self._end_time is not None:
            return self._end_time
        if self.status.completed and self.id is not None:
            try:
                out = (
                    subprocess.check_output(
                        f"sacct -j {self.id} --format=End | sed -n '3 p'", shell=True
                    )
                    .decode("utf-8")
                    .replace("\n", "")
                    .strip()
                )
                self._end_time = parse_slurm_timestamp(out)
                return self._end_time
            except subprocess.CalledProcessError:
                pass
        return -1

    def get_full_status_str(self, extra_verbose: bool = False) -> str:
        """Return pretty-printed rich string for the job status (used in monitor)"""
        s = f"{self.id} - {self.status.formatted_str}"
        if (
            self.status in {JobStatus.FAILED, JobStatus.DONE, JobStatus.RUNNING}
            and self.nodelist is not None
        ):
            s += (
                ""
                if self.exit_code is None or not extra_verbose
                else f" [bold cyan]exit code[/bold cyan]: {self.exit_code}"
            ) + f"[bright_black] on[/bright_black] [honeydew2]{self.nodelist}[/honeydew2]"

        elapsed = self.get_elapsed_time_str(extra_verbose=extra_verbose)
        if len(elapsed) > 0 and self.status in {
            JobStatus.RUNNING,
            JobStatus.DONE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            s += f" [bright_black][{elapsed}][/bright_black]"
        if (
            self.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
            and self._end_time is not None
            and self._end_time > 0
        ):
            s += f" [bright_black][Ended: {readable_timestamp(self._end_time)}][/bright_black]"

        return s


class SlurmExperiment(Experiment):
    """Experiment with SLURM utils, in particular for job arrays support"""

    @classmethod
    def __base_log_file__(cls) -> str:
        """base Slurm log file name for this experiment.

        Note that we do not include the JOBID in the slurm log file to make it easoer to
        access the jobs from the experiment name only (otherwise we would first need
        to parse the jobd ID)
        """
        return SLURM_LOGFILE

    @property
    def job_array_log_file(self) -> str:
        """For job arrays, we are restricted in how/where we can name the output file.
        Thus, when running with job arrays, self.log_file will instead be a symbolic link
        to the true slurm log file contained in self.job_array_log_file"""
        return os.path.join(
            self.log_dir_root,
            SLURM_LOGDIR_FOR_JOB_ARRAY,
            SLURM_LOGFILE_FOR_JOB_ARRAY.format(base_name=self.base_name, exp_idx=f"{self.idx}"),
        )

    def resolve_log_symlink(self, verbose: bool = True):
        log_path = Path(self.log_file)
        if log_path.is_symlink():
            abs_path = log_path.resolve()
            if abs_path.exists():
                tmp_path = log_path.rename(log_path.with_suffix(".tmp" + log_path.suffix))
                abs_path.rename(log_path)
                tmp_path.unlink()
                if verbose:
                    self.print_as_header()
                    print(f"Rooted symlink in {log_path}")

    def remove(self) -> None:
        """Remove this experiment's directory and logs (including the corresponding true
        SLURM log file if logfile is a symlink)"""
        if os.path.islink(self.log_file):
            tgt = read_relative_symlink(self.log_file)
            if os.path.exists(tgt):
                os.remove(tgt)
        super().remove()

    def cancel(self) -> None:
        """Cancel corresponding SLURM experiment"""
        target_jobid = self.jobid
        if target_jobid is not None:
            try:
                out = subprocess.check_output(
                    f"scancel {target_jobid} --verbose",
                    shell=True,
                )
                print(out.decode("utf-8").replace("\n", ""))
            except subprocess.CalledProcessError:
                # typically will fail because of invalid job ID (a job that is not running)
                pass

    def update_job(self) -> None:
        """Update the job status of the corresponding SLURM experiment
        Note that this is superseeded by SlurmExperimentGrid.update_jobs for
        efficiency reason for most operations at the grid level.
        """
        if self._job is None or self._job.id is None:
            self._job = None
            jobid = self.jobid
        else:
            jobid = self._job.id
            self._job = None
        start_time: float | None = None
        nodelist: str | None = None

        if jobid is not None:
            # sacct: for running or completed jobs
            status = (
                subprocess.check_output(
                    f"sacct -j {jobid} --format=State | sed -n '3 p'",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .replace("\n", "")
                .strip()
            )

            # For running/completed job, we also record the elapsed time
            if status.lower() in ["running", "completed"]:
                start_time = parse_slurm_timestamp(
                    subprocess.check_output(
                        f"sacct -j {jobid} --format Start | sed -n '3 p'",
                        shell=True,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .replace("\n", "")
                    .strip()
                )

            # For failed job, we will also display the node they ran on
            nodelist = (
                subprocess.check_output(
                    f"sacct -j {jobid} --format=NodeList%100 | sed -n '3 p'",
                    shell=True,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .replace("\n", "")
                .strip()
            )

            # squeue: for pending jobs
            if len(status) == 0:
                status = (
                    subprocess.check_output(
                        f"squeue -j {jobid} -O State | tail -1",
                        shell=True,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .replace("\n", "")
                    .strip()
                )

            self._job = SlurmJob(
                jobid=jobid,
                # if status can not be parsed (e.g. sacct or squeue
                # error) this will returns NOT_FOUND job status
                status=parse_job_status(status),
                start_time=start_time,
                nodelist=nodelist,
            )
        else:
            # Unsubmitted job
            self._job = SlurmJob(jobid=None, status=JobStatus.UNSCHEDULED)


class SlurmExperimentGrid(ExperimentGrid):
    """Base experiment manager class for SLURM experiments"""

    def __init__(
        self,
        *args: Any,
        partition: str | None = "kyutai",
        work_dir: str = f"/home/{os.environ.get('USER')}",
        slurm_header: str | None = None,
        container_image: str | None = None,
        container_mounts: str | None = None,
        as_slurm_array: bool = True,
        **kwargs: Any,
    ) -> None:
        """Experiment grid customized for usage with SLURM.
        Adds a few extra args compared to the base `ExperimentGrid`

        :param partition: Slurm partition
        :param work_dir: Base dir to start the experiment from
        :param slurm_header: Extra commands added to the SLURM script. E.g. can
            be used to setup some environment variables
        :param container_image: Optional container image for SLURM
        :param container_mounts: Which directories to mount inside the container
        """
        _experiment_constructor: Callable = SlurmExperiment
        if "_experiment_constructor" in kwargs:
            # 😠 Danger zone 😠 : using a custom experiment constructor
            # instead of default `SlurmExperiment`
            assert issubclass(kwargs["_experiment_constructor"], SlurmExperiment)
            _experiment_constructor = kwargs.pop("_experiment_constructor")
            if "_experiment_constructor_kwargs" in kwargs:
                _experiment_constructor = partial(
                    _experiment_constructor,
                    **kwargs.pop("_experiment_constructor_kwargs"),
                )
        super().__init__(
            *args,
            _experiment_constructor=_experiment_constructor,  # type: ignore
            **kwargs,  # type: ignore
        )

        self.partition = partition
        self.container_image = container_image
        self.container_mounts = container_mounts
        self.work_dir = work_dir
        self.slurm_header = slurm_header
        # By default, will submit jobs as Job Arrays
        self.as_slurm_array = as_slurm_array
        self.init_sbatch_kwargs()

    def __iter__(self) -> Iterator[SlurmExperiment]:
        """Only overwriting this to get proper type hints"""
        yield from self._exps_dict.values()  # type: ignore

    def __getitem__(self, idx: int) -> SlurmExperiment:
        """Only overwriting this to get proper type hints"""
        return self._exps_dict[idx]  # type: ignore

    def init_sbatch_kwargs(self):
        # for multinode launch, we add an additional srun inside the sbatch command
        # to setup the parallel environment
        srun_kwargs: dict[str, Any] = {}

        # Set up resources Resources
        # sbatch: Main global resources
        self.sbatch_kwargs: dict[str, Any] = {
            "nodes": self.num_nodes,
            "ntasks-per-node": 1,
            "gpus-per-task": self.num_gpus,
            "cpus-per-task": self.num_cpus_per_gpu * max(1, self.num_gpus),
        }
        if self.partition is not None:
            self.sbatch_kwargs["partition"] = self.partition

        # Container options
        if self.container_image is None:
            (srun_kwargs if self.num_nodes > 1 else self.sbatch_kwargs)["chdir"] = self.work_dir
        else:
            upd = {
                "container-mount-home ": None,
                "container-workdir": self.work_dir,
                "container-image": self.container_image,
            }
            if self.container_mounts is not None:
                upd["container-mounts"] = self.container_mounts
            (srun_kwargs if self.num_nodes > 1 else self.sbatch_kwargs).update(upd)

        self.srun_wrapper = ""
        if len(srun_kwargs):
            formatted_srun_kwargs = format_kwargs(
                srun_kwargs,
                parser_format=SrunArgsParserFormat(),
            )
            self.srun_wrapper = f"srun -K1 {formatted_srun_kwargs} "

    def noarray(self) -> Any:
        """Chainable command that will treat the experiment grid as set of
        individual SLURM jobs rather than a job array"""
        self.as_slurm_array = False
        return self

    def asarray(self) -> Any:
        """Chainable command that will treat the experiment grid as
        a job array rather than individual SLURM jobs"""
        self.as_slurm_array = True
        return self

    def get_full_exp_cmd_with_header(
        self, exp: SlurmExperiment, local: bool = False
    ) -> tuple[str, str]:
        """Return full command with potential multinode wrapping"""
        header = ""
        cmd = exp.get_cmd()
        # for multinode non-interactive jobs; Note that the multinode wrapping is
        # applied to the raw experiment command, before wrapping it with micromamba/git
        if self.num_nodes > 1 and not local:
            hostfiles_dir = os.path.join(
                self.log_dir_root, SLURM_LOGDIR_FOR_JOB_ARRAY, "hostfiles_tmp"
            )
            os.makedirs(hostfiles_dir, exist_ok=True)
            header, cmd = wrap_in_multinode_for_slurm(
                cmd=cmd,
                launcher=self.dist_launcher,
                num_gpus_per_node=self.num_gpus,
                tmp_dir=hostfiles_dir,
            )
        cmd = self.wrap_exp_cmd(cmd, local=local)
        if not local and len(self.srun_wrapper) > 0:
            # wrap in bash so that the full command (incl. git `cd` and micromamba) runs
            # inside each srun task
            cmd = f"{self.srun_wrapper}bash -c {shlex.quote(cmd)}"
        return header, cmd

    def get_full_exp_cmd(self, exp: SlurmExperiment, local: bool = False) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return full command with potential multinode wrapping"""
        header, cmd = self.get_full_exp_cmd_with_header(exp, local=local)
        if len(header.strip()) > 0:
            return f"{header}\n{cmd}"
        return cmd

    def __update_jobs__(self, notset_only: bool = False) -> None:
        """For performance reason, it's faster to update jobs in batch
        (through a single call to sacct), so whenever we operate at the
        experiment grid level, we defer to this function instead of the
        individual experiments' update_job method"""
        # update jobids
        jobids: dict[str, int] = {}
        for exp in self:
            if not notset_only or exp._job is None:
                exp._job = None
                jobid = exp.jobid
                if jobid is None:
                    exp._job = SlurmJob(jobid=None, status=JobStatus.UNSCHEDULED)
                else:
                    jobids[str(jobid)] = exp.idx

        # First: call to squeue (for running and pending jobs)
        if len(jobids) > 0:
            try:
                # -r is important to get proper status for each job in the array
                out = subprocess.check_output(
                    f'squeue -r -j {",".join(jobids.keys())} -o "%.15A %.15i %.10T %.30S %.50N"',
                    shell=True,
                    stderr=subprocess.DEVNULL,
                ).decode("ascii")

                for line in out.splitlines()[1:]:
                    aux = line.split()
                    if len(aux) == 5:
                        jobid, array_jobid, status, start_timestamp, nodelist = aux
                    else:
                        # pending jobs don't have a nodelist assigned yet
                        jobid, array_jobid, status, start_timestamp = aux
                        nodelist = None
                    if array_jobid != jobid:
                        jobid = array_jobid
                    parsed_status = parse_job_status(status)
                    if jobid in jobids and parsed_status != JobStatus.SACCTERROR:
                        self[jobids[jobid]]._job = SlurmJob(
                            jobid=jobid,
                            status=parsed_status,
                            start_time=parse_slurm_timestamp(start_timestamp),
                            nodelist=nodelist,
                        )
                        del jobids[jobid]
            except subprocess.CalledProcessError:
                pass

        # Second: call to sact (for other statuses)
        if len(jobids) > 0:
            try:
                # Note that the format "State%10" is optimized to not display
                # the full "Cancelled by 1+" and avoid an error when parsing the
                # line because of the extra whitespaces
                out = subprocess.check_output(
                    f"sacct -j {','.join(jobids.keys())} "
                    '-o "JobID%15,State%10,NodeList%100,ExitCode%15,Start,End"',
                    shell=True,
                    stderr=subprocess.DEVNULL,
                ).decode("ascii")

                for line in out.splitlines()[2:]:
                    try:
                        jobid, status, nodelist, exit_code, start_time, end_time = line.split()
                        parsed_status = parse_job_status(status)
                        if jobid in jobids and parsed_status != JobStatus.SACCTERROR:
                            self[jobids[jobid]]._job = SlurmJob(
                                jobid=jobid,
                                status=parsed_status,
                                exit_code=exit_code.replace("\n", ""),
                                nodelist=nodelist,
                                start_time=parse_slurm_timestamp(start_time),
                                end_time=parse_slurm_timestamp(end_time),
                            )
                            del jobids[jobid]
                    # typically this can be raise for jobs that were cancelled
                    # while pending: nor sacct nor squeue seem to give valuable outputs
                    except ValueError:
                        continue
            except subprocess.CalledProcessError:
                # sacct not found
                for jobid in jobids:
                    if (job := self[jobids[jobid]]._job) is not None:
                        job.status = JobStatus.SACCTERROR
                jobids = {}

        # If anything is left: marked as unscheduled
        for exp_idx in jobids.values():
            self[exp_idx]._job = SlurmJob(jobid=None, status=JobStatus.UNSCHEDULED)

    def generate_slurm_script(
        self,
        max_parallel_array: int | None = None,
        write_dir: str | None = None,
        extra_sbatch_options: dict[str, str] | None = None,
        verbose: bool = True,
        exp_idx: int | None = None,
    ) -> str:
        """Generate the slurm script for the array of experiments

        :param max_parallel_array: Max number of jobs of the array we can run simulatenously
        :param write_dir: Where to write the generated slurm scripts.
            Uses the experiment's `log_dir` if `write_dir` is not given
        :param extra_sbatch_options: Extra flags passed to sbatch (e.g. nodelist, begin etc.)

        :return: Path to the generated SLURM batch script
        """
        write_dir = write_dir or self.log_dir_root
        assert write_dir is not None
        os.makedirs(write_dir, exist_ok=True)

        out_file = os.path.join(write_dir, f"slurm_{self.base_name}.sh")

        # Sbatch main options
        out_str = "#!/bin/bash\n\n# SBATCH options\n"
        for k, v in self.sbatch_kwargs.items():
            if v is None:
                out_str += f"#SBATCH --{k}\n"
            else:
                out_str += f"#SBATCH --{k}={v}\n"

        if extra_sbatch_options is not None:
            for k, v in extra_sbatch_options.items():
                if v is not None:
                    if isinstance(v, bool) and v:
                        out_str += f"#SBATCH --{k}\n"
                    else:
                        out_str += f"#SBATCH --{k}={v}\n"

        # Job Array settings
        if self.as_slurm_array:
            out_str += f"#SBATCH --array={','.join(str(exp.idx) for exp in self)}"
            out_str += "" if max_parallel_array is None else f"%{max_parallel_array}"
            # Naming
            out_str += f"\n#SBATCH --job-name={self.base_name}_{self.__readable_range__()}"
            # Job array outputs
            slurm_out_file_format = os.path.join(
                self.log_dir_root,
                SLURM_LOGDIR_FOR_JOB_ARRAY,
                SLURM_LOGFILE_FOR_JOB_ARRAY.format(base_name=self.base_name, exp_idx="%a"),
            )
            out_str += f"\n#SBATCH --output={slurm_out_file_format}"
        else:
            assert exp_idx is not None
            exp = self._exps_dict[exp_idx]
            # Naming for individual job
            if not exp.use_hashed_dirname:
                out_str += f"#SBATCH --job-name={exp.name}\n"
            else:
                out_str += f"#SBATCH --job-name={exp.hashed_name}_{self.base_name}\n"
            out_str += f"#SBATCH --output={exp.log_file}\n"

        # Add header
        if self.slurm_header is not None:
            out_str += f"\n\n{self.slurm_header}"

        # Finally, it's time to write the commands
        if self.as_slurm_array:
            # Generate cases expression mapping job index in array to correct command
            for idx, exp in enumerate(self):
                # If an array, at the end of the job, we want to resolve the symlink
                # of slurm.out so that the real slurm output is placed in the
                # experiment's directory
                src = os.path.abspath(exp.job_array_log_file)
                tgt = os.path.abspath(exp.log_file)
                temp = f"{os.path.abspath(exp.log_file)}.tmp"
                q_src, q_tgt, q_temp = shlex.quote(src), shlex.quote(tgt), shlex.quote(temp)
                trap_inner = f"mv {q_tgt} {q_temp} && cp {q_src} {q_tgt} && rm {q_temp}"
                if self.num_nodes > 1:
                    # this trap replaces the one defined in the multinode header
                    trap_inner += '; rm -f "$HOSTFILE_PATH"'
                trap_cmd = f"trap {shlex.quote(trap_inner)} EXIT SIGTERM"
                # setup git env if needed
                # generate command with optional multinode wrapping
                header, wrapped_cmd = self.get_full_exp_cmd_with_header(exp)
                wrapped_cmd = wrapped_cmd.rstrip()
                if not wrapped_cmd.endswith(";"):
                    wrapped_cmd = wrapped_cmd + " ;"
                if idx == 0:  # Prepend common header only once
                    out_str += "\n" + header
                    out_str += "\n# Command Selection"
                    out_str += "\ncase $SLURM_ARRAY_TASK_ID in\n"

                out_str += f"""
    {exp.idx})
    {trap_cmd}
    {wrapped_cmd}
    ;;
    """
            # last case (error)
            out_str += """
    *)
    echo "SLURM: Could not a find command for job $SLURM_ARRAY_TASK_ID :("
    ;;
    esac
    """
        else:
            assert exp_idx is not None
            out_str += "\n# Command\n"
            out_str += self.get_full_exp_cmd(self._exps_dict[exp_idx])  # type: ignore

        # Write
        with open(out_file, "w") as f:
            f.write(out_str)
        if verbose:
            print(f"Wrote SLURM script in {out_file}")
        return out_file

    def __submit_exps__(
        self,
        *_args: Any,
        dry_run: bool = False,
        resume: bool = False,
        overwrite: Any = None,
        keep_slurm_script: bool = False,
        # for job arrays only
        max_parallel: int | None = None,
        max_gpus: int | None = None,
        quiet: bool = False,
        # Extra SLURM options directly propagated to SBATCH
        **_mismatched_flags: Any,
    ) -> int | None:
        """Submit all experiments currently selected/filtered in the experiment grid

        :param dry_run: Display the SLURM script to be run, without actually submitting
            it or affect experiment directory
        :param resume: If given, this will submit an experiment while keeping the previous
            log directory
        :param overwrite: If this is given, erase the existing logdir (on user input) and rerun
            experiments. If you want to auto-accept without user input, use `--overwrite brute`
        :param keep_slurm_script: For debugging, this will keep the generated sbatch script
            after submitting them
        :param max_parallel: Only used if the experiment grid is in job array model.
            This defines the maximum number of jobs of the grid that will be run in prallel
        :param max_gpus: Only used if the experiment grid is in job array model.
            This is used to specify max_parallel using a number of GPUs rather than jobs
        :param nodelist: Same as SBATCH's nodelist (set possible submission hosts)
        :param exclude: Same as SBATCH's exclude (set forbidden submission hosts)
        :param begin: Same as SBATCH's begin (schedules the job but set min begin time
            for the job)
        :param dependency: Same as SBATCH's dependency (only start the job when certain
            dependencies have been met)
        :param time: Same as SBATCH's time (set a time limit for the job)
        """
        assert max_parallel is None or max_gpus is None, (
            "You can only specify one of `max_parallel` or `max_gpus`"
        )
        if max_gpus is not None:
            max_parallel = max_gpus // (self.num_gpus * self.num_nodes)

        # No need for an array for single experiments
        if len(self._exps_dict) == 1:
            self.noarray()

        # SBATCH options
        extra_sbatch_options = {}
        for k in ["nodelist", "exclude", "begin", "dependency", "time"]:
            extra_sbatch_options[k] = _mismatched_flags.pop(k.replace("_", "-"), None)

        if dry_run:
            extra_sbatch_options["test-only"] = True

        # Strict check on the flag to avoid submitting experiment by mistake !
        if len(_args) > 0 or len(_mismatched_flags) > 0:
            rich.print(
                "[red]Error[/]: Found unexpected flags to the `submit` command: ",
                ", ".join(_args) + "," if len(_args) > 0 else "",
                ", ".join(_mismatched_flags.keys()),
            )
            return None

        # For job array, check if we need to reassign existing experiments log file.
        # This can happen if we have modified the grid (e.g. adding a hyperparameter):
        # this will shift the experiment indices, hence it will affect which slurm
        # job array log file they correspond to
        if self.as_slurm_array:
            for exp in self:
                if os.path.islink(exp.log_file):
                    try:
                        target_slurm_logfile = read_relative_symlink(exp.log_file)
                        if os.path.abspath(target_slurm_logfile) != os.path.abspath(
                            exp.job_array_log_file
                        ):
                            rich.print(
                                "[red]Error[/]: Experiment indices do not match"
                                " existing slurm log files."
                                " To fix it, run `reindex` before `submit`"
                            )
                            return None
                    except OSError:
                        pass

        # Filter out experiments
        if not self.__filter_exps_to_submit__(
            overwrite=overwrite, resume=resume, dry_run=dry_run, quiet=quiet
        ):
            return None
        if len(self._exps_dict) == 0:
            if not quiet:
                rich.print(
                    f"[bold red]ERROR:[/bold red] No experiments"
                    f" left to run in grid {self.base_name}"
                )
            return None

        if not dry_run:
            self.setup_git_clone()

        # Submit
        # (Individual) Submit all experiments as individual jobs
        # For max_parallel, there is no direct equivalent for individual jobs,
        # but we can use SLURM dependencies to achieve a similar effect
        jobids_lst = []
        if not self.as_slurm_array:
            num_launched = 0
            for num_launched, exp in enumerate(self):
                # Break if max_parallel is set
                if (
                    max_parallel is not None
                    and num_launched >= max_parallel
                    and len(jobids_lst) > 0
                ):
                    # Replace dependency by first jobs
                    extra_sbatch_options["dependency"] = f"afterany:{jobids_lst.pop(0)}"

                # Otherwise, submit the experiment
                if not dry_run:
                    os.makedirs(exp.log_dir, exist_ok=True)

                slurm_script = self.generate_slurm_script(
                    write_dir=tempfile.mkdtemp(prefix="launcheon_dryrun_") if dry_run else None,
                    extra_sbatch_options=extra_sbatch_options,
                    verbose=keep_slurm_script,
                    exp_idx=exp.idx,
                )

                try:
                    if dry_run:
                        exp.print_as_header()
                        rich.print("[cyan]SLURM script:[/cyan]", flush=True)
                        with open(slurm_script, "r") as slurm_file:
                            print(slurm_file.read())
                        rich.print("\n  [cyan]SBATCH --test-only output:[/cyan]")
                        # Note: sbatch --test-only reports on stderr
                        res = subprocess.run(
                            ["sbatch", slurm_script], capture_output=True, check=True
                        )
                        print(res.stdout.decode("utf-8", errors="replace").strip())
                        print(res.stderr.decode("utf-8", errors="replace").strip())
                        if not keep_slurm_script:
                            os.remove(slurm_script)
                            os.rmdir(os.path.dirname(slurm_script))
                        # Nothing was submitted: do not register any job ID
                        continue
                    out = subprocess.check_output(
                        ["sbatch", slurm_script],
                        stderr=subprocess.PIPE,
                    )
                    jobid = get_jobid_from_slurm_output(out.decode("utf-8", errors="replace"))
                    # for individual experiments we directly record the job id
                    self.register_exp_info(exp, str(jobid))
                    jobids_lst.append(jobid)

                    if not keep_slurm_script:
                        os.remove(slurm_script)
                except subprocess.CalledProcessError as e:
                    if e.stdout:
                        print(e.stdout.decode("utf-8", errors="replace"))
                    if e.stderr:
                        print(e.stderr.decode("utf-8", errors="replace"))
                    raise

            # Return the number of jobs launched
            return num_launched + 1

        # Otherwise submit experiments as a job array
        # Generate job array config and slurm script
        slurm_script = self.generate_slurm_script(
            write_dir=tempfile.mkdtemp(prefix="launcheon_dryrun_") if dry_run else None,
            max_parallel_array=max_parallel,
            verbose=False,
            extra_sbatch_options=extra_sbatch_options,
            exp_idx=None,
        )

        # create dir for slurm outputs
        slurm_output_dir_for_job_array = os.path.join(self.log_dir_root, SLURM_LOGDIR_FOR_JOB_ARRAY)
        if not dry_run:
            os.makedirs(slurm_output_dir_for_job_array, exist_ok=True)

        # create dir for each experiment and symlink to corresponding slurm out
        # file from the job array
        if not dry_run:
            for exp in self:
                os.makedirs(exp.log_dir, exist_ok=True)

                if not os.path.islink(exp.log_file):
                    os.symlink(
                        src=os.path.relpath(exp.job_array_log_file, os.path.dirname(exp.log_file)),
                        dst=exp.log_file,
                    )

        # launch
        try:
            if dry_run:
                rich.print("\n[cyan]SLURM script:[/cyan]", flush=True)
                with open(slurm_script, "r") as slurm_file:
                    print(slurm_file.read())

                rich.print("[cyan]SBATCH --test-only output:[/cyan]")
                res = subprocess.run(["sbatch", slurm_script], capture_output=True, check=True)
                print(res.stdout.decode("utf-8", errors="replace").strip())
                print(res.stderr.decode("utf-8", errors="replace").strip())
            else:
                out = subprocess.check_output(["sbatch", slurm_script])
                print()
                if not quiet:
                    out_print = out.decode("utf-8").replace("\n", "")
                    rich.print(f"[green]{out_print}[/]")

                # store each experiment job id in the respective experiment's directory
                jobid = get_jobid_from_slurm_output(out.decode("utf-8", errors="replace"))
                for exp in self:
                    # for arrays we write the jobid + array index
                    self.register_exp_info(exp, f"{jobid}_{exp.idx}")
                    # Note: we could rename the jobs in the array but it loses
                    # some nice properties of the array e.g. having a unique job
                    # subprocess.run(
                    #    f"scontrol update job={jobid}_{exp.idx} JobName={exp.name}",
                    #    shell=True,
                    # )
        except subprocess.CalledProcessError as e:
            if e.stdout:
                print(e.stdout.decode("utf-8", errors="replace"))
            if e.stderr:
                print(e.stderr.decode("utf-8", errors="replace"))
            raise

        # clean up
        if not keep_slurm_script:
            os.remove(slurm_script)
            if dry_run:
                os.rmdir(os.path.dirname(slurm_script))
        else:
            rich.print(f"\nSlurm script for the job array written in [bold green]{slurm_script}[/]")

        # return number of experiments submitted
        # in array mode, we always submit *all* experiments, but with a varying
        # number of `max_parallel jobs, depending on the cluster status
        return len(self)

    def resolve_log_symlink(self, verbose: bool = True):
        for exp in self:
            exp.resolve_log_symlink(verbose=verbose)

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
        ] = "status",
        num_lines: int | None = None,
    ) -> None:
        """Adds an extra `slurm_cmd` print option"""

        if tgt == "slurm_cmd":
            formatted_batch_kwargs = format_kwargs(self.sbatch_kwargs, SrunArgsParserFormat())
            for exp in self:
                color = "deep_pink4" if exp.has_launched else "bright_black"
                rich.print(f"\n[{color}]{exp.idx:02d} - {exp.name}[/{color}]")
                print(
                    f"sbatch {formatted_batch_kwargs}"
                    f" --job-name {shlex.quote(exp.hashed_name if exp.use_hashed_dirname else exp.name)}"
                    f" --output {shlex.quote(exp.log_file)}"
                    f" --wrap {shlex.quote(self.get_full_exp_cmd(exp, local=False))}"
                )

        else:
            super().print(tgt, num_lines=num_lines)

    def squeue(self, watch: int | None = None) -> None:
        """Display pretty printed squeue command for this experiment grid,
        and updates it every `watch` seconds"""
        jobids = [str(x) for exp in self for x in [exp.jobid] if x is not None]
        if len(jobids) == 0:
            rich.print("[orange]WARNING[/orange]: No scheduled jobs found for this experiment grid")
            return
        cmd = f'squeue --jobs {",".join(jobids)} -o "%.12i %.8P %.30j %.2t %.8M %.6C %.6D %R"'
        if watch is not None:
            cmd = f"watch -n {watch} '{cmd}'"
        subprocess.run(
            cmd,
            shell=True,
            check=False,
        )

    def reindex(self) -> None:
        """Reindex the existing output log files for a Job array if the
        experiment grid has changed. This is only needed because job array's output
        file can only been indexed by the integer ID of the array job, which is not
        as flexible as the indexing by name/kwargs done by the experiment grid"""
        # Check if any experiment is running
        bad = []
        for exp in self:
            if exp.job.status.scheduled:
                bad.append(exp.idx)
        if len(bad) > 0:
            print(f"{len(bad)} experiments are still running or pending. Can't reindex right now.")
            print(f"You can select these experiments with `select {','.join(str(x) for x in bad)}`")
            return

        # Collect the slurm log files which need to be moved, for the currently
        # selected experiments only; other experiments' log files are left untouched
        moves: list[tuple[Experiment, str, str]] = []
        for exp in self:
            # if logfile is not a symlink, this means the experiment was not launched as
            # a job array and/or it was already resolved, hence there is nothing to do
            if not os.path.islink(exp.log_file):
                continue
            old_slurm_logfile = os.path.abspath(read_relative_symlink(exp.log_file))
            new_slurm_logfile = os.path.abspath(exp.job_array_log_file)
            if old_slurm_logfile != new_slurm_logfile:
                moves.append((exp, old_slurm_logfile, new_slurm_logfile))

        if len(moves) == 0:
            rich.print("[green]Nothing to reindex[/green]")
            return

        # Never overwrite a log file which is not itself being moved (e.g. one belonging
        # to an experiment which is not in the current selection)
        moved_sources = {old for _, old, _ in moves}
        conflicts = [new for _, _, new in moves if os.path.exists(new) and new not in moved_sources]
        if len(conflicts) > 0:
            print_error(
                "Reindexing would overwrite the following slurm log files, which do not belong"
                " to any experiment in the current selection. Aborting.\n  "
                + "\n  ".join(conflicts)
            )
            return

        # Two-phase move through a staging directory to handle permutations of indices
        slurm_output_dir_for_job_array = os.path.join(self.log_dir_root, SLURM_LOGDIR_FOR_JOB_ARRAY)
        staging_dir = tempfile.mkdtemp(dir=slurm_output_dir_for_job_array, prefix=".reindex_")
        staged: list[tuple[Experiment, str | None, str, str]] = []
        for i, (exp, old, new) in enumerate(moves):
            staged_file = None
            if os.path.exists(old):
                staged_file = os.path.join(staging_dir, str(i))
                os.rename(old, staged_file)
            staged.append((exp, staged_file, old, new))

        for exp, staged_file, old, new in staged:
            if staged_file is not None:
                os.rename(staged_file, new)
            # replace old symlink by the new one
            os.remove(exp.log_file)
            os.symlink(src=os.path.relpath(new, os.path.dirname(exp.log_file)), dst=exp.log_file)
            rich.print(
                "Fixing mismatch in experiment indices (exp",
                exp.idx,
                f"links to [bright_cyan]{os.path.basename(old)}[/])",
            )
        os.rmdir(staging_dir)
