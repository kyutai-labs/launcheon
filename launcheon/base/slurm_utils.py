"""Utils specific to SLURM clusters"""

import os
import re
import shlex
from datetime import datetime

from launcheon.global_variables import DistLauncher
from launcheon.job import JobStatus


def read_relative_symlink(rel_symlink: str) -> str:
    """Retrieve the true target path of the given relative symbolic link

    :param: Symlink with relative paths

    :return: Absole path to the target of the symlink
    """
    return os.path.join(os.path.dirname(rel_symlink), os.readlink(rel_symlink))


def parse_slurm_timestamp(t: str) -> float:
    """Parse a SLURM-formatted timestamp

    :param t: Pretty-printed time string printed by SLURM (e.g. from squeue output)

    :return: Corresponding timestamp
    """
    try:
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return -1.0


def parse_job_status(status: str) -> JobStatus:
    """Parse the output of squeue/sacct into a proper JobStatus object"""
    status = status.lower().replace("+", "")
    if status == "completed":
        job_status = JobStatus.DONE
    elif status in ["failed", "running", "pending", "cancelled"]:
        job_status = getattr(JobStatus, status.upper())
    else:
        job_status = JobStatus.SACCTERROR
    return job_status


def get_jobid_from_slurm_output(output: str) -> int:
    """Parse the output of a sbatch submit command (e.g. "Submitted batch job 123",
    possibly followed by " on cluster X") to get the ID of the job submitted"""
    match = re.search(r"Submitted batch job (\d+)", output)
    if match is None:
        raise ValueError(f"Could not parse job ID from sbatch output: {output!r}")
    return int(match.group(1))


def __split_on_launcher__(cmd: str, launcher_cmd: str) -> tuple[str, str]:
    """Split the given command around the (first occurence of the) distributed launcher"""
    if launcher_cmd not in cmd:
        raise ValueError(
            f"Multinode jobs with this launcher expect the command to contain `{launcher_cmd}`"
            f" but got: {cmd}"
        )
    before, after = cmd.split(launcher_cmd, 1)
    return before, after


def wrap_in_multinode_for_slurm(
    cmd: str,
    launcher: DistLauncher,
    num_gpus_per_node: int = 8,
    tmp_dir: str = "/tmp",
) -> tuple[str, str]:
    """Wrap a command for multinode support for different common distributed launchers

    :param cmd: Base command, e.g. `deepspeed train.py --gloubi boulga
    :param launcher: Type of launcher to use
    :param srun_wrapper: srun command with options specified by the experiment grid e.g. container
    :param num_gpus_per_node: Number of GPUS requested per node (the wrapper assumes a uniform distribution)
    """
    hostfile_path = os.path.join(tmp_dir, "hostfile_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt")
    hostfile_path_placeholder = os.path.join(
        tmp_dir, "hostfile_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_placeholder.txt"
    )
    header = f"""
# Define multinode hostfile
export HOSTFILE_PATH="{hostfile_path}"
HOSTFILE_PATH_PLACEHOLDER="{hostfile_path_placeholder}"
srun bash -c '''echo $SLURM_NODEID $(hostname)''' > $HOSTFILE_PATH_PLACEHOLDER
sort -n $HOSTFILE_PATH_PLACEHOLDER 

sort -n $HOSTFILE_PATH_PLACEHOLDER | awk 'NF{{print $0 " slots={num_gpus_per_node}"}}' | cut -d' ' -f2- > $HOSTFILE_PATH

rm $HOSTFILE_PATH_PLACEHOLDER
echo 'Using hosts:'
cat $HOSTFILE_PATH

# Define master node address and port
export MASTER_ADDR=$(cat $HOSTFILE_PATH | head -n 1 | cut -f '1' -d ' ')
echo 'Master Node:' $MASTER_ADDR
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
echo 'Master Port:' $MASTER_PORT

# Always remove the hostfile on exit
trap 'rm $HOSTFILE_PATH' EXIT SIGTERM
"""

    # currently deepspeed launcher assumes we fully utilize each node
    if launcher == DistLauncher.DEEPSPEED:
        # Add extra flags to deepspeed
        before, after = __split_on_launcher__(cmd, "deepspeed")
        # SLURM_NODEID  needs to be wrapped under a basch script otherwise it will
        # resolve to the value in the main SBATCH env which is not what we want
        inner = f'{before}deepspeed --node_rank=$SLURM_NODEID "$@"'
        wrapped_cmd = f"bash -c {shlex.quote(inner)}"
        wrapped_cmd += (
            " - --no_ssh --hostfile $HOSTFILE_PATH --num_nodes $SLURM_NNODES"
            " --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
            f"{after}"
        )
        # wrapped_cmd = f"{wrapped_cmd} - {after}"
        return header, wrapped_cmd
    # Note: accelerate wrapper has not been tested yet
    if launcher == DistLauncher.ACCELERATE:
        # src: https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh
        header += f"\nexport GPUS_PER_NODE={num_gpus_per_node}\n"
        before, after = __split_on_launcher__(cmd, "accelerate launch")
        inner = f'{before}accelerate launch --machine_rank=$SLURM_NODEID "$@"'
        wrapped_cmd = f"bash -c {shlex.quote(inner)}"
        wrapped_cmd += (
            f" - --num_machines $SLURM_NNODES --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) "
            f"--rdzv-backend c10d --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT{after}"
        )
        return header, wrapped_cmd
    # Note: torchrun wrapper has not been tested yet
    if launcher == DistLauncher.TORCHRUN:
        # src: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh#L17-L23
        before, after = __split_on_launcher__(cmd, "torchrun")
        wrapped_cmd = f"bash -c {shlex.quote(before + 'torchrun "$@"')}"
        wrapped_cmd += (
            f" - --nnodes $SLURM_NNODES --nproc-per-node 'gpu' --rdzv-id $SLURM_JOB_ID "
            f"--rdzv-backend c10d --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT{after}"
        )
        return header, wrapped_cmd
    # Jax wrapper: nothing to do, how nice
    return "", cmd
