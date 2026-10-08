"""Useful global variables"""

import operator
from enum import Enum, unique

from rich.box import Box  # type: ignore
from rich.color import ANSI_COLOR_NAMES

# When experiment job is launched, we saved all its keyword arguments,
# as well as the jobID in the exp directory, as a file named `EXP_INFO_FILE`
EXP_INFO_FILE = ".launcheon.json"

# Default value displayed when an experiment is missing a metric
DEFAULT_METRIC_VALUE = "N/A"


# Mapping for the `filter` subcommand
STRING_TO_FILTER_OP = {
    "=": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">=": operator.ge,
    ">": operator.gt,
}

# Rich format for the monitor table
RICH_CUSTOM_BOX = Box("    \n    \n ══ \n  │ \n ─┼ \n ─┼ \n  │ \n ── \n")

# Highlighting colors for tables
# generated with print(seaborn.color_palette("RdYlGn", 10).as_hex())
HIGHLIGHT_COLORS = [
    "#d22b27",
    "#ee613e",
    "#fa9b58",
    "#fece7c",
    "#fff1a8",
    "#eef8a8",
    "#c7e77f",
    "#93d168",
    "#57b65f",
    "#17934e",
]

# Convenient list of color names
PRETTY_TABLE_COLORS = [k for k in ANSI_COLOR_NAMES if "gray" not in k and "grey" not in k]

# Sanity check for keys given in the YAML API
ALLOWED_KEYS_IN_GLOBAL_CONFIG = [
    "base_name",
    "base_cmd",
    "log_dir",
    "use_hash_in_dirnames",
    "micromamba_env",
    "parser_format",
    "expand_kwargs_in_name",
    "expand_groups_in_name",
    "include_common_args_in_name",
    "override_log_file",
    "locked",
    "verify_at_submit",
    "git_sync",
    "git_repo",
    "git_branch",
    "git_commit",
    "git_clone_depth",
    "git_dirs_exclude",
    "git_setup",
    "global_seed",
    "set_exp_random_seed",
    "exp_seed_min",
    "exp_seed_max",
    "set_exp_random_port",
    "exp_port_min",
    "exp_port_max",
    "skip_sanity_checks",
]

ALLOWED_KEYS_IN_CLUSTER_GLOBAL_CONFIG = {
    "all": ["type", "num_gpus", "num_cpus_per_gpu", "num_nodes"],
    "slurm": [
        "partition",
        "work_dir",
        "container_image",
        "container_mounts",
        "as_slurm_array",
    ],
}

# For SLURM Experiment Grids
# For job arrays, we are restricted in how/where we can name the output file.
# Thus, when running with job arrays, the true log files written by SLURM are saved in
# SLURM_LOGDIR_FOR_JOB_ARRAY, while self.log_file is only a symbolic link
SLURM_LOGFILE = "slurm.out"
SLURM_LOGDIR_FOR_JOB_ARRAY = "slurm_output_logs"
SLURM_LOGFILE_FOR_JOB_ARRAY = "slurm_array_{base_name}_{exp_idx}.out"


@unique
class DistLauncher(Enum):
    """See slurm_utils/wrap_in_multinode_for_slurm for example usage"""

    NONE = "none"
    AUTO = "auto"
    DEEPSPEED = "deepspeed"
    ACCELERATE = "accelerate"
    TORCHRUN = "torchrun"
    JAX = "jax"
