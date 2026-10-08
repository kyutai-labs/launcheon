"""A simple experiment grid sweep showcasing several features such as:

* `kwargs_nicknames`: This allows to use "nicknames" for both flags names
  and their values rather than their full version. Can be useful for things
  like very long paths for instance

* `kwargs_groups`
"""

import os

from rich.console import Console
from rich.table import Table

from launcheon import SlurmExperimentGrid

# Base name prefix for every experiment in the grid
base_experiment_name = "my_little_experiment"

# Root directory where logs are written.
log_dir = f"/home/{os.environ.get('USER')}/launcheon_test"

# Environment and resources options
micromamba_env = "snake_env"
num_gpus = 2
num_nodes = 1
num_cpus_per_gpu = 8

# Base command to run, shared by all experiments
base_cmd = "python my_little_script.py"

# Feature 1: Kwargs Nicknames
# short nicknames for long arguments
# Maps a nickname to the real string it should resolve to
kwargs_nicknames = {
    "bs": "batch_size",
    "7b": "/very/long/path/to/7b/model",
    "3b": "/very/long/path/to/3b/model",
    "aug": "data_augmentation",
}


# Note: to avoid introducing duplicates by mistakes, it is
# recommended to use the `dict` constructor rather than `{ }` when possible
# Alternatively, the YAML API also provides stricter checks
sweep = dict(
    # simple grid search sweep
    lr=[1e-3, 1e-5, 1e-7],
    weight_decay=[0.1, 0.0],
    # singleton arguments
    # won't be used in experiment naming
    dataset="imagenet",
    # however, if given as a list, it will be used in experiment naming even
    # though all experiments have the same value
    aug=["randaugment"],
    # a boolean value
    clip_grad=True,
    # Advamced feature: Launcheon secret keys
    # These will be resolved on-the-fly at experiment's creation
    # Because it is given as a singleton list, seed will be integrated in the experiment's name
    # in contrast, port, tb_dir and output dir will not be added to the name
    seed=["{{launcheon.exp.seed}}"],
    ssh_port="{{launcheon.exp.port}}",
    main_logs_dir="{{launcheon.exp.log_dir}}",
    tb_dir="/my_other_path/tb_logs/{{launcheon.exp.name}}",
    # Advanced feature: kwargs groups
    # the experiment naming will use the format "arch=3b" but the actual kwargs in the nested dict
    # will be the ones used when generating the command
    arch={
        "3b": {"ckpt_dir": "3b", "bs": 1024, "width": 2048, "num_layers": 12},
        "7b": {"ckpt_dir": "7b", "bs": 256, "width": 4096, "num_layers": 24},
    },
)


class CustomExperimentGrid(SlurmExperimentGrid):
    """Since we are using fire, any method added to the experiment grid
    can be usage from the command line. This can be used to add custom plotting functions for instance
    """

    def print_table(self):
        table = Table(title=self.base_name)
        table.add_column("Exp ID")
        table.add_column("arch")
        table.add_column("lr")
        for exp in self:
            table.add_row(
                str(exp.idx),
                str(exp.get_value_from_anyquery("arch")),
                str(exp.get_value_from_anyquery("lr")),
            )
        Console().print(table)


if __name__ == "__main__":
    CustomExperimentGrid(
        # Set the global config for the experiment grid
        base_name=base_experiment_name,
        base_cmd=base_cmd,
        log_dir=log_dir,
        # If the experiment names are too long, you can set the following options to save
        # their log directory using a unique shortened hash names. Otherwise, the log dirs
        # will default to the exp name, including all unique kwargs
        use_hash_in_dirnames=False,
        # Use shorter experiment names (nicknames and kwargs groups are not expanded, e.g. `arch=3b`)
        expand_kwargs_in_name=False,
        expand_groups_in_name=False,
        micromamba_env=micromamba_env,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_cpus_per_gpu=num_cpus_per_gpu,
        # Set the range of the randomly generated seed and port for each experiemnt
        global_seed=42,
        exp_seed_min=100,
        exp_seed_max=200,
        exp_port_min=42000,
        exp_port_max=45000,
        # Set kwargs nicknames
        kwargs_nicknames=kwargs_nicknames,
        # any other kwargs is treated as part of the sweep
        **sweep,  # type: ignore
    ).fire()
