"""A simple experiment grid sweep showcasing several features such as:

  * `kwargs_nicknames`: This allows to use "nicknames" for both flags names
    and their values rather than their full version. This can be useful to
    replace very long flags (e.g. path to a checkpoint) with a shorter
    alias in experiments' names

  * `secret keys`: Sometimes we may want to refer to the value of a parameter
    in the current experiment while avoiding code duplication: For instance, like
    shown here, we may want to define a flag for tensorboard logs to point to a
    directory under the main output log directory. Launcheon secret keys typically
    have the format "{{launcheon.exp.XXXX}}"

  * `hashed_names`: An experiment is uniquely characterized by its name: i.e., the
    base experiment name following by each kwargs flag/value (omitting, the singleton
    kwargs shared by experiments). however, this may lead to very long experiment (and
    log directory) names. To use a simpler (but harder to interpret) hashed name for
    experiments, you can set `use_hash_in_dirnames` to True when creating the grid.

  * `custom methods`: When using the Python API you can subclass `ExperimentGrid` to
    define custom methods you may want to run on the whole grid (e.g., for visualization)


Some additional advanced features are also introduced in the following example
"02_advanced_grid"
"""

import os

from launcheon import SlurmExperimentGrid

# Base name prefix for every experiment in the grid
base_experiment_name = "my_little_experiment"

# Root directory under which each experiment is saved as
# a dictionary
log_dir = f"/home/{os.environ.get('USER')}/launcheon_test"

# Environment and resources options *per experiment*
# Here, each experiment in the grid uses 2 GPUs on a single node,
# and a total of 8 * 2 = 6 CPUs
num_gpus = 2
num_nodes = 1
num_cpus_per_gpu = 8

# Base command to run, shared by all experiments
base_cmd = "python my_little_script.py"
micromamba_env = "my_little_env"

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
    # Singleton Arguments are given directly as a single value
    # they won't be used in experiment naming so be careful when you change them
    dataset="imagenet",
    clip_grad=True,
    # however, if given as a list of size, it will be used in experiment
    # naming even though all experiments have the same value
    aug=["randaugment"],
    # Advanced feature: Launcheon secret keys
    # These will be resolved on-the-fly at experiment's creation
    # Because it is given as a singleton list, seed will be integrated in the experiment's name
    # in contrast, port, tb_dir and output dir will not be added to the name
    out_dir="{{launcheon.exp.log_dir}}",
    tb_dir="{{launcheon.exp.log_dir}}/tb_dir",
    # Advanced feature: kwargs groups
    # the experiment naming will use the simple format "arch=3b", but the actual kwargs
    # used in the generated command are the ones defined in the nested dictionary
    arch={
        "3b": {"ckpt_dir": "3b", "bs": 1024, "width": 2048, "num_layers": 12},
        "7b": {"ckpt_dir": "7b", "bs": 256, "width": 4096, "num_layers": 24},
    },
)


class CustomExperimentGrid(SlurmExperimentGrid):
    """Example of adding a custom method to the experiment grid. Here we
    inherit from SlurmExperimentGrid as we want to submit experiments
    on a Slurm cluster

    You can use your new command as:

    ```bash
    python examples/01_simple_grid/python_api_usage.py print_tbdir
    ```
    """

    def print_tbdir(self) -> None:
        for exp in self:
            exp.print_as_header()
            print(exp.get_value_from_anyquery("tb_dir"))


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
        # Set the environment
        micromamba_env=micromamba_env,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        num_cpus_per_gpu=num_cpus_per_gpu,
        # Slurm-specific: By default, for Slurm cluster, we treat every
        # experiment grid as Job arrays, however this can be turned
        # off with the following default
        # Set kwargs nicknames
        kwargs_nicknames=kwargs_nicknames,
        # any other kwargs is treated as part of the sweep
        **sweep,  # type: ignore
    ).fire()
