# 🐍 Launcheon - Python API

To use `launcheon` via the Python API, we need to create a python script `my_little_launcheon.py` with the following structure:

```python
from launcheon import SlurmExperimentGrid

if __name__ == "__main__":
    SlurmExperimentGrid(...).fire()
```

We can then use the methods of the `ExperimentGrid` as commands to the CLI using `python my_little_launcheon.py [command]`. See [this page](doc_cli.md) for a list and description of available commands.


**Table of Contents:**
  * [Defining the experiment sweep](#defining-the-experiment-sweep)
  * [kwargs nicknames](#kwargs-nicknames)
  * [``launcheon`` secret keys](#secret-keys)
  * [Initializing the experiment grid](#initializing-the-experiment-grid)
  * [Adding custom commands](#adding-custom-methods--commands)

## Defining the experiment sweep
A **sweep dictionary** defines the set of experiments contained in the experiment grid. The sweep dictionary maps a string to an entry which can be of either three types:

  * **flag_name** $\mapsto$ a **list**: A simple sweep over several values for `flag_name`.
  * **flag_name** $\mapsto$ a **dictionary** (aka, a *kwargs_group*): A nested sweep that can be used to avoid certain combinations of hyperparameters, resulting in a restricted grid search. A typical example usage is for tying model size and batch size, e.g.:
    ```python
    "arch": {
        "3b": {"ckpt_dir": "3b", "bs": 1024, "width": 2048, "num_layers": 12},
        "7b": {"ckpt_dir": "7b", "bs": 256, "width": 4096, "num_layers": 24},
    },
    ```
    Note that experiment naming will not expand the kwargs groups to keep concise names: for isntance, in that case, the corresponding experiments' name will include `_arch=2b` instead of e.g. `_ckpt_dir=3b_bs=1024_num_layers=12`.
  * **flag_name** $\mapsto$ a **tuple**: A fixed value for parameter which accepts a sequence of values (e.g. `append` action in `argparse`, or `multiple=True` in `click`) `flag_name`.
  * **flag_name** $\mapsto$ **anything else**: `flag_name` is a fixed parameter (same value across all experiments). It also won't be used when generating the unique name of the experiment.


## kwargs nicknames
Some flag names or value might be very long which can be cumbersome. To alleviate this, we can also define a dictionary **kwargs_nicknames** which maps *a nickname* for a kwarg name **or** value to the true name of this kwarg name/value, which will only be expanded when generating the experiment's command. 

#### Example
In the following examples, both experiment grids are equivalent (they expand to the same experiment commands), but experiments in `grid2` will have significantly shorter names.

```python
grid1 = SlurmExperimentGrid(sweep={"checkpoint_directory": ["/very/long/path/1", "/very/long/path/2"]})


grid2 = SlurmExperimentGrid(
        sweep={"ckpt": ["dir1", "dir2"]},
        kwargs_nicknames={
            "ckpt": "checkpoint_directory",
            "dir1": "/very/long/path/1",
            "dir2": "/very/long/path/2"
            }
        )
```

## Secret keys
Sometimes, you may want to use a value (e.g. the experiment name, or log directory). To enable this, you can refer 
to special "secret keys" in the sweep values of your experiment. There are currently four available secret keys:
  * `"{{launcheon.exp.name}}"` will resolve to the automatically generated unique experiment name
  * `"{{launcheon.exp.log_dir}}"`will resolve to the automatically generated unique experiment log directory (which is a concatenation of the base log directory and the experiemnt name)
  * `"{{launcheon.exp.seed}}"` will resolve to a unique random seed generated for this experiment. This can be further customized with arguments `exp_seed_min` and `exp_seed_max` when instantiating the experiment grid.
  * `"{{launcheon.exp.port}}"` will resolve to a unique random port generated for this experiment. This can be further customized with arguments `exp_port_min` and `exp_port_max` when instantiating the experiment grid.


**Note:** Currently, secret keys are only supported as values inside the sweep, not in `kwargs_nicknames` nor in the base command.

## Initializing the experiment grid

Once the core sweep is defined, we can finish initializing the experiment grid as follow:

```python
SlurmExperimentGrid(
        # Base name prefix that will be prepended to every experiment's name
        base_name="experiment_name_prefix",
        # Base command that will be prepended to every experiment's kwargs
        base_cmd="python my_script.py",
        # Base log directory
        log_dir=f"/home/{os.environ.get('USER')}/experiments",
        # Resources to request for each individual experiment
        num_gpus=4,
        num_cpus_per_gpu=8,
        num_nodes=1,
        # If given, all generated commands will be wrapped around micromaba run -n [micromamba_env]
        micromamba_env=None,
        # How kwargs are formatted for the command line: "argparse" (default) or "hydra",
        # or a custom `launcheon.parser_format.ParserFormat`
        parser_format="argparse",
        # The experiment grid assigns a random seed and port to each created experiment. These can later
        # be accessed using launcheon secret keys. In addition, we can futher customize the sampling 
        # process for the port/seed using the following args:
        global_seed=42,
        exp_seed_min=100,
        exp_seed_max=200,
        exp_port_min=42000,
        exp_port_max=45000,
        # Nicknames for keyword arguments names and values
        kwargs_nicknames=kwargs_nicknames,
        # FOR SLURM ExperimentGrids only - Extra args to refine the slurm environment
        partition="my_partition",
        work_dir=f"/home/{os.environ.get('USER')}",
        container_image=None,
        # Any extra arguments will be treated as parts of the sweep to build the experiments
        **sweep,
)
```



## Adding custom methods / commands

Since we are using `fire`, any method added to the `ExperimentGrid` class can automatically be used as a command from the CLI. This can be useful to add plotting code snippets for specific experiments.

For instane, the following code snippet will add a new command to the experiment grid which can be in turn called via CLI using `python my_little_launcheon.py print_table [show_kwarg]` 

```python
from launcheon import SlurmExperimentGrid
from rich.console import Console
from rich.table import Table

class CustomExperimentGrid(SlurmExperimentGrid):

    def print_table(self, show_kwarg="lr"):
        table = Table(title=self.base_name)
        table.add_column("Exp ID")
        table.add_column(show_kwarg)
        for exp in self:
            table.add_row(
                str(exp.idx),
                str(exp.get_kwarg_or_group_value(show_kwarg)),
            )
        Console().print(table)


if __name__ == "__main__":
    CustomExperimentGrid(...).fire()
```