

# Simple Experiment Grid Example

A simple example of using `launcheon` via the Python API is given in `examples/01_simple_grid/python_api_usage.py`. The script can then be used directly from the command line, inherting all methods of `ExperimentGrid` as subcommands.

Alternatively, the experiment grid can also be configured via a YAML file. The counterpart YAML example is located in `examples/01_simple_grid/yaml_api_usage.yml` and the corresponding experiment grid CLI can be started with the `launcheon` command.

Below you can find example of useful commands for both APIs. A list of all available commands is given [here](../../docs/doc_cli.md).




|     | Python API | YAML API |
| --- | ---------- | -------- |
| Pros | Can add custom methods + can re-use/share configs across experiment grids |  Simple and structured. Easier to share  |
| Cons | More prone to syntax errors (dictionary formatting) | More rigid format |
| Example | [`examples/01_simple_grid/python_api_usage.py`](python_api_usage.py) | [`examples/01_simple_grid/yaml_api_usage.yml`](yaml_api_usage.yml) |
| Command | `python python_api_usage.py ....` | `launcheon yaml_api_usage.yml ...`|


For instance, to print the names of all experiments in the sweep:
```bash
    python examples/01_simple_grid/python_api_usage.py print name
    # or
    launcheon examples/01_simple_grid/yaml_api_usage.yml print name
```

To print the command corresponding to the 2nd and 7th experiment of the sweep
```bash
    python examples/01_simple_grid/python_api_usage.py select 1,6 print cmd
    # or
    launcheon examples/01_simple_grid/yaml_api_usage.yml select 1,6 print cmd
```

To print the status of all experiments based on a certain flag name/value:
```bash
    python examples/01_simple_grid/python_api_usage.py filter weight_decay ">" 0 print status
    # or 
    launcheon examples/01_simple_grid/yaml_api_usage.yml filter weight_decay ">" 0 print status
```

To submit all experiments as a SLURM job array with at most 4 experiments running in parallel (and 
overwriting the log directory in case one of these experiments has already been run)
```bash
    python examples/01_simple_grid/python_api_usage.py submit --max_parallel 4 --overwrite
    # or 
    launcheon examples/01_simple_grid/yaml_api_usage.yml submit --max_parallel 4 --overwrite
```

To print a live monitor of the last 5 lines of the logfile of all experiments which have a job currently running
```bash
    python examples/01_simple_grid/python_api_usage.py select running monitor -5
    # or 
    launcheon examples/01_simple_grid/yaml_api_usage.yml select running monitor -5
```

To print the log directory of a specific experiment (here, 4th experiment in the grid) and its contents
```bash
    python examples/01_simple_grid/python_api_usage.py select 3 print logdir
    # or 
    launcheon examples/01_simple_grid/yaml_api_usage.yml select 3 print logdir
```

To launch a Tensorboard instance on all the experiemnts' log directories
```bash
    python examples/01_simple_grid/python_api_usage.py tensorboard --port 8899
    # or 
    launcheon examples/01_simple_grid/yaml_api_usage.yml tensorboard --port 8899
```