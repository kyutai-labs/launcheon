
# Available commands
List and description of the commands available via the CLI interface of `launcheon`. They can be used as:

#### Python API
```bash
python my_little_launcheon.py [command]
```

#### via YAML configuration file
```bash
launcheon my_little_launcheon_config.yml [command]
```


**Table of contents:**
  * [Submitting on the cluster](#📡-cluster-submission)
  * [Monitoring experiments](#🔬-monitoring)
  * [Chainable commands (selecting, filtering, etc.)](#🔗-selecting)
  * Cluster-specific commands
    * [SLURM](#slurm-commands)

## 📡 Cluster Submission

  * ``submit``. Main submission command. By default, it will only submit experiments which have neither a log directory or log file. It also accepts the following extra options:
    * `--overwrite`: Submits experiments even if the corresponding log dir already exists, and erase said log dir before starting. By default, this will ask for your user input before removing each logdir. If you want to skip this and 
    auto accept, use `--overwrite brute`
    * `--resume`: Submits experiments even if the corresponding log dir already exists, and continue from the same log directory.
    * `--dry_run`: Displays the cluster command to be run without actually submitting it. E.g., in the case of SLURM, it will display the SBATCH script generated, then run `sbatch --test-only` on it
    * Extra SBATCH flags **only available for ``SlurmExperimentGrid``**:
      * `--max_parallel [n; default = None]`:  Submits the experiment grid as a SLURM job array that allows at most `n` jobs running in parallel (the rest will stay pending); 
      * `--nodelist [nodes]`:  A comma-separated string of nodes names. Restrict the hosts the experiments can be submitted to.
      * `--exclude [nodes]`: A comma-separated string of nodes names to exclude (Opposite of ``nodelist``).
      * `--begin [time]`: Submits the job, but will actually only start scheduling it starting from the given time, e.g. `--begin now+8hours`
      * ``--time [time duration]`: Sets a time limit for the job
      * ``--dependency [dependency]``: Sets dependencies constraint for scheduling this job
      * `--keep_slurm_script`: For debugging purposes. This will keep the SLURM submission script used to submit the experiment grid (the default behavior is to clean them up after submission).

  * ``cancel``. For every experiment in the grid, cancel/stops the associated running job, if any.

  * ``remove``. For every experiment in the grid, delete the corresponding logging directory.



## 🔬 Monitoring 

  * ``count``. Prints the number of experiments in the current grid

  * ``count_res``. Prints the total number of resources required to run all experiments in the current grid

  * ``print``. Prints various useful information about the current experiment grid
    * ``print name``: Prints the name of all experiments
    * `print cmd`: Prints the command of all experiments
    * `print slurm_cmd`: (**Only available for ``SlurmExperimentGrid``**). Prints the full SLURM command of all experiments
    * `print log`: Print the content of all experiments' log file
    * `print log [n > 0]`: Print the first `n` lines of all experiments' log file
    * `print log [n < 0]`: Print the last `n` lines of all experiments' log file
    * `print logdir`: Print the logdir (and their content) of all experiments
    * `print status`: Print the status of the associated cluster job for all experiments
    * `print jobid`: Print the cluster job ID curretly associated to each experiment

  * ``monitor [n = -1] --refresh [sleep = 30]``: The live-updated counterpart of `print log`. Displays a table with the first (`n > 0`) or last (`n < 0`) of the experiments' log file, and refreshing every `sleep` seconds.

  * ``tensorboard --port [default = 8897]``. Launches a tensorboard instance on the given port monitoring the log directories of all the experiment in the grid.

  * ``table``. Pretty-prints a table summary of the experiment grid
    * ``--muted``: Removes colors 
    * ``--short``: Do not display the kwargs nested in kwargs groups in the table

#### Example

```bash
python examples/python_api_usage.py filter weight_decay ">" 0 table --short
````

![Example output of the table command](assets/table_example.png)



## 🔗 Selecting
These commands are *chainable*, i.e. they can precede any other command to select a subset of experiments 

  * ``filter``. Filters experiments for which the kwarg `name` is set to `value`. Note that both `name` and `value` can be an alias (as defined in `kwargs_nicknames`) as well as refer to the name of a kwarg group rather than an actual kwarg. The filter supports the following boolean operations:
    * `filter [name] = [value]`
    * `filter [name] != [value]` 
    * `filter [name] < [value]`
    * `filter [name] <= [value]`
    * `filter [name] > [value]`
    * `filter [name] >= [value]` 
    
*Note for disambiguation: The boolean check for the filter will first look through group names, then kwargs. And will first look through true kwarg names, then nicknames.*

  * ``select``. Selects a subset of experiments based their index in the grid or based on the status of the corresponding cluster job.
    * ``select i,j-k:s,..``: Select experiments based on their indices. Each element in the comma-separated list can either be:
      * the index `i` of the experiment to select
      * a range of indices with the format `j-k`, which select all indices from `j` to `k` (*inclusive*)
      * a sub-range of indices with the format `j-k:s`, which select all indices from `j` to `k` (*inclusive*) with a step of `p`
    * ``select [done|running|pending|cancelled|unscheduled|failed]`: Selects all experiments whose corresponding cluster job matches the requested status
    * ``select active``: Selects all experiment which are currently running or have succesfully completed
    * ``select error``: Selects all experiments which have failed or have been cancelled
    * ``select todo``: Selects all experiments which have failed, have been cancelled, or have never been submitted


  ## Cluster-specific

  ### SLURM commands

  These commands are only available **for ``SlurmExperimentGrid``**.

  * ``squeue``. Pretty prints the output of `squeue` for every experiment in the grid.

  * ``noarray`` (*chainable*). By default, the SLURM experiment grid are treated as Job Arrays. The `noarray` command can be added to turn off this behavior.

  * ``reindex``. Modifying a launcheon script might change the index of an experiment, which will mess up with SLURM job array's logging (log files for jobs inside an array contains their index inside the array). To avoid this, `reindex` appropriately renames SLURM log files to match the indexing in the new launcheon script. *Note that `reindex` can only be run if there are no jobs running or pending in this experiment grid, to avoid conflicting with slurm logging*.
