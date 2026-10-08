# 🐑 Launcheon - YAML API

The main benefit of the [Python API](doc_python.md) is to **(i)** share partial configurations across different experiment scripts by loading their respective modules and **(ii)** customize the experiment grid to augment it with any methods (e.g., plotting or pretty-printing utils). 

Nevertheless, this API can also be cumbersome and more prone to errors when defining the sweep as a dictionary. As an alternative, we can also configure an experiment grid using a simpler and more structured YAML API. 

We can then use all the methods of the `ExperimentGrid` as commands to the CLI using `launcheon my_little_launcheon_config.yml [command]`. See [this page](doc_cli.md) for a full list and description of available commands.

Below is a description of all configuration options for a `launcheon` configuration YAML file:


```yaml
# Cluster type and per-experiment resources
cluster:
  # Cluster type (only SLURM is currently supported)
  type: slurm
  # Per-experiment resources (number of nodes, number of GPU per nodes and number of CPU per GPU)
  # the total number of gpus will be num_gpus * num_nodes
  # the total number of cpus will be num_cpus_per_gpu * num_gpus * num_nodes
  num_nodes: 2
  num_gpus: 2
  num_cpus_per_gpu: 8
  # (Optional) SLURM only options - e.g. partition and container
  partition: "my_partition"
  work_dir: ! "/home/${USER}/src"
  container_image: "/path/to/my_container_image.sqsh"

# Global settings for the experiment grid
config:
  # Base name prefix
  base_name: "my_little_experiment"
  # Main command (shared across all experiments)
  base_cmd: "python my_little_script.py"
  # Log directory for slurm outputs
  # Note the yaml parser can expand environment variables such as ${USER} in any of the values
  # found in the file. However, for quoted strings, the resolving is only applied when 
  # the string is preceded by "! "
  # see https://github.com/yaml/pyyaml/issues/457#issuecomment-1030658939
  log_dir: ! "/home/${USER}/launcheon_test/"
  # Use shorter experiment names (nicknames and kwargs groups are not expanded, e.g. `arch=3b`)
  expand_kwargs_in_name: false
  expand_groups_in_name: false
  # (Optional) If given, `base_cmd` will be wrapped with a call
  # to micromamba run for this environment
  micromamba_env: "snake_env"
  # (Optional) How kwargs are formatted on the command line: "argparse" (default,
  # e.g. `--lr "0.1"`) or "hydra" (e.g. `lr="0.1"`)
  parser_format: "argparse"
  # A unique random port number will be generated for each experiment and can be accessed in the sweep
  # with the secret key "{{launcheon.exp.port}}"; in addition, we can customize the range of the port
  exp_port_min: 42000
  exp_port_max: 45000
  # In addition, the global seed controls this sampling process (defaults to 42)
  global_seed: 42

# Optional: Kwargs nicknames
# Maps a nickname to a kwarg name or value
# The nicknames can be used in the sweep field below and will be properly resolved
# when the command is generated for each experiment
kwargs_nicknames:
  bs: "batch_size"
  7b: "/very/long/path/to/7b/model"
  3b: "/very/long/path/to/3b/model"
  aug: "data_augmentation"

# Sweep
sweep:
    ## simple grid search sweep;
    # just a list of value to test for each 
    lr: [1e-3, 1e-5, 1e-7]
    weight_decay: [0.1, 0.0]
    # singleton arguments
    # won't be used in experiment naming
    dataset: "imagenet"
    # however, if given as a list, it will be used in experiment naming even
    # though all experiments have the same value
    aug: ["randaugment"]
    # Advamced feature: Launcheon secret keys
    # A (unique random seed  and random port will be generated 
    seed: ["{{launcheon.exp.seed}}"]
    ssh_port: "{{launcheon.exp.port}}"
    main_logs_dir: "{{launcheon.exp.log_dir}}"   # launcheon.exp.log_dir will itself expand to /config.log_dir/exp.name
    tb_dir: "/my_other_path/tb_logs/{{launcheon.exp.name}}"
    # Advanced feature: kwargs groups
    # this allows to introduce constraints between certain hyperparameters
    # the experiment naming will use the format "arch=3b" but the actual kwargs in the nested dict
    # will be the ones used when generating the command
    arch:
      3b: 
        ckpt_dir: "3b"
        bs: 1024
        width: 2048
        num_layers: 12
      7b:
        ckpt_dir: "7b"
        bs: 256
        width: 4096
        num_layers: 24
```