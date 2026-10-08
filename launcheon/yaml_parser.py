"""Parser for the launcheon's YAML API: Parses a yaml configuration file
into a valid experiemnt grid"""

import os
import re
from typing import Any

import fire
import yaml

from launcheon.base.slurm_experiment_grid import SlurmExperimentGrid
from launcheon.experiment_grid import ExperimentGrid
from launcheon.global_variables import (
    ALLOWED_KEYS_IN_CLUSTER_GLOBAL_CONFIG,
    ALLOWED_KEYS_IN_GLOBAL_CONFIG,
)


def experiment_grid_from_yaml(yaml_file: str) -> ExperimentGrid:
    """Convertx the given yaml file into a valid experiment grid

    :param yaml_file: Path to a launcheon config file in YAML format

    :return: The experiment grid capturing the experiment described in the launcheon file
    """

    # Resolve any variable with format ${XXX} using environment variables
    # Note: we use a dedicated loader to avoid modifying yaml's global SafeLoader
    class EnvVarLoader(yaml.SafeLoader):  # pylint: disable=too-many-ancestors
        """SafeLoader resolving environment variables"""

    env_var_matcher = re.compile(r".*\$\{([^}^{]+)\}.*")

    def path_constructor(_: Any, node: Any) -> Any:
        value = node.value
        s, offset = "", 0
        for match in re.finditer(r"\$\{([^}^{]+)\}", value):
            i, j = match.span()
            var_name = match.group(1)
            var = os.environ.get(var_name)
            if var is None:
                raise ValueError(
                    f"Environment variable `{var_name}` used in {yaml_file} is not set"
                )
            s += value[offset:i] + var
            offset = j
        s += value[offset : len(value)]
        return s

    EnvVarLoader.add_implicit_resolver("!envvar", env_var_matcher, None)
    EnvVarLoader.add_constructor("!envvar", path_constructor)

    # Load yaml file
    with open(yaml_file, "r") as stream:
        d = yaml.load(stream, Loader=EnvVarLoader)  # noqa: S506 (SafeLoader subclass)

    # Check keys
    assert isinstance(d, dict), f"Invalid launcheon YAML file {yaml_file}"
    assert "cluster" in d, "Missing cluster config for the experiment grid"
    assert "type" in d["cluster"], "Missing cluster type for the experiment grid"
    assert "config" in d, "Missing global config for the experiment grid"
    # empty sections are parsed as None
    for key in ["config", "sweep"]:
        if d.get(key) is None:
            d[key] = {}
    left_over = set(list(d.keys())).difference({"config", "sweep", "kwargs_nicknames", "cluster"})
    if len(left_over) > 0:
        raise AssertionError("Found unexpected key(s) in launcheon YAML script:", left_over)

    # get cluster type and config
    cluster_config_type = d["cluster"]["type"]

    if cluster_config_type == "slurm":
        grid_constructor = SlurmExperimentGrid
    else:
        raise NotImplementedError(
            f"Unknown cluster type in global config: {cluster_config_type}."
            f"Allowed options are: 'slurm'"
        )

    cluster_config_options = {**d["cluster"]}
    del cluster_config_options["type"]
    if len(cluster_config_options):
        left_over = set(cluster_config_options.keys()).difference(
            ALLOWED_KEYS_IN_CLUSTER_GLOBAL_CONFIG["all"]
            + ALLOWED_KEYS_IN_CLUSTER_GLOBAL_CONFIG[cluster_config_type]
        )
        if len(left_over) > 0:
            raise AssertionError(
                "Found unexpected key(s) in launcheon YAML script in `cluster`:",
                left_over,
            )

    # Check keys in global configuration
    protected_keys = (
        ALLOWED_KEYS_IN_GLOBAL_CONFIG + ALLOWED_KEYS_IN_CLUSTER_GLOBAL_CONFIG[cluster_config_type]
    )
    if len(d["config"]):
        left_over = set(list(d["config"].keys())).difference(protected_keys)
        if len(left_over) > 0:
            raise AssertionError(
                "Found unexpected key(s) in launcheon YAML script in `config`:",
                left_over,
            )

    # check that none of the sweep keys are protected keys
    if len(d["sweep"]):
        intersect = set(d["sweep"].keys()).intersection(protected_keys)
        if len(intersect) > 0:
            raise AssertionError(
                "Error in launcheon YAML file. "
                f"`sweep` contains keys which can only appear in the `config` field: {intersect}"
            )

    # build the experiment grid
    return grid_constructor(
        **d["config"],
        kwargs_nicknames=d.get("kwargs_nicknames", None),
        **cluster_config_options,
        **d["sweep"],
    )


def yaml_parser_entry_point() -> None:
    """Entry point for setup.py's launcheon command"""
    fire.Fire(experiment_grid_from_yaml)
