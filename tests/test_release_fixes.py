"""Regression tests for bugs fixed before the public release"""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from launcheon.base.slurm_experiment_grid import SlurmExperimentGrid
from launcheon.base.slurm_utils import get_jobid_from_slurm_output
from launcheon.experiment import is_safe_dirname
from launcheon.parsing_utils import db_parser
from launcheon.utils import std_non_default, strip_url_credentials
from launcheon.yaml_parser import experiment_grid_from_yaml

from .test_experiment_grid import FakeExperimentGrid


# Log directories
def test_is_safe_dirname() -> None:
    assert is_safe_dirname("lr=0.1_bs=2")
    for name in ["", ".", "..", "data=a/b", "ckpt=../../x", "a" * 256]:
        assert not is_safe_dirname(name)


def test_unsafe_names_use_hashed_dirname(tmp_path: Path) -> None:
    grid = FakeExperimentGrid(
        log_dir=str(tmp_path), base_cmd="python main.py", data=["corpus", "corpus/clean", "../x"]
    )
    root = os.path.realpath(tmp_path)
    for exp in grid:
        # every experiment gets its own directory directly under the log root
        assert os.path.dirname(os.path.realpath(exp.log_dir)) == root
    assert not grid[0].use_hashed_dirname
    assert grid[1].use_hashed_dirname and grid[2].use_hashed_dirname
    assert len({exp.log_dir for exp in grid}) == 3


def test_invalid_jobid_is_ignored(tmp_path: Path) -> None:
    grid = FakeExperimentGrid(log_dir=str(tmp_path), base_cmd="python main.py", lr=[0.1])
    exp = grid[0]
    os.makedirs(exp.log_dir)
    for jobid, expected in [("1; rm -rf ~", None), (123, None), ("123_4", "123_4")]:
        with open(exp.info_file, "w") as f:
            json.dump({"jobid": jobid}, f)
        assert exp.jobid == expected


# Filter
@pytest.mark.parametrize(
    "query,expected", [("lr!=0.1", [1, 2]), ("lr<=0.2", [0, 1]), ("lr>=0.2", [1, 2])]
)
def test_filter_multichar_ops(query: str, expected: list[int]) -> None:
    grid = FakeExperimentGrid(base_cmd="python main.py", lr=[0.1, 0.2, 0.3])
    assert sorted(exp.idx for exp in grid.filter(query)) == expected


# Flatten
def test_flatten_keep_idx() -> None:
    grid = FakeExperimentGrid(base_cmd="python main.py", lr=[0.1, 0.2, 0.3])
    assert len(grid.select("0").flatten(keep_idx=True)) == 1

    grid = FakeExperimentGrid(base_cmd="python main.py", lr=list(range(11)))
    flat = grid.flatten(group_by=5, keep_idx=True)
    assert len(flat) == 3
    assert flat[2].command.count("python main.py") == 1


def test_flatten_does_not_double_wrap() -> None:
    grid = FakeExperimentGrid(base_cmd="python main.py", micromamba_env="env", lr=[0.1, 0.2])
    flat = grid.flatten()
    assert grid.get_full_exp_cmd(flat[0]).count("micromamba run") == 1


# Utils
def test_std_non_default_constant() -> None:
    assert std_non_default([0.1, 0.1, 0.1]) == pytest.approx(0.0)


def test_strip_url_credentials() -> None:
    assert strip_url_credentials("https://user:tok@github.com/a/b") == "https://github.com/a/b"
    assert strip_url_credentials("git@github.com:a/b.git") == "git@github.com:a/b.git"


# Metric parsers
def test_db_parser_missing_or_empty(tmp_path: Path) -> None:
    db_file = tmp_path / "results.db"
    assert db_parser(str(db_file), "t", ["acc"]) == {"acc": "N/A"}
    # the missing database is not created
    assert not db_file.exists()

    with sqlite3.connect(db_file) as db:
        db.execute("CREATE TABLE t (acc REAL, step INTEGER)")
    assert db_parser(str(db_file), "t", ["acc"]) == {"acc": "N/A"}

    with sqlite3.connect(db_file) as db:
        db.execute("INSERT INTO t VALUES (0.5, 1), (0.7, 2)")
    assert db_parser(str(db_file), "t", ["acc"]) == {"acc": 0.7}


# Slurm
def test_get_jobid_from_slurm_output() -> None:
    assert get_jobid_from_slurm_output("Submitted batch job 123\n") == 123
    assert get_jobid_from_slurm_output("Submitted batch job 123 on cluster foo\n") == 123
    with pytest.raises(ValueError):
        get_jobid_from_slurm_output("")


def test_multinode_with_micromamba_keeps_args() -> None:
    grid = SlurmExperimentGrid(
        base_cmd="deepspeed main.py",
        micromamba_env="env",
        num_nodes=2,
        dist_launcher="deepspeed",
        lr=[0.1],
        log_dir="/tmp/launcheon_test_multinode",
    )
    _, cmd = grid.get_full_exp_cmd_with_header(grid[0])
    # the training arguments must still be part of the final command
    assert "main.py" in cmd and "--lr" in cmd
    assert cmd.count("micromamba run") == 1


# YAML parser
def _write_yaml(tmp_path: Path, content: str) -> str:
    path = tmp_path / "grid.yml"
    path.write_text(content)
    return str(path)


BASE_YAML = """
cluster:
  type: slurm
  num_gpus: 1
config:
  base_cmd: "python main.py"
  log_dir: "{log_dir}"
"""


def test_yaml_unknown_top_level_key(tmp_path: Path) -> None:
    content = BASE_YAML.format(log_dir=tmp_path) + "sweepp:\n  lr: [0.1]\n"
    with pytest.raises(AssertionError):
        experiment_grid_from_yaml(_write_yaml(tmp_path, content))


def test_yaml_container_mounts_and_parser_format(tmp_path: Path) -> None:
    content = (
        BASE_YAML.format(log_dir=tmp_path).replace(
            "  num_gpus: 1\n", '  num_gpus: 1\n  container_mounts: "/a:/a"\n'
        )
        + '  parser_format: "hydra"\nsweep:\n  lr: [0.1, 0.2]\n'
    )
    grid = experiment_grid_from_yaml(_write_yaml(tmp_path, content))
    assert "countainer" not in grid[0].get_cmd()
    assert "lr=" in grid[0].get_cmd()


def test_yaml_missing_sweep_and_env_var(tmp_path: Path) -> None:
    grid = experiment_grid_from_yaml(_write_yaml(tmp_path, BASE_YAML.format(log_dir=tmp_path)))
    assert len(grid) == 1

    # note: quoted strings are only resolved when preceded by "! "
    content = BASE_YAML.format(log_dir="/tmp/${LAUNCHEON_SURELY_UNSET_VAR}").replace(
        'log_dir: "', 'log_dir: ! "'
    )
    with pytest.raises(ValueError, match="LAUNCHEON_SURELY_UNSET_VAR"):
        experiment_grid_from_yaml(_write_yaml(tmp_path, content))
