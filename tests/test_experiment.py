from typing import Optional

import pytest

from launcheon.experiment import Experiment
from launcheon.parser_format import ArgparseParserFormat


class FakeExperiment(Experiment):
    """An experiment that we can instantiate with placeholder for abstract methods"""

    def cancel(self) -> None:
        pass

    def update_job(self, jobid: Optional[str] = None) -> None:
        pass

    @classmethod
    def __base_log_file__(cls) -> str:
        return "exp.log"


def test_launcheon_secret_key() -> None:
    exp = FakeExperiment(
        idx=0, kwargs={"val1": "gloubiboulga", "val2": "best_{{launcheon.exp.val1}}"}
    )
    assert exp.get_value_from_anyquery("val2") == "best_gloubiboulga"


def test_cyclic_launcheon_secret_key() -> None:
    with pytest.raises(SystemExit):
        FakeExperiment(
            idx=0,
            kwargs={
                "val1": "gloubiboulga_{{launcheon.exp.val2}}",
                "val2": "best_{{launcheon.exp.val1}}",
            },
        )


def test_random_seed_secret_key() -> None:
    exp = FakeExperiment(
        idx=0,
        kwargs={
            "val1": "gloubiboulga_{{launcheon.exp.seed}}",
            "val2": "best_{{launcheon.exp.val1}}",
        },
        random_seed=42,
    )
    assert exp.get_value_from_anyquery("val1") == "gloubiboulga_42"
    assert exp.get_value_from_anyquery("val2") == "best_gloubiboulga_42"


def test_random_port_secret_key() -> None:
    exp = FakeExperiment(
        idx=0,
        kwargs={
            "val1": "gloubiboulga_{{launcheon.exp.port}}",
            "val2": "best_{{launcheon.exp.val1}}",
        },
        random_port=8088,
    )
    assert exp.get_value_from_anyquery("val1") == "gloubiboulga_8088"
    assert exp.get_value_from_anyquery("val2") == "best_gloubiboulga_8088"


def test_name_secret_key() -> None:
    with pytest.raises(SystemExit):
        exp = FakeExperiment(
            idx=0,
            kwargs={
                "val1": "gloubiboulga_{{launcheon.exp.name}}",
                "val2": "best_{{launcheon.exp.val1}}",
            },
        )
    exp = FakeExperiment(
        idx=0,
        exp_name="hoahoa",
        kwargs={
            "val1": "gloubiboulga_{{launcheon.exp.name}}",
        },
    )
    assert exp.get_value_from_anyquery("val1") == "gloubiboulga_hoahoa"
    assert exp.name == "hoahoa"

    exp = FakeExperiment(
        idx=0,
        exp_name="hoahoa_{{launcheon.exp.val2}}",
        kwargs={"val1": "gloubiboulga_{{launcheon.exp.name}}", "val2": "blibli"},
    )
    assert exp.get_value_from_anyquery("val1") == "gloubiboulga_hoahoa_blibli"
    assert exp.name == "hoahoa_blibli"


def test_secret_key_in_command() -> None:
    exp = FakeExperiment(
        idx=0,
        command="python train.py config_{{launcheon.exp.rank}}.json",
        exp_name="train_{{launcheon.exp.rank}}",
        kwargs={
            "rank": "42",
        },
    )
    assert exp.command == "python train.py config_42.json"
    assert exp.name == "train_42"


def test_secret_key_in_repeated_flags() -> None:
    exp = FakeExperiment(
        idx=0,
        kwargs={"val1": ("42", 38, "{{launcheon.exp.val2}}"), "val2": "gloubiboulga"},
    )
    assert exp.get_value_from_anyquery("val1")[0] == "42"  # type: ignore
    assert exp.get_value_from_anyquery("val1")[1] == 38  # type: ignore
    assert exp.get_value_from_anyquery("val1")[2] == "gloubiboulga"  # type: ignore


def test_full_kwargs_dict() -> None:
    exp = FakeExperiment(
        idx=0,
        command="python blou.py --val4 yo_{{launcheon.exp.val2}} --meow felix --meow medor",
        common_args={"val3": "yay"},
        kwargs={"val1": 42, "val2": "gloubiboulga"},
        parser_format=ArgparseParserFormat(),
    )
    d = exp.get_full_kwargs_dict()
    assert "val1" in d and d["val1"] == 42
    assert "val2" in d and d["val2"] == "gloubiboulga"
    assert "val3" in d and d["val3"] == "yay"
    assert "val4" not in d
    assert "meow" not in d

    d = exp.get_full_kwargs_dict(include_base_command=True)
    assert "val1" in d and d["val1"] == 42
    assert "val2" in d and d["val2"] == "gloubiboulga"
    assert "val3" in d and d["val3"] == "yay"
    assert "val4" in d and d["val4"] == "yo_gloubiboulga"
    assert "meow" in d and len(d["meow"]) == 2
    assert d["meow"][0] == "felix"
    assert d["meow"][1] == "medor"


def test_value_from_anywhere() -> None:
    # kwargs takes precedence over attributes
    exp = FakeExperiment(idx=0, kwargs={"random_seed": 42}, random_seed=100)
    assert exp.get_value_from_anyquery("random_seed") == 42
    assert exp.random_seed == 100

    # nicknames also work...
    exp = FakeExperiment(
        idx=0,
        kwargs={"gloubiboulga": 42},
        _kwargs_nicknames={"gloubiboulga": "meow"},
    )
    assert exp.get_value_from_anyquery("meow") == 42
    assert exp.get_value_from_anyquery("gloubiboulga") == 42

    # ...the relation is also bijective...
    exp = FakeExperiment(
        idx=0,
        kwargs={"meow": 42},
        _kwargs_nicknames={"gloubiboulga": "meow"},
    )
    assert exp.get_value_from_anyquery("meow") == 42
    assert exp.get_value_from_anyquery("gloubiboulga") == 42

    # ...but groups still take precedence
    # this is anyway a bit of a weird edge case
    exp = FakeExperiment(
        idx=0,
        kwargs_groups={"gloubiboulga": ("group1", ["meow"])},
        kwargs={"meow": 42},
        _kwargs_nicknames={"gloubiboulga": "meow"},
    )
    assert exp.get_value_from_anyquery("meow") == 42
    assert exp.get_value_from_anyquery("gloubiboulga") == "group1"


def test_filter() -> None:
    exp = FakeExperiment(idx=0, kwargs={"random_seed": 42}, random_seed=100)
    assert exp.filter("random_seed", "=", "42")
    assert exp.filter("random_seed", "!=", "100")

    exp = FakeExperiment(idx=0, kwargs={}, random_seed=100)
    print(exp.get_value_from_anyquery("random_seed"))
    assert exp.filter("random_seed", "!=", "42")
    assert exp.filter("random_seed", "=", "100")
