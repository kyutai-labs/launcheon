from typing import Any, Optional

import pytest

from launcheon.experiment import Experiment
from launcheon.experiment_grid import ExperimentGrid


class FakeExperiment(Experiment):
    """An experiment that we can instantiate with placeholder for abstract methods"""

    def cancel(self) -> None:
        pass

    def update_job(self, jobid: Optional[str] = None) -> None:
        pass

    @classmethod
    def __base_log_file__(cls) -> str:
        """[Need override] Log file for this experiment, relative to the base log directory"""
        return "exp.log"


class FakeExperimentGrid(ExperimentGrid):
    """An experiment grid that we can instantiate with placeholder for abstract methods"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            _experiment_constructor=FakeExperiment,
            **kwargs,  # type: ignore
        )

    def __submit_exps__(self, *args: Any, **kwargs: Any) -> Optional[int]:
        del args, kwargs
        return 0


def test_duplicate() -> None:
    # this is fine (override)
    FakeExperimentGrid(
        val1=["gloubiboulga"], group={"g1": {"val1": "blip"}, "g2": {"val1": "bloup"}}
    )
    # this throws an error
    with pytest.raises(SystemExit):
        FakeExperimentGrid(
            val1=["gloubiboulga", "bad"],
            group={"g1": {"val1": "blip"}, "g2": {"val1": "bloup"}},
        )

    # this is fine (override)
    FakeExperimentGrid(
        kwargs_nicknames={"meow": "cat"}, cat=["gloubiboulga"], meow=["blip", "bloup"]
    )

    # this throws an error
    with pytest.raises(SystemExit):
        FakeExperimentGrid(
            kwargs_nicknames={"meow": "cat"},
            cat=["gloubiboulga", "bad"],
            meow=["blip", "bloup"],
        )

    # this is kinda weird and playing with fire, but it works
    FakeExperimentGrid(
        val1=["gloubiboulga", "medium"],
        group={"val1": {"meow": "blip"}, "val2": {"meow": "bloup"}},
    )

    # this works but will print a warning about val1 being found in the command
    # and being overriden by kwargs
    FakeExperimentGrid(base_cmd="python train.py --val1 gloubiboulgi", val1=["gloubiboulga"])


def test_illegal_nicknames() -> None:
    with pytest.raises(SystemExit):
        FakeExperimentGrid(
            val1=["gloubiboulga"],
            kwargs_nicknames={"val1": "bloublou", "bloublou": "val2"},
        )
    # this is fine
    FakeExperimentGrid(
        val1=["gloubiboulga"],
        kwargs_nicknames={"val1": "bloublou"},
    )
    # this is not
    with pytest.raises(SystemExit):
        FakeExperimentGrid(
            val1=["gloubiboulga"],
            kwargs_nicknames={"val1": "bloublou", "val2": "bloublou"},
        )


def test_illegal_nested_groups() -> None:
    # groups in groups values are not allowed
    with pytest.raises(SystemExit):
        FakeExperimentGrid(
            val1=["gloubiboulga", "medium"],
            group={"val1": {"meow": {"blip": "blouii"}}, "val2": {"meow": "bloup"}},
        )

    # but tuples are ok
    d = FakeExperimentGrid(
        val1=["gloubiboulga", "medium"],
        group={"val1": {"meow": ("blip", "blouii")}, "val2": {"meow": "bloup"}},
    )
    assert len(d) == 4

    # lists are now ok too !
    d = FakeExperimentGrid(
        val1=["gloubiboulga", "medium"],
        group={"val1": {"meow": ["blip", "blouii"]}, "val2": {"meow": "bloup"}},
    )
    assert len(d) == 6
