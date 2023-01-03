from abc import abstractmethod
from collections import OrderedDict
from typing import cast

from objax import VarCollection
from numpy import prod


class StateDictObject:
    @abstractmethod
    def state_dict(self) -> "OrderedDict[str, float]":
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: "OrderedDict[str, float]") -> None:
        pass


class StateDictObjectMetricTracker(StateDictObject):
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = True,
        mode: str = "maximize",
    ):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if mode not in ["minimize", "maximize"]:
            raise ValueError("Argument mode must be either minimize or maximize")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def state_dict(self) -> "OrderedDict[str, float]":
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return OrderedDict(
            [
                ("counter", self.counter),
                ("best_score", cast(float, self.best_score)),
                ("mode", self.mode),
            ]
        )

    def load_state_dict(self, state_dict: "OrderedDict[str, float]") -> None:
        """Method replace internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.mode = state_dict["mode"]

    def update(self, score: float) -> bool:

        match self.mode:
            case "maximize":
                if self.best_score is None:
                    self.best_score = score
                elif score <= self.best_score + self.min_delta:
                    if not self.cumulative_delta and score > self.best_score:
                        self.best_score = score
                    self.counter += 1
                    if self.counter >= self.patience:
                        return True
                else:
                    self.best_score = score
                    self.counter = 0
            case "minimize":
                if self.best_score is None:
                    self.best_score = score
                elif score >= self.best_score - self.min_delta:
                    if not self.cumulative_delta and score < self.best_score:
                        self.best_score = score
                    self.counter += 1
                    if self.counter >= self.patience:
                        return True
                else:
                    self.best_score = score
                    self.counter = 0
            case _:
                raise RuntimeError("This is not supported and should not happen :/")


def get_num_params(model_vars: VarCollection) -> int:
    return sum([prod(v.shape) for v in model_vars.values()])
