from abc import ABC, abstractmethod
import numpy as np
from collections import OrderedDict
from typing import cast, Optional


class Schedule(ABC):
    def __init__(self, base_lr: float) -> None:
        super().__init__()
        self._base_lr: float = base_lr

    @abstractmethod
    def __next__(self) -> float:
        pass

    def update_score(self, score: float) -> None:
        pass


class LinearSchedule(Schedule):
    def __init__(self, base_lr, max_steps: int) -> None:
        super().__init__(base_lr)
        self._max_steps: int = max_steps
        self._steps: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        if self._steps < self._max_steps:
            new_lr = self.calc_lr(self._base_lr, self._steps, self._max_steps)
            self._steps += 1
            return new_lr
        raise StopIteration

    @staticmethod
    @abstractmethod
    def calc_lr(learning_rate: float, step: int, max_steps: int):
        pass


class CosineSchedule(LinearSchedule):
    @staticmethod
    def calc_lr(learning_rate: float, step: int, max_steps: int):
        return learning_rate * 0.5 * (1 + np.cos(np.pi * (step + 1) / (max_steps + 1)))


class ConstantSchedule(LinearSchedule):
    @staticmethod
    def calc_lr(learning_rate: float, step: int, max_steps: int):
        return learning_rate


class ReduceOnPlateau(Schedule):  # pylint:disable=too-many-instance-attributes
    """
    Similar to early stopping

    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase since the
        last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        lr: float,
        patience: int,
        factor: float = 0.9,
        min_delta: float = 0.0,
        cumulative_delta: bool = True,
        mode: str = "maximize",
    ):
        super().__init__(lr)
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if mode not in ["minimize", "maximize"]:
            raise ValueError("Argument mode must be either minimize or maximize")

        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None  # type: Optional[float]

    def update_score(self, score: float) -> None:  # pylint:disable=too-many-branches
        if self.mode == "maximize":
            if self.best_score is None:
                self.best_score = score
            elif score <= self.best_score + self.min_delta:
                if not self.cumulative_delta and score > self.best_score:
                    self.best_score = score
                self.counter += 1
                if self.counter >= self.patience:
                    self._adapt_lr()
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == "minimize":
            if self.best_score is None:
                self.best_score = score
            elif score >= self.best_score - self.min_delta:
                if not self.cumulative_delta and score < self.best_score:
                    self.best_score = score
                self.counter += 1
                if self.counter >= self.patience:
                    self._adapt_lr()
            else:
                self.best_score = score
                self.counter = 0
        else:
            raise RuntimeError("This is not supported and should not happen :/")

    def __next__(self) -> float:
        return self._base_lr

    def __iter__(self):
        return self

    def _adapt_lr(self):
        self._base_lr *= self.factor
        self.counter = 0

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
