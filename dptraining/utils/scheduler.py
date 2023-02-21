from abc import ABC, abstractmethod
import numpy as np
from bisect import bisect_right

from dptraining.utils.misc import StateDictObjectMetricTracker


class Schedule(ABC):
    def __init__(self, base_lr: float) -> None:
        self._base_lr: float = base_lr

    @abstractmethod
    def __next__(self) -> float:
        pass

    def update_score(self, score: float) -> None:
        pass

    def __iter__(self):
        return self


class ManualSchedule(Schedule):
    def __init__(self, base_lr: float, lr_list: list[float], epochs: list[int]) -> None:
        super().__init__(base_lr)
        self.lr_list: list[float] = lr_list
        self.epochs: list[int] = epochs
        assert len(self.lr_list) == len(
            self.epochs
        ), "Number of epoch steps and given learning rates must be the same"
        self._steps: int = 0

    def __next__(self) -> float:
        i = bisect_right(self.epochs, self._steps)
        self._steps += 1
        return self.lr_list[i - 1] if i > 0 else self._base_lr


class LinearSchedule(Schedule):
    def __init__(self, base_lr, max_steps: int) -> None:
        super().__init__(base_lr)
        self._max_steps: int = max_steps
        self._steps: int = 0

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


class ReduceOnPlateau(
    Schedule, StateDictObjectMetricTracker
):  # pylint:disable=too-many-instance-attributes
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
        base_lr: float,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = True,
        mode: str = "maximize",
        factor: float = 0.9,
    ):
        Schedule.__init__(
            self,  # pylint:disable=too-many-function-args
            base_lr=base_lr,
        )
        StateDictObjectMetricTracker.__init__(
            self,
            patience,
            min_delta=min_delta,
            cumulative_delta=cumulative_delta,
            mode=mode,
        )
        self.factor = factor

    def update_score(self, score: float) -> None:  # pylint:disable=too-many-branches
        if self.update(score):
            self._adapt_lr()

    def __next__(self) -> float:
        return self._base_lr

    def _adapt_lr(self):
        self._base_lr *= self.factor
        self.counter = 0
