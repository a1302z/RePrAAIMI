from abc import ABC, abstractmethod
import numpy as np


class Schedule(ABC):
    def __init__(self, base_lr: float) -> None:
        super().__init__()
        self._base_lr: float = base_lr

    @abstractmethod
    def __next__(self) -> float:
        pass


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


## nice option would be an extension for an adaptive schedule
