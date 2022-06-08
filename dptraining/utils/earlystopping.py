"""
Adapted ignite early stopping. So far only for increasing scores (i.e. no losses)
"""
from collections import OrderedDict
from typing import cast, Optional

# It would actually be nice to incorporate this pylint warning
# and merge early stopping with reduce on plateau scheduler
# pylint:disable=duplicate-code


class EarlyStopping:
    """EarlyStopping handler can be used to stop the training if
    no improvement after a given number of events.
    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count
            as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase since the last
            `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """

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
        self.best_score = None  # type: Optional[float]

    def __call__(self, score: float) -> bool:

        if self.mode == "maximize":
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
        elif self.mode == "minimize":
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
        return False

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
