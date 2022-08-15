import abc
from enum import Enum
from typing import Callable
import objax


class Reduction(Enum):
    SUM = 0
    MEAN = 1


class LossFunctionCreator(abc.ABC):
    def __init__(self, config) -> None:
        self._config = config["loss"]
        reduction = config["loss"]["reduction"]
        if reduction == "sum":
            self._reduction = Reduction.SUM
        elif reduction == "mean":
            self._reduction = Reduction.MEAN
        else:
            raise ValueError(f"Reduction {reduction} not supported")

    @abc.abstractmethod
    def create_loss_fn(self, model_vars, model) -> Callable:
        pass


class CSELogitsSparse(LossFunctionCreator):
    def create_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            loss = objax.functional.loss.cross_entropy_logits_sparse(logit, label)
            if self._reduction == Reduction.SUM:
                return loss.sum()
            elif self._reduction == Reduction.MEAN:
                return loss.mean()
            else:
                raise RuntimeError("No supported loss reduction")

        return loss_fn
