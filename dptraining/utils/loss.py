import abc
from enum import Enum
from typing import Callable
import objax

from jax import numpy as jnp


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


class CombinedLoss(LossFunctionCreator):
    def __init__(self, config, losses) -> None:
        super().__init__(config)
        self._losses = losses

    def create_loss_fn(self, model_vars, model):
        self._losses = [loss.create_loss_fn(model_vars, model) for loss in self._losses]

        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            return sum((l(inpt, label) for l in self._losses))

        return loss_fn


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


class L2Regularization(LossFunctionCreator):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._regularization = config["hyperparams"]["l2regularization"]

    def create_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(*_):
            return (
                self._regularization
                * 0.5
                * sum(
                    (
                        jnp.sum(jnp.square(x.value))
                        for k, x in model_vars.items()
                        if k.endswith(".w")
                    )
                )
            )

        return loss_fn
