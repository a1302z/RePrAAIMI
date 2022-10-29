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
    def create_train_loss_fn(self, model_vars, model) -> Callable:
        pass

    @abc.abstractmethod
    def create_test_loss_fn(self) -> Callable:
        pass


class CombinedLoss(LossFunctionCreator):
    def __init__(self, config, losses) -> None:
        super().__init__(config)
        self._train_losses = losses
        self._test_losses = losses

    def create_train_loss_fn(self, model_vars, model):
        self._train_losses = [
            loss.create_train_loss_fn(model_vars, model) for loss in self._train_losses
        ]

        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            return sum((l(inpt, label) for l in self._train_losses))

        return loss_fn

    def create_test_loss_fn(self) -> Callable:
        self._test_losses = [loss.create_test_loss_fn() for loss in self._train_losses]

        def loss_fn(predicted, correct):
            return sum((l(predicted, correct) for l in self._test_losses))


class CSELogitsSparse(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
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

    def create_test_loss_fn(self):
        def loss_fn(predicted, correct):
            loss = objax.functional.loss.cross_entropy_logits_sparse(predicted, correct)
            if self._reduction == Reduction.SUM:
                return loss.sum()
            elif self._reduction == Reduction.MEAN:
                return loss.mean()
            else:
                raise RuntimeError("No supported loss reduction")

        return loss_fn


class L1Loss(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            loss = objax.functional.loss.mean_absolute_error(logit, label)
            if self._reduction == Reduction.SUM:
                return loss.sum()
            elif self._reduction == Reduction.MEAN:
                return loss.mean()
            else:
                raise RuntimeError("No supported loss reduction")

        return loss_fn

    def create_test_loss_fn(self):
        def loss_fn(predicted, correct):
            loss = objax.functional.loss.mean_absolute_error(predicted, correct)
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

    def create_train_loss_fn(self, model_vars, model):
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

    def create_test_loss_fn(self, model_vars, model) -> Callable:
        return 0
