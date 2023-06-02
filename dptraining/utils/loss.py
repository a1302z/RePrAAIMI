import abc
from typing import Callable
from dptraining.config import Config
from dptraining.config.config import LossReduction
import objax

from jax import numpy as jnp


class LossFunctionCreator(abc.ABC):
    def __init__(self, config: Config) -> None:
        self._config = config.loss

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

        return loss_fn


class CSELogitsSparse(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            loss = objax.functional.loss.cross_entropy_logits_sparse(logit, label)
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as unsupported:
                    raise RuntimeError(f"Unsupported loss reduction '{unsupported}'")

        return loss_fn

    def create_test_loss_fn(self):
        def loss_fn(predicted, correct):
            loss = objax.functional.loss.cross_entropy_logits_sparse(predicted, correct)
            return loss.mean()
        return loss_fn

class BCSELogitsSparse(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True).squeeze()
            label = label.squeeze()
            assert logit.shape == label.shape , f"Found mismatching shapes for logits ({logit.shape}) and labels ({label.shape})"
            loss = objax.functional.loss.sigmoid_cross_entropy_logits(logit, label)
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as reduction:
                    raise RuntimeError(f"Not supported loss reduction: {reduction}")

        return loss_fn

    def create_test_loss_fn(self):
        def loss_fn(predicted, correct):
            predicted = predicted.squeeze()
            assert predicted.shape == correct.shape , f"Found mismatching shapes for logits ({predicted.shape}) and labels ({correct.shape})"
            loss = objax.functional.loss.sigmoid_cross_entropy_logits(predicted, correct)
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as reduction:
                    raise RuntimeError(f"Not supported loss reduction: {reduction}")

        return loss_fn

class L1Loss(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            loss = objax.functional.loss.mean_absolute_error(logit, label)
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as reduction:
                    raise RuntimeError(f"Not supported loss reduction: {reduction}")

        return loss_fn

    def create_test_loss_fn(self):
        def loss_fn(predicted, correct):
            loss = objax.functional.loss.mean_absolute_error(predicted, correct)
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as reduction:
                    raise RuntimeError(f"Not supported loss reduction: {reduction}")

        return loss_fn


class L2Regularization(LossFunctionCreator):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._regularization = config.hyperparams.l2regularization

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

    def create_test_loss_fn(self, _, __) -> Callable:
        return 0
