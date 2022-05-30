import abc
from typing import Callable
import objax


class LossFunctionCreator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def create_loss_fn(model_vars, model) -> Callable:
        pass


class CSELogitsSparse(LossFunctionCreator):
    @staticmethod
    def create_loss_fn(model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            return objax.functional.loss.cross_entropy_logits_sparse(
                logit, label
            ).mean()

        return loss_fn
