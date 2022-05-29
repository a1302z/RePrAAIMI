from typing import Callable
import objax
from abc import ABC, abstractstaticmethod


class LossFunctionCreator(ABC):
    @abstractstaticmethod
    def create_loss_fn(model_vars, model) -> Callable:
        pass


class CSELogitsSparse(LossFunctionCreator):
    def create_loss_fn(model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(x, label):
            logit = model(x, training=True)
            return objax.functional.loss.cross_entropy_logits_sparse(
                logit, label
            ).mean()

        return loss_fn
