import abc
from typing import Callable, Sequence, Optional, Union
from dptraining.config import Config
from dptraining.config.config import LossReduction
import objax
from objax.typing import JaxArray

from jax import numpy as jnp, vmap, nn


def f_score_from_counts(
    true_positive,
    predictions,
    ground_truth,
    beta,
    nominator_epsilon=0,
    denominator_epsilon=0,
):
    beta_squared = beta * beta
    return ((1 + beta_squared) * true_positive + nominator_epsilon) / (
        beta_squared * predictions + ground_truth + denominator_epsilon
    )


def fscores_from_one_hot(
    gt_one_hot, pr_one_hot, betas=1, nominator_epsilon=0, denominator_epsilon=0
):
    tp_sum = vmap(jnp.sum, -1)(gt_one_hot * pr_one_hot)
    gt_sum = vmap(jnp.sum, -1)(gt_one_hot)
    pr_sum = vmap(jnp.sum, -1)(pr_one_hot)

    if isinstance(betas, Sequence):
        return vmap(f_score_from_counts, (None, None, None, 0, None, None))(
            tp_sum,
            pr_sum,
            gt_sum,
            jnp.array(betas),
            nominator_epsilon,
            denominator_epsilon,
        )
    else:
        return f_score_from_counts(
            tp_sum, pr_sum, gt_sum, betas, nominator_epsilon, denominator_epsilon
        )


def f_scores_from_confidence(
    confidence,
    labels,
    betas=1,
    binary=False,
    nominator_epsilon=0,
    denominator_epsilon=0,
):
    if binary:
        ground_truth = jnp.array(labels, dtype=confidence.dtype)
    else:
        ground_truth = nn.one_hot(
            labels,
            confidence.shape[-1],
            axis=-1,
            dtype=confidence.dtype,
        ).squeeze(-2)
    return fscores_from_one_hot(
        ground_truth, confidence, betas, nominator_epsilon, denominator_epsilon
    )


# loss is batchwise, for samplewise dice loss use jax.vmap + jnp.mean
def f_score_loss(
    labels: JaxArray,
    confidence_or_logits: JaxArray,
    beta: Union[JaxArray, float] = 1,
    class_weights: JaxArray = None,
    binary: bool = False,
    logits: bool = False,
    epsilon: float = 1e-5,
    threshold: Optional[float] = None,
):
    beta = jnp.squeeze(beta)
    assert beta.size == 1
    confidence_or_logits = confidence_or_logits.swapaxes(1, -1)  # turn NCHWZ to NHWZC
    labels = labels.swapaxes(1, -1)

    if logits:
        if binary:
            confidence_value = nn.sigmoid(confidence_or_logits)
        else:
            confidence_value = nn.softmax(confidence_or_logits, axis=-1)  # N,HWZ,C
    else:
        confidence_value = confidence_or_logits

    if binary:
        if class_weights is None or class_weights[0]:
            confidence_value = jnp.concatenate(
                (1 - confidence_value, confidence_value), -1
            )
            if class_weights is not None:
                class_weights = class_weights[1:]
        else:
            class_weights = class_weights[1:]

    if threshold:
        if binary:
            confidence_value = jnp.where(confidence_value > threshold, 1.0, 0.0)
        else:
            confidence_value = jnp.argmax(confidence_value, axis=-1, keepdims=True)

    f_score_values = f_scores_from_confidence(
        confidence=confidence_value,
        labels=labels,
        betas=beta,
        nominator_epsilon=0,
        denominator_epsilon=epsilon,
        binary=False,
    )

    return 1 - jnp.average(f_score_values, -1, class_weights)


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
        self._test_losses = [loss.create_test_loss_fn() for loss in self._test_losses]

        def loss_fn(predicted, correct):
            return sum((l(predicted, correct) for l in self._test_losses))

        return loss_fn


class CrossEntropy(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        if self._config.binary_loss:

            @objax.Function.with_vars(model_vars)
            def loss_fn(inpt, label):
                logit = model(inpt, training=True)
                loss = objax.functional.loss.sigmoid_cross_entropy_logits(
                    logit.squeeze(), label
                )
                match self._config.reduction:
                    case LossReduction.sum:
                        return loss.sum()
                    case LossReduction.mean:
                        return loss.mean()
                    case _ as unsupported:
                        raise RuntimeError(
                            f"Unsupported loss reduction '{unsupported}'"
                        )

            return loss_fn
        else:

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
                        raise RuntimeError(
                            f"Unsupported loss reduction '{unsupported}'"
                        )

            return loss_fn

    def create_test_loss_fn(self):
        if self._config.binary_loss:

            def loss_fn(predicted, correct):
                loss = objax.functional.loss.sigmoid_cross_entropy_logits(
                    predicted.squeeze(), correct
                )
                match self._config.reduction:
                    case LossReduction.sum:
                        return loss.sum()
                    case LossReduction.mean:
                        return loss.mean()
                    case _ as reduction:
                        raise RuntimeError(f"Not supported loss reduction: {reduction}")

            return loss_fn
        else:

            def loss_fn(predicted, correct):
                loss = objax.functional.loss.cross_entropy_logits_sparse(
                    predicted, correct
                )
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

    def create_test_loss_fn(self) -> Callable:
        def fake_loss(*_):
            return 0

        return fake_loss


class DiceLoss(LossFunctionCreator):
    def create_train_loss_fn(self, model_vars, model):
        @objax.Function.with_vars(model_vars)
        def loss_fn(inpt, label):
            logit = model(inpt, training=True)
            loss = f_score_loss(
                confidence_or_logits=logit,
                labels=label,
                logits=True,
                binary=self._config.binary_loss,
            )
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
            loss = f_score_loss(
                confidence_or_logits=predicted,
                labels=correct,
                logits=False,
                binary=self._config.binary_loss,
            )
            match self._config.reduction:
                case LossReduction.sum:
                    return loss.sum()
                case LossReduction.mean:
                    return loss.mean()
                case _ as reduction:
                    raise RuntimeError(f"Not supported loss reduction: {reduction}")

        return loss_fn
