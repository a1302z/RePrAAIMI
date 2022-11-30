from __future__ import division
from __future__ import unicode_literals

import numpy as np
from jax import numpy as jnp, lax

from typing import Dict
from objax import TrainVar, Module, StateVar, ModuleList, TrainRef

# import torch


# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
# Objax version based on:
# https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py

zero_element_vector = jnp.array(0, np.uint32)


class ExponentialMovingAverage(Module):
    """
    Maintains (exponential) moving average of a set of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
            Note that EMA is computed on *all* provided parameters,
            regardless of whether or not they have `requires_grad = True`;
            this allows a single EMA object to be consistantly used even
            if which parameters are trainable changes step to step.

            If you want to some parameters in the EMA, do not pass them
            to the object in the first place. For example:

                ExponentialMovingAverage(
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    decay=0.9
                )

            will ignore parameters that do not require grad.

        decay: The exponential decay.

        use_num_updates: Whether to use number of updates when computing
            averages.
    """

    def __init__(
        self,
        model_vars: Dict[str, TrainVar],
        decay: float,
        use_num_updates: bool = True,
        update_every: int = 1,
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = StateVar(jnp.array(decay))
        self.num_updates = StateVar(jnp.array(0, jnp.uint32))
        self.use_num_updates = use_num_updates
        self.shadow_params = ModuleList(
            StateVar(jnp.array(p.value.copy()))
            for p in model_vars.values()
            if isinstance(p, TrainVar)
        )
        self.model_vars = ModuleList(TrainRef(x) for x in model_vars.subset(TrainVar))
        assert len(self.shadow_params) == len(self.model_vars)
        self.update_every = jnp.array(update_every, jnp.uint32)

    # def parallel_reduce(self):
    #     self.shadow_params = pmean([sp for sp in self.shadow_params])

    def update(self) -> None:
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        decay = self.decay
        self.num_updates.value += 1.0
        if self.use_num_updates:
            decay = jnp.minimum(
                self.decay.value,
                (1.0 + self.num_updates.value) / (10.0 + self.num_updates.value),
            )
        one_minus_decay = 1.0 - decay
        for i, (p_s, param) in enumerate(zip(self.shadow_params, self.model_vars)):
            tmp = p_s - param.value
            tmp *= one_minus_decay
            self.shadow_params[i].value -= tmp

    def copy_to(self) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        for param, s_p in zip(self.model_vars, self.shadow_params):
            lax.cond(
                jnp.all(
                    jnp.isclose(
                        jnp.mod(self.num_updates.value, self.update_every),
                        zero_element_vector,
                    ),
                ),
                lambda _: param.assign(s_p.value),  # pylint:disable=cell-var-from-loop
                lambda _: None,  # pylint:disable=cell-var-from-loop
                operand=(),
            )

    def __call__(self):
        self.update()
        self.copy_to()
