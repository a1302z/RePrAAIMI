from __future__ import division
from __future__ import unicode_literals

import numpy as np
from jax import numpy as jnp

from typing import Optional, Dict
from objax import TrainVar, Module, StateVar, ModuleList

# import torch


# Partially based on:
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
# Objax version based on:
# https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py
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
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = StateVar(jnp.array(decay))
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = ModuleList(
            StateVar(jnp.array(p.value.copy())) for p in model_vars.values()
        )
        self.collected_params = None
        # By maintaining only a weakref to each parameter,
        # we maintain the old GC behaviour of ExponentialMovingAverage:
        # if the model goes out of scope but the ExponentialMovingAverage
        # is kept, no references to the model or its parameters will be
        # maintained, and the model will be cleaned up.
        # self._params_refs = {k: weakref.ref(p) for k, p in model_vars.items()}

    # def _get_parameters(
    #     self, model_vars: Optional[Dict[str, TrainVar]]
    # ) -> Dict[str, TrainVar]:
    #     if model_vars is None:
    #         model_vars = {k: p() for k, p in self._params_refs.items()}
    #         if any(p is None for p in model_vars.values()):
    #             raise ValueError(
    #                 "(One of) the parameters with which this "
    #                 "ExponentialMovingAverage "
    #                 "was initialized no longer exists (was garbage collected);"
    #                 " please either provide `parameters` explicitly or keep "
    #                 "the model to which they belong from being garbage "
    #                 "collected."
    #             )
    #         return model_vars
    #     else:
    #         model_vars = dict(model_vars)
    #         if len(model_vars) != len(self.shadow_params):
    #             raise ValueError(
    #                 "Number of parameters passed as argument is different "
    #                 "from number of shadow parameters maintained by this "
    #                 "ExponentialMovingAverage"
    #             )
    #         return model_vars

    def update(self, parameters: Optional[Dict[str, TrainVar]] = None) -> None:
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
        # parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        for i, (p_s, param) in enumerate(zip(self.shadow_params, parameters.values())):
            if len(param.shape) == len(p_s.shape) + 1:
                tmp = p_s - np.mean(param, axis=0)
            else:
                tmp = p_s - param
            # tmp will be a new tensor so we can do in-place
            tmp *= one_minus_decay
            self.shadow_params[i].value -= tmp

    def copy_to(self, parameters: Optional[Dict[str, TrainVar]] = None) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        # parameters = self._get_parameters(parameters)
        for (k, param), s_p in zip(parameters.items(), self.shadow_params):
            # param.data.copy_(s_param.data)
            if len(param.shape) == len(s_p.shape) + 1:
                parameters[k].assign(
                    jnp.array(
                        jnp.repeat(
                            s_p.value[jnp.newaxis, :],
                            param.shape[0],
                            axis=0,
                        )
                    )
                )
            else:
                parameters[k].assign(s_p.value)

    def __call__(self, parameters):
        self.update(parameters)
        self.copy_to(parameters)

    # def store(self, parameters: Optional[Dict[str, TrainVar]] = None) -> None:
    #     """
    #     Save the current parameters for restoring later.

    #     Args:
    #         parameters: Iterable of `torch.nn.Parameter`; the parameters to be
    #             temporarily stored. If `None`, the parameters of with which this
    #             `ExponentialMovingAverage` was initialized will be used.
    #     """
    #     parameters = self._get_parameters(parameters)
    #     self.collected_params = {
    #         k: param.value.copy() for k, param in parameters.items()
    #     }

    # def restore(self, parameters: Optional[Dict[str, TrainVar]] = None) -> None:
    #     """
    #     Restore the parameters stored with the `store` method.
    #     Useful to validate the model with EMA parameters without affecting the
    #     original optimization process. Store the parameters before the
    #     `copy_to` method. After validation (or model saving), use this to
    #     restore the former parameters.

    #     Args:
    #         parameters: Iterable of `torch.nn.Parameter`; the parameters to be
    #             updated with the stored parameters. If `None`, the
    #             parameters with which this `ExponentialMovingAverage` was
    #             initialized will be used.
    #     """
    #     if self.collected_params is None:
    #         raise RuntimeError(
    #             "This ExponentialMovingAverage has no `store()`ed weights "
    #             "to `restore()`"
    #         )
    #     parameters = self._get_parameters(parameters)
    #     for k in parameters.keys():
    #         # param.data.copy_(c_param.data)
    #         parameters[k].assign(self.collected_params[k])

    # @contextlib.contextmanager
    # def average_parameters(self, parameters: Optional[Dict[str, TrainVar]] = None):
    #     r"""
    #     Context manager for validation/inference with averaged parameters.

    #     Equivalent to:

    #         ema.store()
    #         ema.copy_to()
    #         try:
    #             ...
    #         finally:
    #             ema.restore()

    #     Args:
    #         parameters: Iterable of `torch.nn.Parameter`; the parameters to be
    #             updated with the stored parameters. If `None`, the
    #             parameters with which this `ExponentialMovingAverage` was
    #             initialized will be used.
    #     """
    #     parameters = self._get_parameters(parameters)
    #     self.store(parameters)
    #     self.copy_to(parameters)
    #     try:
    #         yield
    #     finally:
    #         self.restore(parameters)

    # def to(self, device=None, dtype=None) -> None:
    #     r"""Move internal buffers of the ExponentialMovingAverage to `device`.

    #     Args:
    #         device: like `device` argument to `torch.Tensor.to`
    #     """
    #     # .to() on the tensors handles None correctly
    #     self.shadow_params = [
    #         p.to(device=device, dtype=dtype)
    #         if p.is_floating_point()
    #         else p.to(device=device)
    #         for p in self.shadow_params
    #     ]
    #     if self.collected_params is not None:
    #         self.collected_params = [
    #             p.to(device=device, dtype=dtype)
    #             if p.is_floating_point()
    #             else p.to(device=device)
    #             for p in self.collected_params
    #         ]
    #     return

    # def state_dict(self) -> dict:
    #     r"""Returns the state of the ExponentialMovingAverage as a dict."""
    #     # Following PyTorch conventions, references to tensors are returned:
    #     # "returns a reference to the state and not its copy!" -
    #     # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
    #     return {
    #         "decay": self.decay,
    #         "num_updates": self.num_updates,
    #         "shadow_params": self.shadow_params,
    #         "collected_params": self.collected_params,
    #     }

    # def load_state_dict(self, state_dict: dict) -> None:
    #     r"""Loads the ExponentialMovingAverage state.

    #     Args:
    #         state_dict (dict): EMA state. Should be an object returned
    #             from a call to :meth:`state_dict`.
    #     """
    #     # deepcopy, to be consistent with module API
    #     state_dict = copy.deepcopy(state_dict)
    #     self.decay = state_dict["decay"]
    #     if self.decay < 0.0 or self.decay > 1.0:
    #         raise ValueError("Decay must be between 0 and 1")
    #     self.num_updates = state_dict["num_updates"]
    #     assert self.num_updates is None or isinstance(
    #         self.num_updates, int
    #     ), "Invalid num_updates"

    #     self.shadow_params = state_dict["shadow_params"]
    #     assert isinstance(self.shadow_params, list), "shadow_params must be a list"
    #     # assert all(
    #     #     isinstance(p, torch.Tensor) for p in self.shadow_params
    #     # ), "shadow_params must all be Tensors"

    #     self.collected_params = state_dict["collected_params"]
    #     if self.collected_params is not None:
    #         assert isinstance(
    #             self.collected_params, list
    #         ), "collected_params must be a list"
    #         # assert all(
    #         #     isinstance(p, torch.Tensor) for p in self.collected_params
    #         # ), "collected_params must all be Tensors"
    #         assert len(self.collected_params) == len(
    #             self.shadow_params
    #         ), "collected_params and shadow_params had different lengths"

    #     if len(self.shadow_params) == len(self._params_refs):
    #         # Consistant with torch.optim.Optimizer, cast things to consistant
    #         # device and dtype with the parameters
    #         params = [p() for p in self._params_refs]
    #         # If parameters have been garbage collected, just load the state
    #         # we were given without change.
    #         if not any(p is None for p in params):
    #             # ^ parameter references are still good
    #             for i, p in enumerate(params):
    #                 self.shadow_params[i] = self.shadow_params[i].to(
    #                     device=p.device, dtype=p.dtype
    #                 )
    #                 if self.collected_params is not None:
    #                     self.collected_params[i] = self.collected_params[i].to(
    #                         device=p.device, dtype=p.dtype
    #                     )
    #     else:
    #         raise ValueError(
    #             "Tried to `load_state_dict()` with the wrong number of "
    #             "parameters in the saved state."
    #         )
