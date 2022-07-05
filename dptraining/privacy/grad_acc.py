from jax import numpy as jnp
from objax.privacy.dpsgd import PrivateGradValues


from typing import Optional, Callable, Tuple


from objax import random, ModuleList, StateVar
from objax.variable import VarCollection


class PrivateGradValuesAccumulation(PrivateGradValues):
    def __init__(
        self,
        loss_fn: Callable,
        variables: VarCollection,
        noise_multiplier: float,
        l2_norm_clip: float,
        microbatch: int,
        batch_axis: Tuple[Optional[int], ...] = (0,),
        keygen: random.Generator = random.DEFAULT_GENERATOR,
        use_norm_accumulation: bool = False,
        gradient_accumulation_steps: int = 1,
        noise_scaling_factor: float = 1.0,
    ):
        super().__init__(
            loss_fn,
            variables,
            noise_multiplier,
            l2_norm_clip,
            microbatch,
            batch_axis,
            keygen,
            use_norm_accumulation,
        )
        self.noise_scaling_factor = noise_scaling_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.counter = StateVar(jnp.array(0, dtype=jnp.int32))
        self.num_elements = StateVar(jnp.array(0, dtype=jnp.int32))
        self.accumulated_loss = StateVar(jnp.array(0.0))
        self.accumulated_grads = ModuleList(
            StateVar(jnp.zeros_like(tv)) for tv in variables
        )

    def _calc_std(self, num_microbatches):
        return (
            self.l2_norm_clip * self.noise_multiplier / num_microbatches
        ) * self.noise_scaling_factor

    def _add_noise(self, grad, stddev, grad_scale_factor=1.0):
        return [
            gx * grad_scale_factor
            + random.normal(gx.shape, stddev=stddev, generator=self.keygen)
            for gx in grad
        ]

    def setup_grad_step(self, *args):
        self.counter.value += 1
        batch = args[0].shape[0]
        assert batch % self.microbatch == 0
        num_microbatches = batch // self.microbatch
        stddev = self._calc_std(num_microbatches)
        clipped_grad, loss_value = self.clipped_grad(
            *[self.reshape_microbatch(x) for x in args]
        )
        return batch, stddev, clipped_grad, loss_value

    def no_acc_step(self, clipped_grad, stddev):
        """To be called after setup_grad_step"""
        noised_clipped_grad = self._add_noise(clipped_grad, stddev)
        return noised_clipped_grad

    def accumulate_grad(self, clipped_grads, batch, loss_values):
        assert len(clipped_grads) == len(self.accumulated_grads)
        for i, clipped_grad in enumerate(clipped_grads):
            self.accumulated_grads[i].value += clipped_grad * batch
        assert len(loss_values) == 1, "We assumed only one loss term"
        self.accumulated_loss.value += loss_values[0]
        self.num_elements.value += batch

    def apply_accumulated_grads(self, stddev):
        noised_clipped_grad = self._add_noise(
            self.accumulated_grads,
            stddev,
            grad_scale_factor=1.0 / self.num_elements.value,
        )
        # to be conform with standard function
        loss_value = (self.accumulated_loss.value,)
        self._reset_values()
        return noised_clipped_grad, loss_value

    def _reset_values(self):
        for i, accumulated_grad in enumerate(self.accumulated_grads):
            self.accumulated_grads[i].value = jnp.zeros_like(accumulated_grad)
        self.accumulated_loss.value = jnp.array(0.0)
        self.counter.value = jnp.array(0).astype(jnp.int32)
        self.num_elements.value = jnp.array(0).astype(jnp.int32)

    def is_gradient_accumulated(self) -> bool:
        return self.counter.value.item() % self.gradient_accumulation_steps == 0

    def __call__(self, *args):
        """Returns the computed DP-SGD gradients.
        Note: This function is not serializable. Hence we recommend
        to use the single function calls which are serializable.

        Returns:
            A tuple (gradients, value of f)."""
        batch, stddev, clipped_grad, loss_value = self.setup_grad_step(*args)
        if self.gradient_accumulation_steps == 1:
            noised_clipped_grad = self.no_acc_step(clipped_grad, stddev)
            return noised_clipped_grad, loss_value
        self.accumulate_grad(clipped_grad, batch, loss_value)
        if self.is_gradient_accumulated():
            return self.apply_accumulated_grads(stddev)
        return [jnp.zeros_like(tv) for tv in clipped_grad], 0.0
