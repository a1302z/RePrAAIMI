from jax import numpy as jnp
from objax.privacy.dpsgd import PrivateGradValues


from typing import Optional, Callable, Tuple


from objax import random, ModuleList, StateVar
from objax.variable import VarCollection


class GradAccumulatorSimple:
    """This is just as training with a smaller batch size but with extra steps"""

    def __init__(self, train_vars, update_frequency=1) -> None:
        self.grad_container = [jnp.zeros_like(tv) for tv in train_vars]
        self.counter = 0
        self.update_frequency = update_frequency

    def step(self, opt, grads, args, force_apply=True, **kwargs):
        assert len(self.grad_container) == len(grads)
        self.counter += 1
        for i in range(len(self.grad_container)):
            self.grad_container[i] += grads[i]
        if self.counter % self.update_frequency == 0 or force_apply:
            opt(grads=self.grad_container, *args, **kwargs)
            self.counter = 0
            for i in range(len(self.grad_container)):
                self.grad_container[i] = jnp.zeros_like(self.grad_container[i])


class PrivateGradValuesAccumulation(PrivateGradValues):
    def __init__(
        self,
        f: Callable,
        vc: VarCollection,
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
            f,
            vc,
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
        self.accumulated_grads = ModuleList(StateVar(jnp.zeros_like(tv)) for tv in vc)

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
        g, v = self.clipped_grad(*[self.reshape_microbatch(x) for x in args])
        return batch, stddev, g, v

    def no_acc_step(self, clipped_grad, stddev):
        """To be called after setup_grad_step"""
        noised_clipped_grad = self._add_noise(clipped_grad, stddev)
        return noised_clipped_grad

    def accumulate_grad(self, clipped_grad, batch, v):
        for i in range(len(self.accumulated_grads)):
            self.accumulated_grads[i].value += clipped_grad[i] * batch
        assert len(v) == 1, "We assumed only one loss term"
        self.accumulated_loss.value += v[0]
        self.num_elements.value += batch

    def apply_accumulated_grads(self, stddev):
        noised_clipped_grad = self._add_noise(
            self.accumulated_grads,
            stddev,
            grad_scale_factor=1.0 / self.num_elements.value,
        )
        v = [self.accumulated_loss.value]
        # reset all variables
        for i in range(len(self.accumulated_grads)):
            self.accumulated_grads[i].value = jnp.zeros_like(self.accumulated_grads[i])
        self.accumulated_loss.value = jnp.array(0.0)
        self.counter.value = jnp.array(0).astype(jnp.int32)
        self.num_elements.value = jnp.array(0).astype(jnp.int32)
        return noised_clipped_grad, v

    def is_gradient_accumulated(self) -> bool:
        return self.counter.value.item() % self.gradient_accumulation_steps == 0

    def __call__(self, *args):
        """Returns the computed DP-SGD gradients.
        Note: This function is not serializable. Hence we recommend
        to use the single function calls which are serializable.

        Returns:
            A tuple (gradients, value of f)."""
        batch, stddev, clipped_grad, v = self.setup_grad_step(*args)
        if self.gradient_accumulation_steps == 1:
            noised_clipped_grad = self.no_acc_step(clipped_grad, stddev)
            return noised_clipped_grad, v
        self.accumulate_grad(clipped_grad, batch, v)
        if self.is_gradient_accumulated():
            return self.apply_accumulated_grads(stddev)
        return [jnp.zeros_like(tv) for tv in clipped_grad], 0.0
