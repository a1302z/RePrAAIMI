import functools
import jax
from jax import numpy as jnp

from typing import Optional, Callable, Tuple
from objax.gradient import GradValues
from objax.module import Function, Vectorize
from objax import random, ModuleList, StateVar
from objax.variable import VarCollection
from objax.privacy.dpsgd import PrivateGradValues


class ClipAndAccumulateGrads(PrivateGradValues):
    def __init__(
        self,
        loss_fn: Callable,
        variables: VarCollection,
        l2_norm_clip: float,
        batch_axis: Tuple[Optional[int], ...] = (0,),
        use_norm_accumulation: bool = False,
        gradient_accumulation_steps: int = 1,
        num_augmented_samples: int = 1,
    ):
        super().__init__(
            loss_fn,
            variables,
            0.0,
            l2_norm_clip,
            num_augmented_samples,
            batch_axis,
            use_norm_accumulation,
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.counter = StateVar(jnp.array(0, dtype=jnp.int32))
        # self.num_elements = StateVar(jnp.array(0, dtype=jnp.int32))
        self.accumulated_loss = StateVar(jnp.array(0.0))
        self.accumulated_grads = ModuleList(
            StateVar(jnp.zeros_like(tv)) for tv in variables
        )

    def calc_per_sample_grads(self, *args):
        self.counter.value += 1
        clipped_grad, loss_value = self.clipped_grad(*args)
        return clipped_grad, loss_value

    def accumulate_grad(self, clipped_grads, loss_values):
        assert len(clipped_grads) == len(self.accumulated_grads)
        for i, clipped_grad in enumerate(clipped_grads):
            self.accumulated_grads[i].value += clipped_grad
        assert len(loss_values) == 1, "We assumed only one loss term"
        self.accumulated_loss.value += loss_values[0]

    def get_accumulated_grads(self):
        # accumulated_grads = [jnp.copy(g) for g in self.accumulated_grads]
        # self._reset_values()
        # return accumulated_grads
        return [gx.value for gx in self.accumulated_grads]

    def reset_accumulated_grads(self):
        self._reset_values()

    def _reset_values(self):
        for i, accumulated_grad in enumerate(self.accumulated_grads):
            self.accumulated_grads[i].value = jnp.zeros_like(accumulated_grad)
        self.accumulated_loss.value = jnp.array(0.0)
        self.counter.value = jnp.array(0).astype(jnp.int32)

    def __call__(self, *args):
        """Returns the computed DP-SGD gradients.
        Note: This function is not serializable. Hence we recommend
        to use the single function calls which are serializable.

        Returns:
            A tuple (gradients, value of f)."""
        return self.calc_per_sample_grads(*args)

    @staticmethod
    def add_noise(grad, stddev, generator):
        return [
            gx + random.normal(gx.shape, stddev=stddev, generator=generator)
            for gx in grad
        ]

    def _make_clipped_grad_fn_simple(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, values = gv(*args)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            return [g * idivisor for g in grads], values

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(*args):
            grads, loss = clipped_grad_vectorized(*args)
            grads, loss = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads, loss)
            )
            return grads, loss

        return clipped_grad

    def _make_clipped_grad_fn_norm_accumulation(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def grad_norm_fn(*args):
            grads, values = gv(*args)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            return total_grad_norm, values

        grad_norm_fn_vectorized = Vectorize(grad_norm_fn, batch_axis=self.batch_axis)
        loss_fn_vectorized = Vectorize(
            Function.with_vars(gv.vars())(f), batch_axis=self.batch_axis
        )

        def weighted_loss_fn(weights, *args):
            returned = loss_fn_vectorized(*args)
            # the following assumes that the loss is the first element output by loss_fn_vectorized
            if isinstance(returned, jnp.ndarray):
                loss_vector = returned
            else:
                loss_vector = returned[0]
            return jnp.mean(loss_vector * weights)

        weighted_grad_fn = GradValues(weighted_loss_fn, vc)

        def clipped_grad(*args):
            grad_norms, values = grad_norm_fn_vectorized(*args)
            idivisor = 1 / jnp.maximum(grad_norms / self.l2_norm_clip, 1.0)
            clipped_grads, _ = weighted_grad_fn(idivisor, *args)
            return (
                clipped_grads,
                jax.tree_map(functools.partial(jnp.sum, axis=0), values),
            )

        return clipped_grad
