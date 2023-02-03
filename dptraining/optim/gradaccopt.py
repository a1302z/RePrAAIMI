from jax import numpy as jnp

from typing import Optional, Callable, Tuple, Union
from objax.gradient import GradValues
from objax.module import Module
from objax import ModuleList, StateVar
from objax.variable import VarCollection


class AccumulateGrad(Module):
    def __init__(
        self,
        f: Union[Module, Callable],
        variables: Optional[VarCollection],
        input_argnums: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.counter = StateVar(jnp.array(0, dtype=jnp.int32))
        self.accumulated_loss = StateVar(jnp.array(0.0))
        self.accumulated_grads = ModuleList(
            StateVar(jnp.zeros_like(tv)) for tv in variables
        )
        self.gv = GradValues(f, variables, input_argnums)

    def accumulate_grad(self, grads, loss_values):
        assert len(grads) == len(self.accumulated_grads)
        for i, grad in enumerate(grads):
            self.accumulated_grads[i].value += grad
        assert len(loss_values) == 1, "We assumed only one loss term"
        self.accumulated_loss.value += loss_values[0]

    def get_accumulated_grads(self):
        return [gx.value for gx in self.accumulated_grads]

    def reset_accumulated_grads(self):
        self._reset_values()

    def _reset_values(self):
        for i, accumulated_grad in enumerate(self.accumulated_grads):
            self.accumulated_grads[i].value = jnp.zeros_like(accumulated_grad)
        self.accumulated_loss.value = jnp.array(0.0)
        self.counter.value = jnp.array(0).astype(jnp.int32)

    def __call__(self, *args, **kwargs):
        return self.gv(*args, **kwargs)
