from typing import List

from jax import numpy as jn

from objax import functional
from objax.module import Module, ModuleList
from objax.typing import JaxArray
from objax.variable import TrainRef, StateVar, TrainVar, VarCollection


class NAdam(Module):  # pylint:disable=too-many-instance-attributes
    def __init__(  # pylint:disable=too-many-arguments
        self,
        model_vars: VarCollection,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 0.004,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lmbda = weight_decay
        self.psi = momentum_decay
        self.step = StateVar(jn.array(0, jn.uint32), reduce=lambda x: x[0])
        self.train_vars = ModuleList(TrainRef(x) for x in model_vars.subset(TrainVar))
        self.momentum = ModuleList(
            StateVar(jn.zeros_like(x.value)) for x in self.train_vars
        )
        self.velocity = ModuleList(
            StateVar(jn.zeros_like(x.value)) for x in self.train_vars
        )
        # self.mu = ModuleList(StateVar(jn.ones_like(x.value)) for x in self.train_vars)
        self.cumulative_mu = ModuleList(
            StateVar(jn.ones_like(x.value)) for x in self.train_vars
        )
        # for mu, c_mu in zip(self.mu, self.c_mu):
        for c_mu in self.cumulative_mu:
            mult = self.beta1 * (1 - 0.5 * 0.96**self.psi)
            c_mu.value *= mult

    def __call__(  # pylint:disable=arguments-differ # no clue why this warning comes up after all
        self,
        lr: float,
        grads: List[JaxArray],
    ):
        assert len(grads) == len(
            self.train_vars
        ), "Expecting as many gradients as trainable variables"
        self.step.value += 1
        # lr *= jn.sqrt(1 - beta2**self.step.value) / (1 - beta1**self.step.value)
        # for g, p, m, v in zip(grads, self.train_vars, self.m, self.v):
        #     m.value += (1 - beta1) * (g - m.value)
        #     v.value += (1 - beta2) * (g**2 - v.value)
        #     p.value -= lr * m.value * functional.rsqrt(v.value + self.eps)
        mu = self.beta1 * (  # pylint:disable=invalid-name
            1.0 - 0.5 * 0.96 ** (self.step.value * self.psi)
        )
        for grad, param, momentum, veloc, cumul_mu in zip(
            grads, self.train_vars, self.momentum, self.velocity, self.cumulative_mu
        ):
            grad += self.lmbda * param.value
            momentum.value += (1 - self.beta1) * (grad - momentum.value)
            veloc.value += (1 - self.beta2) * (grad**2 - veloc.value)

            pre_update = (1.0 - mu) * grad / (1 - cumul_mu.value)
            mu = self.beta1 * (  # pylint:disable=invalid-name
                1.0 - 0.5 * 0.96 ** ((self.step.value + 1) * self.psi)
            )
            cumul_mu.value *= mu
            post_update = (mu * momentum.value) / (1.0 - cumul_mu.value)

            m_hat = pre_update + post_update
            v_hat = veloc.value / (1.0 - self.beta2**self.step.value)

            param.value -= lr * m_hat * functional.rsqrt(v_hat + self.eps)

    def __repr__(self):
        return f"NAdam(beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
