import jax, objax, functools
from jax import numpy as jnp

from typing import Optional, Callable, Tuple
from objax.gradient import GradValues
from objax.module import Function, Vectorize
from objax import random, ModuleList, StateVar, Module
from objax.variable import VarCollection


class ClipAndAccumulateGrads(Module):
    def __init__(
        self,
        loss_fn: Callable,
        variables: VarCollection,
        l2_norm_clip: float,
        batch_axis: Tuple[Optional[int], ...] = (0,),
        use_norm_accumulation: bool = False,
        gradient_accumulation_steps: int = 1,
        num_augmented_samples: int = 1,
        log_grad_metrics: bool = True,
        bam:bool=False,
        r:float=0.05,
        alpha=0.8,
    ):
        super().__init__()
        self.batch_axis = batch_axis
        self.l2_norm_clip = l2_norm_clip
        self.log_grad_metrics = log_grad_metrics
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.counter = StateVar(jnp.array(0, dtype=jnp.int32))
        # self.num_elements = StateVar(jnp.array(0, dtype=jnp.int32))
        self.accumulated_grads = ModuleList(
            StateVar(jnp.zeros_like(tv)) for tv in variables
        )
        gv = GradValues(loss_fn, variables)
        if not bam:
            if use_norm_accumulation:
                self.clipped_grad = self.make_clipped_grad_fn_norm_accumulation(
                    loss_fn, gv, variables
                )
            else:
                self.clipped_grad = self.make_clipped_grad_fn_simple(
                    loss_fn, gv, variables
                )
        else:
            print("... creating clip function for BAM")
            self.clipped_grad = self.make_clipped_grad_fn_bam(
                    loss_fn, gv, variables
                )
            self.r = r
            self.alpha = alpha

    def calc_per_sample_grads(self, *args):
        self.counter.value += 1
        clipped_grad, loss_value = self.clipped_grad(*args)
        return clipped_grad, loss_value

    def accumulate_grad(self, clipped_grads):
        assert len(clipped_grads) == len(self.accumulated_grads)
        for i, clipped_grad in enumerate(clipped_grads):
            self.accumulated_grads[i].value += clipped_grad

    def get_accumulated_grads(self):
        return [gx.value for gx in self.accumulated_grads]

    def reset_accumulated_grads(self):
        self._reset_values()

    def _reset_values(self):
        for i, accumulated_grad in enumerate(self.accumulated_grads):
            self.accumulated_grads[i].value = jnp.zeros_like(accumulated_grad)
        self.counter.value = jnp.array(0).astype(jnp.int32)

    def cos_sim(self, g1, g2):
        """Computes cosine similarity between g1 and g2"""
        flatten = lambda grad_list: jnp.concatenate(
            [g.flatten() for g in grad_list], axis=0
        )
        dot_product = jnp.dot(flatten(g1), flatten(g2))
        norm_g1 = jnp.linalg.norm(jnp.array([jnp.linalg.norm(g) for g in g1]))
        norm_g2 = jnp.linalg.norm(jnp.array([jnp.linalg.norm(g) for g in g2]))
        return dot_product / (norm_g1 * norm_g2)

    def l2_bias(self, g, g_clip):
        """Computes L2-norm of bias vector ||Bias(\hat{g}, \hat{g}_priv)||_2"""
        flatten = lambda grad_list: jnp.concatenate(
            [g.flatten() for g in grad_list], axis=0
        )
        return jnp.linalg.norm(flatten(g_clip) - flatten(g))

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

    def make_clipped_grad_fn_simple(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, values = gv(*args)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            unclipped_grads = grads if self.log_grad_metrics else None
            return [g * idivisor for g in grads], values, unclipped_grads

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(*args):
            grads, loss, unclipped_grads = clipped_grad_vectorized(*args)
            grads, loss = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads, loss)
            )
            if self.log_grad_metrics:
                unclipped_grads = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (unclipped_grads)
                )
                stoch_alignment = self.cos_sim(unclipped_grads, grads)
                bias = self.l2_bias(unclipped_grads, grads)

                metric_dict = {"stoch_aligment": stoch_alignment, "l2_bias": bias}
                values = (loss, metric_dict)
            else:
                values = (loss, {})
            return grads, values

        return clipped_grad

    def make_clipped_grad_fn_norm_accumulation(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        if self.log_grad_metrics:
            raise NotImplementedError(
                "logging grad metrics isn't supported for norm accumulation yet"
            )

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
            loss = jax.tree_map(functools.partial(jnp.sum, axis=0), values)
            values = (loss, {})
            return (
                clipped_grads,
                values,
            )

        return clipped_grad
    
    def make_clipped_grad_fn_bam(
            self, f: Callable, gv: GradValues, vc: VarCollection
        ) -> Callable:
            @Function.with_vars(gv.vars())
            def clipped_grad_single_example(*args):
                grads, values = gv(*args)
                grad_norm = jnp.linalg.norm(
                    jnp.array([jnp.linalg.norm(g) for g in grads])
                )
                # pertub parameters by scaled gradient
                for v, g in zip(vc, grads):
                    v.assign(v.value + (self.r * g / grad_norm))
                l_dash_grad_fn = GradValues(f, vc)
                l_dash_grads, _ = l_dash_grad_fn(*args)
                # BAM gradient
                total_grad = [
                    ((1 - self.alpha) * g) + (self.alpha * dash_grad)
                    for g, dash_grad in zip(grads, l_dash_grads)
                ]
                # undo parameter perturbation
                for v, g in zip(vc, grads):
                    v.assign(v.value - (self.r * g / grad_norm))
                del grads
                del l_dash_grads
                # clipping business as usual
                total_grad_norm = jnp.linalg.norm(
                    jnp.array([jnp.linalg.norm(g) for g in total_grad])
                )
                idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
                unclipped_grads = grads if self.log_grad_metrics else None
                return [g * idivisor for g in total_grad], values, unclipped_grads

            clipped_grad_vectorized = Vectorize(
                clipped_grad_single_example, batch_axis=self.batch_axis
            )

            def clipped_grad(*args):
                grads, loss, unclipped_grads = clipped_grad_vectorized(*args)
                grads, loss = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (grads, loss)
                )
                if self.log_grad_metrics:
                    unclipped_grads = jax.tree_util.tree_map(
                        functools.partial(jnp.sum, axis=0), (unclipped_grads)
                    )
                    stoch_alignment = self.cos_sim(unclipped_grads, grads)
                    bias = self.l2_bias(unclipped_grads, grads)
                    metric_dict = {"stoch_aligment": stoch_alignment, "l2_bias": bias}
                    values = (loss, metric_dict)
                else:
                    values = (loss, {})
                return grads, values

            return clipped_grad