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
        bam: bool = False,
        SAT: bool = False,
        double_backprop: bool = False,
        r: float = 0.05,
    ):
        super().__init__()
        self.SAT = SAT
        self.batch_axis = batch_axis
        self.l2_norm_clip = l2_norm_clip
        self.log_grad_metrics = log_grad_metrics
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.counter = StateVar(jnp.array(0, dtype=jnp.int32))
        # self.num_elements = StateVar(jnp.array(0, dtype=jnp.int32))
        self.accumulated_grads = ModuleList(
            StateVar(jnp.zeros_like(tv)) for tv in variables
        )
        self.prev_grad = None
        self.prev_grad_norm = None
        self.r = r
        gv = GradValues(loss_fn, variables)
        if use_norm_accumulation:
            raise NotImplementedError()
        if bam:
            print("... creating clip function for BAM")
            if double_backprop:
                self.clipped_grad = self.make_clipped_grad_fn_bam_db(
                    loss_fn, gv, variables
                )
            else:
                self.clipped_grad = self.make_clipped_grad_fn_bam(
                    loss_fn, gv, variables
                )
        elif SAT:
            self.clipped_grad = self.make_clipped_grad_fn_SAT(loss_fn, gv, variables)
        else:
            self.clipped_grad = self.make_clipped_grad_fn_simple(loss_fn, gv, variables)

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

    def cos_sim(self, g1, g2, batch_size: int):
        """Computes cosine similarity between g1 and g2"""
        flatten = lambda grad_list: jnp.concatenate(
            [g.flatten() for g in grad_list], axis=0
        )
        g1 = [g / batch_size for g in g1]
        g2 = [g / batch_size for g in g2]
        return jnp.dot(flatten(g1), flatten(g2)) / (
            jnp.linalg.norm(flatten(g1)) * jnp.linalg.norm(flatten(g2))
        )

    def l2_bias(self, g, g_clip, batch_size: int):
        """Computes L2-norm of bias vector ||Bias(\hat{g}, \hat{g}_priv)||_2"""
        flatten = lambda grad_list: jnp.concatenate(
            [g.flatten() for g in grad_list], axis=0
        )
        g = [pg / batch_size for pg in g]
        g_clip = [pg / batch_size for pg in g_clip]
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
        self,
        f: Callable,
        gv: GradValues,
        vc: VarCollection,
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, loss = gv(*args)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            unclipped_grads = grads if self.log_grad_metrics else None
            values = (loss, {"sample_grad_norm": total_grad_norm})
            return [g * idivisor for g in grads], values, unclipped_grads

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(images, targets):
            grads, values, unclipped_grads = clipped_grad_vectorized(
                images, targets
            )
            batch_size = grads[0].shape[0]
            grads = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads)
            )  # sum for grads
            values = jax.tree_util.tree_map(
                functools.partial(jnp.mean, axis=0), (values)
            )  # mean for metrics
            if self.log_grad_metrics:
                unclipped_grads = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (unclipped_grads)
                )
                stoch_alignment = self.cos_sim(
                    unclipped_grads, grads, batch_size=batch_size
                )
                bias = self.l2_bias(unclipped_grads, grads, batch_size=batch_size)
                values[1]["stoch_aligment"] = stoch_alignment
                values[1]["l2_bias"] = bias
            return grads, values

        return clipped_grad

    def make_clipped_grad_fn_SAT(
        self,
        f: Callable,
        gv: GradValues,
        vc: VarCollection,
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(image, target, prev_grad, prev_grad_norm):
            if self.prev_grad is not None:
                # pertub parameters by scaled gradient
                for v, g in zip(vc, prev_grad):
                    v.assign(v.value + (self.r * g / prev_grad_norm + 1e-6))
            grads, loss = gv(image, target)
            if self.prev_grad is not None:
                # undo parameter pertubation
                for v, g in zip(vc, prev_grad):
                    v.assign(v.value - (self.r * g / prev_grad_norm + 1e-6))

            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            unclipped_grads = grads if self.log_grad_metrics else None
            values = (loss, {"sample_grad_norm": total_grad_norm})
            return [g * idivisor for g in grads], values, unclipped_grads

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(images, targets, prev_grad, prev_grad_norm):
            grads, values, unclipped_grads = clipped_grad_vectorized(
                images, targets, prev_grad, prev_grad_norm
            )
            batch_size = grads[0].shape[0]
            grads = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads)
            )  # sum for grads
            values = jax.tree_util.tree_map(
                functools.partial(jnp.mean, axis=0), (values)
            )  # mean for metrics
            if self.log_grad_metrics:
                unclipped_grads = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (unclipped_grads)
                )
                stoch_alignment = self.cos_sim(
                    unclipped_grads, grads, batch_size=batch_size
                )
                bias = self.l2_bias(unclipped_grads, grads, batch_size=batch_size)
                values[1]["stoch_aligment"] = stoch_alignment
                values[1]["l2_bias"] = bias
            return grads, values

        return clipped_grad

    def make_clipped_grad_fn_bam(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, _ = gv(*args)
            grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(g) for g in grads]))
            # pertub parameters by scaled gradient
            for v, g in zip(vc, grads):
                v.assign(v.value + (self.r * g / grad_norm))
            l_dash_grad_fn = GradValues(f, vc)
            bam_grads, loss = l_dash_grad_fn(*args)
            # undo parameter perturbation
            for v, g in zip(vc, grads):
                v.assign(v.value - (self.r * g / grad_norm))
            del grads
            # clipping business as usual
            total_grad_norm = jnp.linalg.norm(
                jnp.array([jnp.linalg.norm(g) for g in bam_grads])
            )
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            unclipped_grads = bam_grads if self.log_grad_metrics else None
            values = (loss, {"sample_grad_norm": total_grad_norm})
            return [g * idivisor for g in bam_grads], values, unclipped_grads

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(*args):
            grads, values, unclipped_grads = clipped_grad_vectorized(*args)
            batch_size = grads[0].shape[0]
            grads = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads)
            )  # sum for grads
            values = jax.tree_util.tree_map(
                functools.partial(jnp.mean, axis=0), (values)
            )  # mean for metrics
            if self.log_grad_metrics:
                unclipped_grads = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (unclipped_grads)
                )
                stoch_alignment = self.cos_sim(
                    unclipped_grads, grads, batch_size=batch_size
                )
                bias = self.l2_bias(unclipped_grads, grads, batch_size=batch_size)
                values[1]["stoch_aligment"] = stoch_alignment
                values[1]["l2_bias"] = bias
            return grads, values

        return clipped_grad

    def make_clipped_grad_fn_bam_db(
        self, f: Callable, gv: GradValues, vc: VarCollection
    ) -> Callable:
        """Creates a function that computes gradients clipped per-sample.
        This algorithm vectorizes the computation of a clipped gradient defined over a single sample.
        Args:
          f: the function for which to compute gradients.
          gv: the GradValues object that computes non-clipped gradients.
          vc: the variables for which to compute gradients.
        Returns:
          clipped_grad: the function computing the average of gradients clipped per-sample.
        """
        vec_gv = objax.Vectorize(gv, batch_axis=(0, 0))

        @Function.with_vars(gv.vars())
        def reg_loss(*args):
            grads, loss = gv(*args)
            total_grad_norm = jnp.linalg.norm(
                jnp.array([jnp.linalg.norm(g) for g in grads])
            )
            return loss[0] + (self.lambda_reg * total_grad_norm)

        reg_grad_fn = GradValues(reg_loss, vc)

        @Function.with_vars(gv.vars())
        def clipped_grad_single_example(*args):
            grads, loss = reg_grad_fn(*args)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
            idivisor = 1 / jnp.maximum(total_grad_norm / self.l2_norm_clip, 1.0)
            unclipped_grads = grads if self.log_grad_metrics else None
            values = (loss, {"sample_grad_norm": total_grad_norm})
            return [g * idivisor for g in grads], values, unclipped_grads

        clipped_grad_vectorized = Vectorize(
            clipped_grad_single_example, batch_axis=self.batch_axis
        )

        def clipped_grad(*args):
            grads, values, unclipped_grads = clipped_grad_vectorized(*args)
            batch_size = grads[0].shape[0]
            grads = jax.tree_util.tree_map(
                functools.partial(jnp.sum, axis=0), (grads)
            )  # sum for grads
            values = jax.tree_util.tree_map(
                functools.partial(jnp.mean, axis=0), (values)
            )  # mean for metrics
            if self.log_grad_metrics:
                unclipped_grads = jax.tree_util.tree_map(
                    functools.partial(jnp.sum, axis=0), (unclipped_grads)
                )
                stoch_alignment = self.cos_sim(
                    unclipped_grads, grads, batch_size=batch_size
                )
                bias = self.l2_bias(unclipped_grads, grads, batch_size=batch_size)
                values[1]["stoch_aligment"] = stoch_alignment
                values[1]["l2_bias"] = bias
            return grads, values

        return clipped_grad
