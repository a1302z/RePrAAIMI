from objax.privacy.dpsgd import PrivateGradValues
from objax import random
from jax.lax import rsqrt

one_by_sqrt_two = rsqrt(2.0)

# pylint:disable=all




class ComplexPrivateGradValues(PrivateGradValues):
    """Computes differentially private gradients as required by DP-SGD.
    This module can be used in place of GradVals, and automatically makes
    the optimizer differentially private."""

    def __call__(self, *args):
        """Returns the computed DP-SGD gradients.

        Returns:
            A tuple (gradients, value of f)."""
        batch = args[0].shape[0]
        assert batch % self.microbatch == 0
        num_microbatches = batch // self.microbatch
        stddev = self.l2_norm_clip * self.noise_multiplier / num_microbatches
        g, v = self.clipped_grad(*[self.reshape_microbatch(x) for x in args])
        g = [
            gx
            + (
                random.normal(
                    gx.shape, stddev=stddev * one_by_sqrt_two, generator=self.keygen
                )
            )
            for gx in g
        ]
        return g, v
