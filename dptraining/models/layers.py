from objax import Module
from objax.typing import JaxArray
from objax.functional import flatten


class Flatten(Module):
    def __call__(self, x: JaxArray) -> JaxArray:  # pylint:disable=arguments-differ
        return flatten(x)
