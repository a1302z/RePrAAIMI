from functools import partial
from typing import Callable, List
from objax import Module, ModuleList
from jax import numpy as jn


class Ensemble(Module):
    def __init__(
        self, ensemble: List[Callable], reduction=partial(jn.mean, axis=0)
    ) -> None:
        super().__init__()
        self.ensemble = ModuleList(ensemble)
        self.reduction = reduction

    def __call__(self, x, *args, **kwargs):
        return self.reduction(
            jn.stack([m(x, *args, **kwargs) for m in self.ensemble])
        )  # tree_map(lambda m: m(x), self.ensemble))
