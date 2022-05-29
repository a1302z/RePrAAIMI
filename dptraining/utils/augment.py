import objax
from typing import Callable, Dict, Any, Union
from torchvision import transforms
from omegaconf import DictConfig


from dptraining.utils.transform import (
    PILToJAXNumpy,
    NormalizeJAX,
    RandomVerticalFlipsJax,
    RandomHorizontalFlipsJax,
    RandomImageShiftsJax,
)

torchvision_transforms = {
    mn: getattr(transforms, mn)
    for mn in dir(transforms)
    if callable(getattr(transforms, mn))
}


class Transformation:
    _mapping: dict[str, Callable] = {
        "random_horizontal_flips": RandomHorizontalFlipsJax,
        "random_vertical_flips": RandomVerticalFlipsJax,
        "random_img_shift": RandomImageShiftsJax,
        "pil_to_jax": PILToJAXNumpy,
        "normalize_jax": NormalizeJAX,
        **torchvision_transforms,
    }

    def __init__(self, transformations: list[Callable]):
        self._transformations: list[Callable]
        if all([isinstance(var, Callable) for var in transformations]):
            self._transformations = transformations
        else:
            raise ValueError("Uncallable transforms")

    @classmethod
    def from_string_list(cls, transformations: list[str]):
        if not all([isinstance(var, str) for var in transformations]):
            raise ValueError("Transforms need to be defined consistently as strings")
        if not all(
            [
                var in Transformation._mapping
                for var in transformations
                if isinstance(var, str)
            ]
        ):

            raise ValueError(
                f"{[var for var in transformations if var not in Transformation._mapping]} not known."
                f" Supported ops are: {Transformation._mapping.keys()}"
            )
        return cls([Transformation._mapping[aug]() for aug in transformations])

    @classmethod
    def from_dict_list(cls, transformations: dict[dict[str, Any]]):
        if not (
            isinstance(transformations, dict) or isinstance(transformations, DictConfig)
        ):
            raise ValueError(
                "Transforms with args need to be defined as dict (of dicts)"
            )
        if not all(
            [
                var in Transformation._mapping
                for var in transformations.keys()
                if isinstance(var, DictConfig) or isinstance(var, dict)
            ]
        ):
            raise ValueError(
                f"{[var for var in transformations.keys() if var not in Transformation._mapping]} not known."
                f" Supported ops are: {Transformation._mapping.keys()}"
            )
        return cls(
            [
                Transformation._mapping[aug](**kwargs)
                if kwargs is not None
                else Transformation._mapping[aug]()
                for aug, kwargs in transformations.items()
            ]
        )

    def __call__(self, x):
        for transf in self._transformations:
            x = transf(x)
        return x

    def append_transform(self, new_transform: Callable):
        if not isinstance(new_transform, Callable):
            raise ValueError("Transforms need to be defined as strings")
        self._transformations.append(new_transform)

    def append_str_transform(self, new_transform: str):
        if not isinstance(new_transform, str):
            raise ValueError("Transforms need to be defined as strings")
        if not new_transform in Transformation._mapping.keys():
            raise ValueError(
                f"{new_transform} not known. Supported ops are: {Transformation._mapping.keys()}"
            )
        self._transformations.append(new_transform)

    def create_vectorized_transform(self):
        @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
        def augment_op(x):
            for transf in self._transformations:
                x = transf(x)
            return x

        return objax.Vectorize(augment_op)
