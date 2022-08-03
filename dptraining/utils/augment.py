from typing import Any, Callable, Optional

import objax
from jax import numpy as jnp
from dptraining.utils.transform import (
    MakeComplexOnlyReal,
    MakeComplexRealAndImaginary,
    NormalizeJAX,
    NormalizeJAXBatch,
    NormalizeNumpyBatch,
    NormalizeNumpyImg,
    PILToJAXNumpy,
    PILToNumpy,
    RandomHorizontalFlipsJax,
    RandomHorizontalFlipsJaxBatch,
    RandomImageShiftsJax,
    RandomImageShiftsJaxBatch,
    RandomVerticalFlipsJax,
    RandomVerticalFlipsJaxBatch,
    TransposeNumpyBatchToCHW,
    TransposeNumpyImgToCHW,
)
from omegaconf import DictConfig
from torchvision import transforms

from random import choice, seed

seed(0)

torchvision_transforms = {
    mn: getattr(transforms, mn)
    for mn in dir(transforms)
    if callable(getattr(transforms, mn))
}


class ConsecutiveAugmentations:
    def __init__(self, *args, **kwargs) -> None:
        assert (len(args) > 0) ^ (len(kwargs) > 0), "Either args or kwargs, not both"
        if isinstance(args, tuple) and len(args) == 1:
            args = args[0]
        self.complex = False
        if len(args) > 0:
            stack_augmentations = False
            if "stack_augmentations" in args:
                stack_augmentations = True
                args.remove("stack_augmentations")
            if "complex" in args:
                self.complex = True
                args.remove("complex")
            assert all((isinstance(arg, dict) for arg in args))
            self.aug = Transformation(
                [Transformation.from_dict_list(arg) for arg in args],
                stack_augmentations=stack_augmentations,
            )
        if len(kwargs) > 0:
            if "complex" in kwargs:
                self.complex = kwargs["complex"]
                del kwargs["complex"]
            self.aug = Transformation.from_dict_list(kwargs)

    def __call__(self, data, labels=None):
        return (
            self.aug(data.real) + 1j * self.aug(data.imag)
            if self.complex
            else self.aug(data)
        )


class RandomTransform:
    def __init__(self, *args, **kwargs) -> None:
        if isinstance(args, tuple) and len(args) == 1:
            args = args[0]
        assert (len(args) > 0) ^ (len(kwargs) > 0), "Either args or kwargs, not both"
        if len(args) > 0:
            self.aug = [Transformation.from_dict_list(arg) for arg in args]
        elif len(kwargs) > 0:
            self.aug = [Transformation.from_dict_list(kwarg) for kwarg in kwargs]
        else:
            raise ValueError("There should be a transform")

    def __call__(self, data) -> Any:
        transf = choice(self.aug)
        return transf(data)


class Transformation:
    _mapping: dict[str, Callable] = {
        "random_horizontal_flips": RandomHorizontalFlipsJax,
        "random_vertical_flips": RandomVerticalFlipsJax,
        "random_img_shift": RandomImageShiftsJax,
        "random_horizontal_flips_batch": RandomHorizontalFlipsJaxBatch,
        "random_vertical_flips_batch": RandomVerticalFlipsJaxBatch,
        "random_img_shift_batch": RandomImageShiftsJaxBatch,
        "pil_to_numpy": PILToNumpy,
        "normalize_np_img": NormalizeNumpyImg,
        "normalize_np_batch": NormalizeNumpyBatch,
        "pil_to_jax": PILToJAXNumpy,
        "normalize_jax": NormalizeJAX,
        "normalize_jax_batch": NormalizeJAXBatch,
        "make_complex_real": MakeComplexOnlyReal,
        "make_complex_both": MakeComplexRealAndImaginary,
        "numpy_batch_to_chw": TransposeNumpyBatchToCHW,
        "numpy_img_to_chw": TransposeNumpyImgToCHW,
        "consecutive_augmentations": ConsecutiveAugmentations,
        "random_augmentations": RandomTransform,
        **torchvision_transforms,
    }

    def __init__(
        self, transformations: list[Callable], stack_augmentations: bool = False
    ):
        self._transformations: list[Callable]
        self._stack_augmentations: bool = stack_augmentations
        if all(isinstance(var, Callable) for var in transformations):
            self._transformations = transformations
        else:
            raise ValueError("Uncallable transforms")

    def __len__(self):
        return len(self._transformations)

    def get_n_augmentations(self):
        if self._stack_augmentations:
            return len(self._transformations)
        else:
            return max(
                [
                    t.get_n_augmentations()
                    for t in self._transformations
                    if isinstance(t, Transformation)
                ]
                + [
                    t.aug.get_n_augmentations()
                    for t in self._transformations
                    if isinstance(t, ConsecutiveAugmentations)
                ]
                + [1]
            )

    @classmethod
    def from_string_list(cls, transformations: list[str]):
        if not all(isinstance(var, str) for var in transformations):
            raise ValueError("Transforms need to be defined consistently as strings")
        if not all(
            var in Transformation._mapping
            for var in transformations
            if isinstance(var, str)
        ):
            u_transf = [
                var for var in transformations if var not in Transformation._mapping
            ]
            raise ValueError(
                f"{u_transf} not known."
                f" Supported ops are: {Transformation._mapping.keys()}"
            )
        return cls([Transformation._mapping[aug]() for aug in transformations])

    @classmethod
    def from_dict_list(  # pylint:disable=too-many-branches
        cls, transformations: Optional[dict[dict[str, Any]]]
    ):
        if transformations is None:
            return cls([])
        stack_augmentations = False
        if "stack_augmentations" in transformations:
            if isinstance(transformations, dict):
                stack_augmentations = transformations["stack_augmentations"]
                del transformations["stack_augmentations"]
        if not isinstance(transformations, (dict, DictConfig)):  # or list_of_dicts):
            raise ValueError(
                "Transforms with args need to be defined as dict (of dicts)"
            )
        if not all(  # not list_of_dicts and
            var in Transformation._mapping
            for var in transformations.keys()
            if isinstance(var, (DictConfig, dict))
        ):
            u_transf = [
                var
                for var in transformations.keys()
                if var not in Transformation._mapping
            ]
            raise ValueError(
                f"{u_transf} not known. "
                f"Supported ops are: {Transformation._mapping.keys()}"
            )
        tfs = []
        for aug, kwargs in transformations.items():
            if kwargs is not None:
                if isinstance(kwargs, dict):
                    transform = Transformation._mapping[aug](**kwargs)
                else:
                    transform = Transformation._mapping[aug](kwargs)
            else:
                transform = Transformation._mapping[aug]()
            tfs.append(transform)
        return cls(
            tfs,
            stack_augmentations,
        )

    def __call__(self, x):  # pylint:disable=invalid-name
        if self._stack_augmentations:
            x = jnp.stack([t(x) for t in self._transformations], axis=0)
        else:
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
                f"{new_transform} not known. "
                f"Supported ops are: {Transformation._mapping.keys()}"
            )
        self._transformations.append(new_transform)

    def create_vectorized_transform(self):
        if self._stack_augmentations:

            @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
            def augment_op(x):  # pylint:disable=invalid-name
                x = jnp.stack([t(x) for t in self._transformations], axis=0)
                return x

        else:

            @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
            def augment_op(x):  # pylint:disable=invalid-name
                for transf in self._transformations:
                    x = transf(x)
                return x

        return objax.Vectorize(augment_op)
