from typing import Any, Callable, Optional

import objax
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

torchvision_transforms = {
    mn: getattr(transforms, mn)
    for mn in dir(transforms)
    if callable(getattr(transforms, mn))
}


class ComplexAugmentations:
    def __init__(self, **augmentations: dict) -> None:
        self.aug = Transformation.from_dict_list(augmentations)

    def __call__(self, data, labels=None):
        return self.aug(data.real) + 1j * self.aug(data.imag)


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
        "complex_augmentations": ComplexAugmentations,
        **torchvision_transforms,
    }

    def __init__(self, transformations: list[Callable]):
        self._transformations: list[Callable]
        if all(isinstance(var, Callable) for var in transformations):
            self._transformations = transformations
        else:
            raise ValueError("Uncallable transforms")

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
    def from_dict_list(cls, transformations: Optional[dict[dict[str, Any]]]):
        if transformations is None:
            return cls([])
        if not isinstance(transformations, (dict, DictConfig)):
            raise ValueError(
                "Transforms with args need to be defined as dict (of dicts)"
            )
        if not all(
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
        return cls(
            [
                Transformation._mapping[aug](**kwargs)
                if kwargs is not None
                else Transformation._mapping[aug]()
                for aug, kwargs in transformations.items()
            ]
        )

    def __call__(self, x):  # pylint:disable=invalid-name
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
        @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
        def augment_op(x):  # pylint:disable=invalid-name
            for transf in self._transformations:
                x = transf(x)
            return x

        return objax.Vectorize(augment_op)
