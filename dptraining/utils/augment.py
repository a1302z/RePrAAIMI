from math import prod
from typing import Any, Callable

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
    RandomZFlipsJax,
    RandomImageShiftsJax,
    RandomImageShiftsJaxBatch,
    RandomVerticalFlipsJax,
    RandomVerticalFlipsJaxBatch,
    CenterCrop,
    TransposeNumpyBatchToCHW,
    TransposeNumpyImgToCHW,
    FFT,
    IFFT,
    JaxFFT,
    JaxIFFT,
    GaussianNoise,
    AddRandomPhase,
    AddRandomPhaseJAX,
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
        if len(args) > 0:
            if isinstance(args, tuple) and len(args) == 1:
                args = args[0]
            assert all((isinstance(elmnt, dict) for elmnt in args))
            multi_arg = [a for a in args if "multiplicity" in a]
            assert len(multi_arg) <= 1, "Multiplicity specified multiple times"
            if len(multi_arg) == 0:
                self.multiplicity = 1
            else:
                self.multiplicity = multi_arg[0]["multiplicity"]
                args.remove(multi_arg[0])
            complex_arg = [a for a in args if "complex" in a]
            assert len(complex_arg) <= 1, "Complex specified multiple times"
            if len(complex_arg) == 0:
                self.complex = False
            else:
                self.complex = complex_arg[0]["complex"]
                args.remove(complex_arg[0])
            self.aug = [Transformation.from_dict_list(tf_dict) for tf_dict in args]

        elif len(kwargs) > 0:
            if "multiplicity" in kwargs:
                self.multiplicity = kwargs["multiplicity"]
                del kwargs["multiplicity"]
            else:
                self.multiplicity = 1
            if "complex" in kwargs:
                self.complex = kwargs["complex"]
                del kwargs["complex"]
            else:
                self.complex = False
            self.aug = Transformation.from_dict_list(kwargs)
        else:
            raise ValueError("Either args or kwargs necessary")

    def __call__(self, data):  # pylint:disable=too-many-branches
        if isinstance(self.aug, list):  # pylint:disable=too-many-nested-blocks
            augm = []
            for _ in range(self.multiplicity):
                aug_data = jnp.copy(data)
                for sub_aug in self.aug:
                    if len(aug_data.shape) != len(data.shape):
                        augs = []
                        for i in range(aug_data.shape[0]):
                            if self.complex:
                                augs.append(
                                    sub_aug(aug_data[i].real)
                                    + 1j * sub_aug(aug_data[i].imag)
                                )
                            else:
                                augs.append(sub_aug(aug_data[i]))
                        shape_equality = [
                            len(aug_shape.shape) == len(data.shape)
                            for aug_shape in augs
                        ]
                        if all(shape_equality):
                            aug_data = jnp.stack(augs, axis=0)
                        else:
                            aug_data = jnp.concatenate(augs, axis=0)
                    else:
                        if self.complex:
                            aug_data = sub_aug(aug_data.real) + 1j * sub_aug(
                                aug_data.imag
                            )
                        else:
                            aug_data = sub_aug(aug_data)
                augm.append(aug_data)
            if self.multiplicity == 1:
                return aug_data
            elif any((len(a.shape) > len(data.shape) for a in augm)):
                augm = [  # pylint:disable=use-a-generator
                    a[jnp.newaxis, ...] if len(a.shape) == len(data.shape) else a
                    for a in augm
                ]
                augm = jnp.concatenate(augm, axis=0)
            else:
                augm = jnp.stack(augm, axis=0)
        else:
            if self.complex:
                augm = [
                    self.aug(data.real) + 1j * self.aug(data.imag)
                    for _ in range(self.multiplicity)
                ]
            else:
                augm = [self.aug(data) for _ in range(self.multiplicity)]
            if self.multiplicity == 1:
                return augm[0]
            else:
                augm = jnp.stack(augm, axis=0)

        if len(augm.shape) > len(data.shape) + 1:
            raise RuntimeError("augmentation multiplicity extended too far")
        return augm

    def get_total_multiplicity(self):
        if isinstance(self.aug, list):
            return self.multiplicity * prod(
                [child.get_n_augmentations() for child in self.aug]
            )
        else:
            return self.multiplicity * self.aug.get_n_augmentations()


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
        "random_z_flips": RandomZFlipsJax,
        "random_img_shift": RandomImageShiftsJax,
        "random_horizontal_flips_batch": RandomHorizontalFlipsJaxBatch,
        "random_vertical_flips_batch": RandomVerticalFlipsJaxBatch,
        "random_img_shift_batch": RandomImageShiftsJaxBatch,
        "center_crop": CenterCrop,
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
        "fft": FFT,
        "ifft": IFFT,
        "jaxfft": JaxFFT,
        "jaxifft": JaxIFFT,
        "gaussiannoise": GaussianNoise,
        "randomphase": AddRandomPhase,
        "randomphasejax": AddRandomPhaseJAX,
        **torchvision_transforms,
    }

    def __init__(
        self,
        transformations: list[Callable],
    ):
        self._transformations: list[Callable]
        if all(isinstance(var, Callable) for var in transformations):
            self._transformations = transformations
        else:
            raise ValueError("Uncallable transforms")

    def __len__(self):
        return len(self._transformations)

    def get_n_augmentations(self):
        multi_tfs = [
            tf.get_total_multiplicity()
            for tf in self._transformations
            if isinstance(tf, ConsecutiveAugmentations)
        ]
        random_tfs = [
            a.get_n_augmentations()
            for tf in self._transformations
            if isinstance(tf, RandomTransform)
            for a in tf.aug
        ]
        complete_list = multi_tfs + random_tfs
        complete_list = list(filter((1).__ne__, complete_list))
        return sum(complete_list) if len(complete_list) > 0 else 1

    # @classmethod
    # def from_string_list(cls, transformations: list[str]):
    #     if not all(isinstance(var, str) for var in transformations):
    #         raise ValueError("Transforms need to be defined consistently as strings")
    #     if not all(
    #         var in Transformation._mapping
    #         for var in transformations
    #         if isinstance(var, str)
    #     ):
    #         u_transf = [
    #             var for var in transformations if var not in Transformation._mapping
    #         ]
    #         raise ValueError(
    #             f"{u_transf} not known."
    #             f" Supported ops are: {Transformation._mapping.keys()}"
    #         )
    #     return cls([Transformation._mapping[aug]() for aug in transformations])

    @classmethod
    def from_dict_list(  # pylint:disable=too-many-branches
        cls, transformations: dict[dict[str, Any]]
    ):
        if transformations is None:
            return cls([])
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
