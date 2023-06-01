from jax import numpy as jnp
from math import prod
import objax
from omegaconf import DictConfig, OmegaConf
from random import choice, seed
from torchvision import transforms
from typing import TypedDict, Callable, Literal, Optional, Any, Union

from dptraining.transform.image_transform import ImageTransform
from dptraining.transform.label_transform import LabelTransform
from dptraining.transform.image_label_transform import ImageLabelTransform

from dptraining.transform.transforms.complex import (
    MakeComplexOnlyReal,
    MakeComplexRealAndImaginary,
)
from dptraining.transform.transforms.convert import PILToNumpy, PILToJAXNumpy
from dptraining.transform.transforms.crop import CenterCrop, RandomCrop, RandomCropJAX
from dptraining.transform.transforms.fft import FFT, JaxFFT, IFFT, JaxIFFT
from dptraining.transform.transforms.flip import (
    RandomVerticalFlipsJax,
    RandomHorizontalFlipsJax,
    RandomHorizontalFlipsJaxBatch,
    RandomVerticalFlipsJaxBatch,
    RandomZFlipsJax,
)
from dptraining.transform.transforms.noise import GaussianNoise
from dptraining.transform.transforms.normalize import (
    NormalizeJAX,
    NormalizeJAXBatch,
    NormalizeNumpyBatch,
    NormalizeNumpyImg,
)
from dptraining.transform.transforms.random_phase import (
    AddRandomPhase,
    AddRandomPhaseJAX,
)
from dptraining.transform.transforms.shift import (
    RandomImageShiftsJax,
    RandomImageShiftsJaxBatch,
)
from dptraining.transform.transform import Transform
from dptraining.transform.transforms.transpose import (
    TransposeNumpyImgToCHW,
    TransposeNumpyBatchToCHW,
)


seed(0)

torchvision_transforms = {
    mn: getattr(transforms, mn)
    for mn in dir(transforms)
    if callable(getattr(transforms, mn))
}


class ScopedTransformation(TypedDict):
    apply: Callable
    to: Literal["images", "labels", "images_and_labels"]


class ConsecutiveAugmentations(ImageTransform, ImageLabelTransform):
    def __init__(self, apply_to: Literal["images"], **kwargs) -> None:
        if apply_to not in ("images"):
            raise ValueError(
                "Consecutive augmentations are only supported for images at the moment"
            )
        self._multiplicity: int = (
            kwargs["multiplicity"] if "multiplicity" in kwargs else 1
        )
        self._complex: bool = kwargs["complex"] if "complex" in kwargs else False
        if "augmentations" not in kwargs or len(kwargs["augmentations"]) == 0:
            raise ValueError("Augmentations must be specified")
        transform_configs = kwargs["augmentations"]
        if isinstance(transform_configs, tuple) and len(transform_configs) == 1:
            transform_configs = transform_configs[0]
        if isinstance(transform_configs, list):
            if not all(isinstance(config, dict) for config in transform_configs):
                raise ValueError("Augmentation list must contain only dicts")
            self._transformations = [
                TransformPipeline.from_dict_list(config) for config in transform_configs
            ]
        elif isinstance(transform_configs, dict):
            self._transformations = [
                TransformPipeline.from_dict_list(
                    {transform_name: transform_config},
                    apply_to_override_deep=self._apply_to,
                )
                for transform_name, transform_config in transform_configs.items()
            ]
        else:
            raise ValueError(
                "augmentations need to be specified as a dict or list of dicts"
            )

    def transform_image(self, image):  # pylint:disable=too-many-branches
        if isinstance(
            self._transformations, list
        ):  # pylint:disable=too-many-nested-blocks
            augm = []
            for _ in range(self._multiplicity):
                aug_data = jnp.copy(image)
                for transformation in self._transformations:
                    if len(aug_data.shape) != len(image.shape):
                        augs = []
                        for i in range(aug_data.shape[0]):
                            if self._complex:
                                augs.append(
                                    transformation.transform_image(aug_data[i].real)
                                    + 1j
                                    * transformation.transform_image(aug_data[i].imag)
                                )
                            else:
                                augs.append(transformation.transform_image(aug_data[i]))
                        shape_equality = [
                            len(aug_shape.shape) == len(image.shape)
                            for aug_shape in augs
                        ]
                        if all(shape_equality):
                            aug_data = jnp.stack(augs, axis=0)
                        else:
                            aug_data = jnp.concatenate(augs, axis=0)
                    else:
                        if self._complex:
                            aug_data = transformation.transform_image(
                                aug_data.real
                            ) + 1j * transformation.transform_image(aug_data.imag)
                        else:
                            aug_data = transformation.transform_image(aug_data)
                augm.append(aug_data)
            if self._multiplicity == 1:
                return aug_data
            elif any((len(a.shape) > len(image.shape) for a in augm)):
                augm = [  # pylint:disable=use-a-generator
                    a[jnp.newaxis, ...] if len(a.shape) == len(image.shape) else a
                    for a in augm
                ]
                augm = jnp.concatenate(augm, axis=0)
            else:
                augm = jnp.stack(augm, axis=0)
        else:
            if self._complex:
                augm = [
                    self._transformations.transform_image(image.real)
                    + 1j * self._transformations.transform_image(image.imag)
                    for _ in range(self._multiplicity)
                ]
            else:
                augm = [
                    self._transformations.transform_image(image)
                    for _ in range(self._multiplicity)
                ]
            if self._multiplicity == 1:
                return augm[0]
            else:
                augm = jnp.stack(augm, axis=0)

        if len(augm.shape) > len(image.shape) + 1:
            raise RuntimeError("augmentation multiplicity extended too far")
        return augm

    def transform_image_label(self, image, label):  # pylint:disable=too-many-branches
        if isinstance(
            self._transformations, list
        ):  # pylint:disable=too-many-nested-blocks
            augmented_images = []
            augmented_labels = []
            for _ in range(self._multiplicity):
                augmented_image = jnp.copy(image)
                augmented_label = jnp.copy(label)
                for transformation in self._transformations:
                    if len(augmented_image.shape) != len(image.shape):
                        image_augmentations = []
                        label_augmentations = []
                        for i in range(augmented_image.shape[0]):
                            if self._complex:
                                (
                                    image_aug_real,
                                    label_aug_real,
                                ) = transformation.transform_image_label(
                                    augmented_image[i].real, augmented_label[i].real
                                )
                                (
                                    image_aug_imag,
                                    label_aug_imag,
                                ) = transformation.transform_image_label(
                                    augmented_image[i].imag, augmented_label[i].imag
                                )
                                image_augmentations.append(
                                    image_aug_real + 1j * image_aug_imag
                                )
                                label_augmentations.append(
                                    label_aug_real + 1j * label_aug_imag
                                )
                            else:
                                (
                                    image_aug,
                                    label_aug,
                                ) = transformation.transform_image_label(
                                    augmented_image[i], augmented_label[i]
                                )
                                image_augmentations.append(image_aug)
                                label_augmentations.append(label_aug)
                        image_augmentation_shapes_equal_image_shape = all(
                            len(image_augmentation_shape.shape) == len(image.shape)
                            for image_augmentation_shape in image_augmentations
                        )
                        if image_augmentation_shapes_equal_image_shape:
                            augmented_image = jnp.stack(image_augmentations, axis=0)
                            augmented_label = jnp.stack(label_augmentations, axis=0)
                        else:
                            augmented_image = jnp.concatenate(
                                image_augmentations, axis=0
                            )
                            augmented_label = jnp.concatenate(
                                label_augmentations, axis=0
                            )
                    else:
                        if self._complex:
                            (
                                image_aug_real,
                                label_aug_real,
                            ) = transformation.transform_image_label(
                                augmented_image.real, augmented_label.real
                            )
                            (
                                image_aug_imag,
                                label_aug_imag,
                            ) = transformation.transform_image_label(
                                augmented_image.imag, augmented_label.imag
                            )
                            augmented_image = image_aug_real + 1j * image_aug_imag
                            augmented_label = label_aug_real + 1j * label_aug_imag
                        else:
                            (
                                augmented_image,
                                augmented_label,
                            ) = transformation.transform_image_label(
                                augmented_image, augmented_label
                            )
                augmented_images.append(augmented_image)
                augmented_labels.append(augmented_label)
            if self._multiplicity == 1:
                return augmented_image, augmented_label
            elif any((len(a.shape) > len(image.shape) for a in augmented_images)):
                augmented_images = [  # pylint:disable=use-a-generator
                    augmented_image[jnp.newaxis, ...]
                    if len(augmented_image.shape) == len(image.shape)
                    else augmented_image
                    for augmented_image in augmented_images
                ]
                augmented_images = jnp.concatenate(augmented_images, axis=0)
                augmented_labels = [  # pylint:disable=use-a-generator
                    augmented_label[jnp.newaxis, ...]
                    if len(augmented_label.shape) == len(label.shape)
                    else augmented_label
                    for augmented_label in augmented_labels
                ]
                augmented_labels = jnp.concatenate(augmented_labels, axis=0)
            else:
                augmented_images = jnp.stack(augmented_images, axis=0)
                augmented_labels = jnp.stack(augmented_labels, axis=0)
        else:
            if self._complex:
                augmented_image_labels_real = [
                    self._transformations.transform_image_label(image.real, label.real)
                    for _ in range(self._multiplicity)
                ]
                augmented_image_labels_imag = [
                    self._transformations.transform_image_label(image.imag, label.imag)
                    for _ in range(self._multiplicity)
                ]
                augmented_images = [
                    augmented_image_label_real[0] + 1j * augmented_image_label_imag[0]
                    for augmented_image_label_real, augmented_image_label_imag in zip(
                        augmented_image_labels_real, augmented_image_labels_imag
                    )
                ]
                augmented_labels = [
                    augmented_image_label_real[1] + 1j * augmented_image_label_imag[1]
                    for augmented_image_label_real, augmented_image_label_imag in zip(
                        augmented_image_labels_real, augmented_image_labels_imag
                    )
                ]
            else:
                augmented_image_labels = [
                    self._transformations.transform_image_label(image, label)
                    for _ in range(self._multiplicity)
                ]
                augmented_images = [
                    augmented_image_label[0]
                    for augmented_image_label in augmented_image_labels
                ]
                augmented_labels = [
                    augmented_image_label[1]
                    for augmented_image_label in augmented_image_labels
                ]
            if self._multiplicity == 1:
                return augmented_images[0], augmented_labels[0]
            else:
                augmented_images = jnp.stack(augmented_images, axis=0)
                augmented_labels = jnp.stack(augmented_labels, axis=0)

        if (
            len(augmented_images.shape) > len(image.shape) + 1
            or len(augmented_labels.shape) > len(label.shape) + 1
        ):
            raise RuntimeError("augmentation multiplicity extended too far")
        return augmented_images, augmented_labels

    def get_total_multiplicity(self):
        if isinstance(self._transformations, list):
            return self._multiplicity * prod(
                [child.get_n_augmentations() for child in self._transformations]
            )
        else:
            return self._multiplicity * self._transformations.get_n_augmentations()


class RandomTransform(ImageTransform, ImageLabelTransform):
    def __init__(
        self,
        apply_to: Literal["images", "labels", "images_and_labels"],
        augmentations: list[dict[str, Any]],
    ) -> None:
        if not isinstance(augmentations, dict) or len(augmentations.keys()) == 0:
            raise ValueError(
                "Augmentations for random augmentations have to be a non-empty list of dicts"
            )
        self._apply_to: Literal["images", "labels", "images_and_labels"] = apply_to
        self._transformation_choices: list[TransformPipeline] = []
        for transform_name, transform_kwargs in augmentations.items():
            transform_kwargs["apply_to"] = apply_to
            transformation = TransformPipeline.from_dict_list(
                {transform_name: transform_kwargs}
            )
            self._transformation_choices.append(transformation)
        self._transformation_choices: list[TransformPipeline] = [
            TransformPipeline.from_dict_list(
                {transform_name: init_kwargs},
            )
            for transform_name, init_kwargs in augmentations.items()
        ]

    def transform_image(self, image):
        if self._apply_to != "images":
            raise ValueError(
                "This random transform can not be used to transform images"
            )
        return self._get_random_transformation().transform_image(image)

    def transform_image_label(self, image, label):
        if self._apply_to != "images_and_labels":
            raise ValueError(
                "This random transform can not be used to transform images with labels"
            )
        return self._get_random_transformation().transform_image_label(image, label)

    def get_n_augmentations_list_for_all_choices(self):
        return [tf.get_n_augmentations() for tf in self._transformation_choices]

    def _get_random_transformation(self):
        return choice(self._transformation_choices)


class TransformPipeline:
    _mapping: dict[str, Union[type, Callable]] = {
        "random_horizontal_flips": RandomHorizontalFlipsJax,
        "random_vertical_flips": RandomVerticalFlipsJax,
        "random_z_flips": RandomZFlipsJax,
        "random_img_shift": RandomImageShiftsJax,
        "random_horizontal_flips_batch": RandomHorizontalFlipsJaxBatch,
        "random_vertical_flips_batch": RandomVerticalFlipsJaxBatch,
        "random_img_shift_batch": RandomImageShiftsJaxBatch,
        "center_crop": CenterCrop,
        "random_crop_jax": RandomCropJAX,
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
        "random_crop": RandomCrop,
        "random_crop_jax": RandomCropJAX,
        "fft": FFT,
        "ifft": IFFT,
        "jaxfft": JaxFFT,
        "jaxifft": JaxIFFT,
        "gaussiannoise": GaussianNoise,
        "randomphase": AddRandomPhase,
        "randomphasejax": AddRandomPhaseJAX,
        **torchvision_transforms,
    }

    def __init__(self, transformations: list[ScopedTransformation]):
        if any(
            not callable(transformation["apply"]) for transformation in transformations
        ):
            raise ValueError("Uncallable transforms")
        for transformation in transformations:
            if "to" not in transformation:
                transformation["to"] = "images"
        self._transformations: list[ScopedTransformation] = transformations

    def __len__(self):
        return len(self._transformations)

    def get_n_augmentations(self):
        all_counts = []
        for tf in self._transformations:
            transformation_obj = tf["apply"].__self__
            transformation_type = type(transformation_obj)
            if transformation_type == ConsecutiveAugmentations:
                transformation_obj: ConsecutiveAugmentations
                all_counts.append(transformation_obj.get_total_multiplicity())
            elif transformation_type == RandomTransform:
                transformation_obj: RandomTransform
                all_counts.extend(
                    transformation_obj.get_n_augmentations_list_for_all_choices()
                )
        all_counts = list(filter((1).__ne__, all_counts))
        return sum(all_counts) if len(all_counts) > 0 else 1

    @classmethod
    def from_dict_list(  # pylint:disable=too-many-branches
        cls, transform_configs: Optional[dict[str, Any]]
    ):
        if transform_configs is None:
            return cls([])
        if not isinstance(transform_configs, (dict, DictConfig)):
            raise ValueError(
                "Transforms with args need to be defined as dict (of dicts)"
            )
        if not all(
            transform_name in TransformPipeline._mapping
            for transform_name in transform_configs.keys()
            if isinstance(transform_name, (DictConfig, dict))
        ):
            unknown_transform_names = [
                name
                for name in transform_configs.keys()
                if name not in TransformPipeline._mapping
            ]
            raise ValueError(
                f"Augmentations {unknown_transform_names} not known. "
                f"Supported augmentation names: {TransformPipeline._mapping.keys()}"
            )
        transform_fns = []
        for transform_name, transform_kwargs in transform_configs.items():
            transform_class = TransformPipeline._mapping[transform_name]
            apply_to = (
                transform_kwargs.pop("apply_to", "images")
                if isinstance(transform_kwargs, dict)
                else "images"
            )
            transform_object = TransformPipeline._instantiate_transform_class(
                transform_class, transform_kwargs, apply_to
            )
            # Transformations not implemented by us (e.g. torchvision's)
            if not issubclass(transform_class, Transform):
                if not callable(transform_object):
                    raise ValueError(f"Augmentation {transform_name} is not callable")
                transform_fns.append({"apply": transform_object, "to": apply_to})
                continue
            transform_object: Transform
            match apply_to:
                case "images":
                    if not issubclass(transform_class, ImageTransform):
                        raise ValueError(
                            f"Ignoring augmentation {transform_name}, as it can't be used to transform images"
                        )
                    transform_object: ImageTransform
                    transform_fns.append(
                        {
                            "apply": transform_object.transform_image,
                            "to": apply_to,
                        }
                    )
                case "labels":
                    if not issubclass(transform_class, LabelTransform):
                        raise ValueError(
                            f"Ignoring augmentation {transform_name}, as it can't be used to transform labels"
                        )
                    transform_object: LabelTransform
                    transform_fns.append(
                        {
                            "apply": transform_object.transform_label,
                            "to": apply_to,
                        }
                    )
                case "images_and_labels":
                    if apply_to == "images_and_labels" and not issubclass(
                        transform_class, ImageLabelTransform
                    ):
                        raise ValueError(
                            f"Ignoring augmentation {transform_name}, as it can't be used to transform images together with "
                            f"their labels"
                        )
                    transform_object: ImageLabelTransform
                    transform_fns.append(
                        {
                            "apply": transform_object.transform_image_label,
                            "to": apply_to,
                        }
                    )
        return cls(transform_fns)

    @staticmethod
    def _instantiate_transform_class(
        transform_class: type,
        init_kwargs: Optional[Union[dict, Any]],
        apply_to: Literal["images", "labels", "images_and_labels"] = "images",
    ):
        if transform_class in (ConsecutiveAugmentations, RandomTransform):
            if not isinstance(init_kwargs, dict):
                raise ValueError(
                    f"Consecutive and random augmentations have to be configured with a dict"
                )
            init_kwargs = init_kwargs | {"apply_to": apply_to}
        if init_kwargs is None:
            return transform_class()
        if isinstance(init_kwargs, dict):
            return transform_class(**init_kwargs)
        else:
            return transform_class(init_kwargs)

    def transform_image(self, image):
        for transformation in self._transformations:
            apply_transformation = transformation["apply"]
            if transformation["to"] == "images":
                image = apply_transformation(image)
        return image

    def transform_image_label(self, image, label):
        for transformation in self._transformations:
            apply_transformation = transformation["apply"]
            match transformation["to"]:
                case "images":
                    image = apply_transformation(image)
                case "labels":
                    label = apply_transformation(label)
                case "images_and_labels":
                    image, label = apply_transformation(image, label)
        return image, label

    def create_vectorized_image_transform(self):
        if not self._transformations:
            return lambda x: x

        @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
        def apply_transform_pipeline(image):
            for transformation in self._transformations:
                if transformation["to"] != "images":
                    raise ValueError(
                        "Can't use vectorized image transform for transformations operating on labels"
                    )
                apply_transformation = transformation["apply"]
                image = apply_transformation(image)
            return image

        return objax.Vectorize(apply_transform_pipeline)

    def create_vectorized_image_label_transform(self):
        if not self._transformations:
            return lambda image, label: (image, label)

        @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
        def apply_transform_pipeline(image, label):
            for transformation in self._transformations:
                apply_transformation = transformation["apply"]
                match transformation["to"]:
                    case "images":
                        image = apply_transformation(image)
                    case "labels":
                        label = apply_transformation(label)
                    case "images_and_labels":
                        image, label = apply_transformation(image, label)
            return image, label

        return objax.Vectorize(apply_transform_pipeline, batch_axis=(0, 0))

    def get_transformations(self):
        return self._transformations


def make_augment_transformation(augmentations_config: Optional[dict[str, Any]]):
    config_dict = (
        OmegaConf.to_container(augmentations_config) if augmentations_config else {}
    )
    return TransformPipeline.from_dict_list(config_dict)
