from typing import Callable
import objax, jax, jax.numpy as jn


class Augmentation:
    _mapping: dict[str, Callable]

    def __init__(self, augmentations: list[Callable]):
        self._augmentations: list[Callable]
        if all([isinstance(var, Callable) for var in augmentations]):
            self._augmentations = augmentations
        else:
            raise ValueError("Uncallable augmentations")

    @classmethod
    def from_string_list(cls, augmentations: list[str]):
        if not all([isinstance(var, str) for var in augmentations]):
            raise ValueError("Augmentations need to be defined as strings")
        if not all([var in Augmentation._mapping for var in augmentations]):
            raise ValueError(
                f"{[var for var in augmentations if var not in Augmentation._mapping]} not known."
                f" Supported ops are: {Augmentation._mapping.keys()}"
            )
        return cls([Augmentation._mapping[aug] for aug in augmentations])

    def append_augmentation_method(self, new_augmentation: Callable):
        if not isinstance(new_augmentation, Callable):
            raise ValueError("Augmentations need to be defined as strings")
        self._augmentations.append(new_augmentation)

    def append_augmentation_str(self, new_augmentation: str):
        if not isinstance(new_augmentation, str):
            raise ValueError("Augmentations need to be defined as strings")
        if not new_augmentation in Augmentation._mapping.keys():
            raise ValueError(
                f"{new_augmentation} not known. Supported ops are: {Augmentation._mapping.keys()}"
            )
        self._augmentations.append(new_augmentation)

    @staticmethod
    def random_vertical_flips(x, flip_prob=0.5):
        return jax.lax.cond(
            objax.random.uniform(()) > flip_prob,
            lambda t: t,
            lambda t: t[:, :, ::-1],
            operand=x,
        )

    @staticmethod
    def random_horizontal_flips(x, flip_prob=0.5):
        return jax.lax.cond(
            objax.random.uniform(()) > flip_prob,
            lambda t: t,
            lambda t: t[:, ::-1, :],
            operand=x,
        )

    @staticmethod
    def random_img_shift(x, img_size=(3, 32, 32), max_shift=4):
        x_pad = jn.pad(
            x, [[0, 0], [max_shift, max_shift], [max_shift, max_shift]], "reflect"
        )
        offset = objax.random.randint((2,), 0, max_shift)
        return jax.lax.dynamic_slice(x_pad, (0, offset[0], offset[1]), img_size)

    def create_augmentation_op(self):
        @objax.Function.with_vars(objax.random.DEFAULT_GENERATOR.vars())
        def augment_op(x):
            for aug in self._augmentations:
                x = aug(x)
            return x

        return objax.Vectorize(augment_op)


Augmentation._mapping = {
    "random_horizontal_flips": Augmentation.random_horizontal_flips,
    "random_vertical_flips": Augmentation.random_vertical_flips,
    "random_img_shift": Augmentation.random_img_shift,
}
