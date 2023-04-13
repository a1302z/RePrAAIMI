import jax
import objax

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class RandomVerticalFlipsJax(UnaryImageAndLabelTransform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def unary_transform(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, ::-1],
            operand=x,
        )


class RandomHorizontalFlipsJax(UnaryImageAndLabelTransform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def unary_transform(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, ::-1, :],
            operand=x,
        )


class RandomZFlipsJax(UnaryImageAndLabelTransform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def unary_transform(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, :, ::-1],
            operand=x,
        )


class RandomVerticalFlipsJaxBatch(UnaryImageAndLabelTransform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def unary_transform(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, :, ::-1],
            operand=x,
        )


class RandomHorizontalFlipsJaxBatch(UnaryImageAndLabelTransform):
    def __init__(self, flip_prob=0.5) -> None:
        super().__init__()
        self._flip_prob = flip_prob

    def unary_transform(self, x):
        return jax.lax.cond(
            objax.random.uniform(()) > self._flip_prob,
            lambda t: t,
            lambda t: t[:, :, ::-1, :],
            operand=x,
        )
