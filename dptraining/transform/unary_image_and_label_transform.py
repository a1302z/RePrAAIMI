import abc

from dptraining.transform.image_transform import ImageTransform
from dptraining.transform.label_transform import LabelTransform
from dptraining.transform.image_label_transform import (
    ImageLabelTransform,
)


class UnaryImageAndLabelTransform(
    ImageTransform, LabelTransform, ImageLabelTransform, metaclass=abc.ABCMeta
):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "unary_transform")
            and callable(subclass.unary_transform)
            or NotImplemented
        )

    def transform_image(self, image):
        return self.unary_transform(image)

    def transform_label(self, label):
        return self.unary_transform(label)

    def transform_image_label(self, image, label):
        return self.unary_transform(image), self.unary_transform(label)

    @abc.abstractmethod
    def unary_transform(self, image_or_label):
        raise NotImplementedError
