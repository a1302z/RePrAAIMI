import abc
from dptraining.transform.transform import Transform


class ImageLabelTransform(Transform, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "transform_image_label")
            and callable(subclass.transform_image_label)
            or NotImplemented
        )

    @abc.abstractmethod
    def transform_image_label(self, image, label):
        raise NotImplementedError
