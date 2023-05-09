import abc
from dptraining.transform.transform import Transform


class ImageTransform(Transform, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "transform_image")
            and callable(subclass.transform_image)
            or NotImplemented
        )

    @abc.abstractmethod
    def transform_image(self, image):
        raise NotImplementedError
