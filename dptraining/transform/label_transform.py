import abc
from dptraining.transform.transform import Transform


class LabelTransform(Transform, metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "transform_label")
            and callable(subclass.transform_label)
            or NotImplemented
        )

    @abc.abstractmethod
    def transform_label(self, label):
        raise NotImplementedError
