import numpy as np

from dptraining.transform.unary_image_and_label_transform import (
    UnaryImageAndLabelTransform,
)


class TransposeNumpyBatchToCHW(UnaryImageAndLabelTransform):
    def unary_transform(self, data: np.array):
        return data.transpose(0, 3, 1, 2)


class TransposeNumpyImgToCHW(UnaryImageAndLabelTransform):
    def unary_transform(self, data: np.array):
        return data.transpose(2, 0, 1)
