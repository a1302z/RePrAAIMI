import numpy as np
from objax.zoo import resnet_v2, vgg

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from dptraining.models.complex.converter import ComplexModelConverter


def test_conversion_resnet():
    m = resnet_v2.ResNet18(3, 2)
    converter = ComplexModelConverter()
    m2 = converter(m)
    print(m2)
    data = np.random.randn(10, 3, 224, 224) + 1j * np.random.randn(10, 3, 224, 224)
    m2(data, training=False)


# the vgg implementation requires a downloaded model so we omit this
# def test_conversion_vgg():
#     m = vgg.VGG19()
#     converter = ComplexModelConverter()
#     m2 = converter(m)
#     print(m2)
#     data = np.random.randn(10, 3, 224, 224) + 1j * np.random.randn(10, 3, 224, 224)
#     m2(data, training=False)
