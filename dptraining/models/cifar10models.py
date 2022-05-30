import objax

from functools import partial


class Cifar10ConvNet(objax.nn.Sequential):
    def __init__(self, nclass=10):
        ops = (
            objax.nn.Conv2D(3, 32, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),
            objax.nn.Conv2D(32, 64, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),
            objax.nn.Conv2D(64, 64, k=3, strides=1, padding=1),
            objax.functional.relu,
            partial(objax.functional.average_pool_2d, size=2, strides=2),
            objax.nn.Conv2D(64, 128, k=3, strides=1, padding=1),
            objax.functional.relu,
            lambda x: x.mean((2, 3)),
            objax.functional.flatten,
            objax.nn.Linear(128, nclass, use_bias=True),
        )
        super().__init__(ops)
