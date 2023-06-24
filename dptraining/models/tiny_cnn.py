# copied from objax MNIST example
import objax

def simple_net_block(nin, nout):
    return objax.nn.Sequential([
        objax.nn.Conv2D(nin, nout, k=3),
        objax.nn.GroupNorm2D(nout, groups=nout//2),
        objax.functional.leaky_relu,
        objax.functional.max_pool_2d,
        objax.nn.Conv2D(nout, nout, k=3),
        objax.nn.GroupNorm2D(nout, groups=nout//2),
        objax.functional.leaky_relu,
    ])


class SimpleNet(objax.Module):
    def __init__(self, num_classes, in_channels, n=64):
        self.pre_conv = objax.nn.Sequential([objax.nn.Conv2D(in_channels, n, k=3), objax.functional.leaky_relu])
        self.block1 = simple_net_block(1 * n, 2 * n)
        self.block2 = simple_net_block(2 * n, 4 * n)
        self.post_conv = objax.nn.Conv2D(4 * n, num_classes, k=3)

    def __call__(self, x, training=False):  # x = (batch, colors, height, width)
        y = self.pre_conv(x)
        y = self.block1(y)
        y = self.block2(y)
        logits = self.post_conv(y).mean((2, 3))  # logits = (batch, nclass)
        return logits