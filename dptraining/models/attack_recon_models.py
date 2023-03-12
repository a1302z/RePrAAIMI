from objax import Module, nn, functional


# Balle et al propose this as (x, 1000, 1000, y) in their sample code
class MLP(Module):
    def __init__(
        self,
        inp_size: int,
        outp_size: int,
        sizes: list[int],
        activation_fn=functional.relu,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        layers: list[Module] = [nn.Linear(inp_size, sizes[0]), activation_fn]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation_fn)
        layers.append(nn.Linear(sizes[-1], outp_size))
        if activate_final:
            layers.append(activation_fn)
        self.mlp = nn.Sequential(layers)

    def __call__(self, x, **kwargs):
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x, **kwargs)
