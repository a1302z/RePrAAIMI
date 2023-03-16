from objax import Module, nn, functional
from objax.typing import JaxArray
from jax import numpy as jn
from typing import Optional, Callable, Union


# Balle et al propose this as (x, 1000, 1000, y) in their sample code
class MLP(Module):
    def __init__(
        self,
        inp_size: int,
        outp_size: int,
        sizes: list[int],
        activation_fn=functional.relu,
        normalization_layer: Optional[Callable] = None,
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

    def __str__(self) -> str:
        return str(self.mlp)


class MIAComparisonModel(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model: Module = model

    def __call__(self, x: Union[JaxArray, tuple[JaxArray, JaxArray]], **kwargs):
        if isinstance(x, tuple) and len(x) == 2:
            pred_a = self.model(x[0], **kwargs)
            pred_b = self.model(x[1], **kwargs)
        elif x.shape[1] == 2:
            pred_a = self.model(x[:, 0], **kwargs)
            pred_b = self.model(x[:, 1], **kwargs)
        else:
            raise ValueError(
                f"Input format somehow not aligned with expected two inputs"
            )

        return functional.sigmoid(pred_b - pred_a)


class MIAWeightImageModel(Module):
    def __init__(
        self, weight_model: Module, img_model: Module, compare_model: Module
    ) -> None:
        super().__init__()
        self.weight_model: Module = weight_model
        self.img_model: Module = img_model
        self.compare_model: Module = compare_model

    def __call__(
        self, weights_and_samples: tuple[JaxArray, JaxArray, JaxArray], **kwargs
    ):
        w_pred = self.weight_model(weights_and_samples[0], **kwargs)
        img_pred_a = self.img_model(weights_and_samples[1], **kwargs)
        img_pred_b = self.img_model(weights_and_samples[2], **kwargs)
        pred_a = self.compare_model(jn.concatenate([w_pred, img_pred_a], axis=1))
        pred_b = self.compare_model(jn.concatenate([w_pred, img_pred_b], axis=1))
        return functional.sigmoid(pred_b - pred_a)
