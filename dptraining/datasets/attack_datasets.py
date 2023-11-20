import numpy as np
from typing import Union, Optional, Callable
from torch import load as torchload
from torch.utils.data import Dataset
from copy import deepcopy
from objax import VarCollection, Module, GradValues, Function, Vectorize
from jax import numpy as jn, tree_util
from functools import partial


from dptraining.datasets.base_creator import DataLoaderCreator
from dptraining.config import Config, AttackType, AttackInput
from dptraining.utils.attack_utils import rescale_and_shrink_network_params
from dptraining.models import make_model_from_config
from dptraining.utils import make_loss_from_config
from dptraining.utils.loss import LossFunctionCreator


class MIAWeightImage(Dataset):
    def __init__(
        self,
        model_weights: np.array,
        differing_samples: np.array,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        deterministic: bool = False,  # Always compare the same two samples
        model_architecture: Optional[Module] = None,
        return_model_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.model_weights = model_weights
        self.differing_samples = differing_samples
        self.transform: Callable = transform if transform else lambda _: _
        self.label_transform: Callable = (
            label_transform if label_transform else lambda _: _
        )
        self.deterministic = deterministic
        self.model_architecture = model_architecture
        self.return_model_outputs = return_model_outputs
        assert (self.return_model_outputs and self.model_architecture) or (
            not self.return_model_outputs
        )

    def __len__(self) -> int:
        return self.model_weights.shape[0]

    def __getitem__(self, index: int) -> tuple[tuple[np.array, np.array], int]:
        L = self.__len__()
        if self.deterministic:
            rng = np.random.Generator(np.random.PCG64(index))
            order = rng.choice(a=[True, False])
            other_idx = (index - rng.integers(self.__len__() - 1) - 1) % L
        else:
            order = np.random.choice(a=[True, False])
            other_idx = (
                index - np.random.randint(self.__len__() - 1) - 1
            ) % L  # this should prevent that the same other and idx are equal
        assert index != other_idx, "This shouldn't happen"
        model_weights = self.model_weights[index]
        true_sample = self.differing_samples[index]
        false_sample = self.differing_samples[other_idx]
        out: Optional[np.array] = None
        if self.return_model_outputs:
            match_weights_to_params(model_weights, self.model_architecture.vars())
            out = self.model_architecture(np.concatenate([true_sample, false_sample]))
        # This order so label 0 means first element is correct
        return (
            [
                model_weights,
                false_sample if order else true_sample,
                true_sample if order else false_sample,
                out,
            ],
            int(order),
        )


def flatten_grad(grad: VarCollection):
    return np.concatenate([g.reshape(g.shape[0], -1) for g in grad], axis=1)


class MIAOutGrad(Dataset):
    def __init__(
        self,
        model_weights: np.array,
        differing_samples: np.array,
        model_architecture: Optional[Module],
        sample_labels: Optional[np.array] = None,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        deterministic: bool = False,  # Always compare the same two samples
        return_grad: bool = False,
        loss_class: Optional[LossFunctionCreator] = None,
    ) -> None:
        super().__init__()
        self.model_weights: np.array = model_weights
        self.differing_samples: np.array = differing_samples
        self.transform: Callable = transform if transform else lambda _: _
        self.label_transform: Callable = (
            label_transform if label_transform else lambda _: _
        )
        self.deterministic: bool = deterministic
        self.model_architecture: Optional[Module] = model_architecture
        self.return_grad: bool = return_grad
        self.sample_labels: Optional[np.array] = sample_labels
        self.loss_fn: Optional[Union[Module, Callable]] = (
            loss_class.create_train_loss_fn(
                self.model_architecture.vars(), self.model_architecture
            )
            if loss_class is not None
            else None
        )
        assert (not self.return_grad) or (
            self.loss_fn is not None and self.sample_labels is not None
        )
        self.gv: Optional[GradValues] = (
            GradValues(self.loss_fn, self.model_architecture.vars())
            if self.return_grad
            else None
        )
        if self.gv:

            @Function.with_vars(self.gv.vars())
            def clipped_grad_single_example(*args):
                grads, values = self.gv(*args)
                return grads, values[0]

            self.calc_grad_vectorized = Vectorize(
                clipped_grad_single_example, batch_axis=(0, 0)
            )

    def __len__(self) -> int:
        return self.model_weights.shape[0]

    def __getitem__(self, index: int) -> tuple[tuple[np.array, np.array], int]:
        L = self.__len__()
        if self.deterministic:
            rng = np.random.Generator(np.random.PCG64(index))
            order = rng.choice(a=[True, False])
            other_idx = (index - rng.integers(self.__len__() - 1) - 1) % L
        else:
            order = np.random.choice(a=[True, False])
            other_idx = (
                index - np.random.randint(self.__len__() - 1) - 1
            ) % L  # this should prevent that the same other and idx are equal
        assert index != other_idx, "This shouldn't happen"
        model_weights = self.model_weights[index]
        match_weights_to_params(model_weights, self.model_architecture.vars())
        sample_idcs = [other_idx, index] if order else [index, other_idx]
        inp = self.differing_samples[sample_idcs]
        out = self.model_architecture(inp)
        if self.return_grad:
            labels = self.sample_labels[sample_idcs]
            grads, loss = self.calc_grad_vectorized(
                np.expand_dims(inp, axis=1), np.expand_dims(labels, axis=1)
            )
            grads, loss = tree_util.tree_map(partial(jn.stack, axis=0), (grads, loss))
            flat_grad = flatten_grad(grads)
            grad_norm = jn.linalg.norm(flat_grad, axis=1, keepdims=True)
            grad_mean = jn.mean(flat_grad, axis=1, keepdims=True)
            grad_std = jn.std(flat_grad, axis=1, keepdims=True)
            loss = np.expand_dims(loss, axis=1)
            return np.concatenate(
                [out, grad_norm, grad_mean, grad_std, loss], axis=1
            ), int(order)
        return out, int(order)


class ReconAttackDataset(Dataset):
    def __init__(
        self,
        attack_input: np.array,
        attack_output: np.array,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.attack_input = attack_input
        self.attack_output = attack_output
        self.transform: Callable = transform if transform else lambda _: _
        self.label_transform: Callable = (
            label_transform if label_transform else lambda _: _
        )

    def __getitem__(
        self, index: Union[int, list[int], np.array]
    ) -> tuple[np.array, np.array]:
        return (self.transform(self.attack_input[index]), self.attack_output[index])

    def __len__(self):
        return self.attack_input.shape[0]


def match_weights_to_params(weights: np.array, params: VarCollection) -> None:
    pointer: int = 0
    for key, param in params.items():
        n_params = np.prod(param.shape)
        weight = weights[pointer : pointer + n_params].reshape(param.shape)
        param.assign(jn.array(weight))
        pointer += n_params
    assert pointer == weights.shape[0]


class AttackCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: Config, transforms: tuple
    ) -> tuple[Dataset, Dataset, Dataset]:
        attack_data_dict = torchload(config.dataset.attack.attack_data_path)
        attack_train_weights = attack_data_dict["attack_train_weights"]
        attack_train_targets = attack_data_dict["reconstruction_train_targets"]
        attack_eval_weights = attack_data_dict["attack_eval_weights"]
        attack_eval_targets = attack_data_dict["reconstruction_eval_targets"]

        # if config.attack.orig_model:
        #     cfg = deepcopy(config)
        #     cfg.model = config.attack.orig_model
        #     orig_model = make_model_from_config(cfg)
        #     N = 100
        #     for i in range(10):
        #         match_weights_to_params(attack_train_weights[i], orig_model.vars())
        #         result = orig_model(attack_train_targets[:N].reshape(N, -1))
        #         labels = attack_data_dict["reconstruction_train_labels"]
        #         preds = functional.softmax(result, axis=1).argmax(axis=1)
        #         correct = (labels[:N] == preds).sum()
        #         print(f"\tModel[{i}]: \t{correct}/{N} ({100*correct/N:.1f}%)")

        if config.dataset.attack.rescale_params or config.dataset.attack.pca_dim:
            assert (
                config.dataset.attack.attack_input == AttackInput.weights_and_images
            ), "No PCA possible if weights are used for models"
            (
                attack_train_weights,
                attack_eval_weights,
            ) = rescale_and_shrink_network_params(
                config.dataset.attack.rescale_params,
                config.dataset.attack.pca_dim,
                attack_train_weights,
                attack_eval_weights,
                config.dataset.attack.include_eval_data_in_rescale_and_pca,
            )
        if config.dataset.attack.rescale_images or config.dataset.attack.pca_imgs:
            (
                attack_train_targets,
                attack_eval_targets,
            ) = rescale_and_shrink_network_params(
                config.dataset.attack.rescale_images,
                config.dataset.attack.pca_imgs,
                attack_train_targets,
                attack_eval_targets,
                config.dataset.attack.include_eval_data_in_rescale_and_pca,
            )

        match config.attack.type:
            case AttackType.RECON_INFORMED:
                train_ds = ReconAttackDataset(
                    attack_train_weights, attack_train_targets, transform=transforms[0]
                )
                test_split = int(
                    round(config.dataset.test_split * attack_eval_weights.shape[0])
                )
                random_samples = np.arange(attack_eval_weights.shape[0])
                rng = np.random.Generator(
                    np.random.PCG64(config.dataset.datasplit_seed)
                )
                rng.shuffle(random_samples)
                eval_idcs, test_idcs = (
                    random_samples[test_split:],
                    random_samples[:test_split],
                )
                eval_ds = ReconAttackDataset(
                    attack_eval_weights[eval_idcs],
                    attack_eval_targets[eval_idcs],
                    transform=transforms[1],
                )
                test_ds = ReconAttackDataset(
                    attack_eval_weights[test_idcs],
                    attack_eval_targets[test_idcs],
                    transform=transforms[2],
                )
            case AttackType.MIA_INFORMED:
                test_split = int(
                    round(config.dataset.test_split * attack_eval_weights.shape[0])
                )
                random_samples = np.arange(attack_eval_weights.shape[0])
                rng = np.random.Generator(
                    np.random.PCG64(config.dataset.datasplit_seed)
                )
                rng.shuffle(random_samples)
                eval_idcs, test_idcs = (
                    random_samples[test_split:],
                    random_samples[:test_split],
                )
                if config.dataset.attack.attack_input == AttackInput.weights_and_images:
                    train_ds = MIAWeightImage(
                        attack_train_weights,
                        attack_train_targets,
                        transform=transforms[0],
                    )
                    eval_ds = MIAWeightImage(
                        attack_eval_weights[eval_idcs],
                        attack_eval_targets[eval_idcs],
                        transform=transforms[1],
                        deterministic=True,
                    )
                    test_ds = MIAWeightImage(
                        attack_eval_weights[test_idcs],
                        attack_eval_targets[test_idcs],
                        transform=transforms[2],
                        deterministic=True,
                    )
                elif config.dataset.attack.attack_input in [
                    AttackInput.outputs,
                    AttackInput.outputs_and_grads,
                ]:
                    assert (
                        config.loader.num_workers == 0
                    ), "Unfortunately jax and torch multiprocessing are incompatible"
                    cfg = deepcopy(config)
                    cfg.model = config.attack.orig_model
                    orig_model = make_model_from_config(cfg)
                    calc_grad = (
                        config.dataset.attack.attack_input
                        == AttackInput.outputs_and_grads
                    )
                    loss_class = None
                    train_labels = None
                    eval_labels = None
                    test_labels = None
                    if calc_grad:
                        cfg = deepcopy(config)
                        cfg.loss = config.attack.orig_loss_fn
                        loss_class = make_loss_from_config(cfg)
                        train_labels = attack_data_dict["reconstruction_train_labels"]
                        eval_labels = attack_data_dict["reconstruction_eval_labels"]
                        test_labels = eval_labels[test_idcs]
                        eval_labels = eval_labels[eval_idcs]

                    train_ds = MIAOutGrad(
                        attack_train_weights,
                        attack_train_targets,
                        sample_labels=train_labels,
                        transform=transforms[0],
                        model_architecture=orig_model,
                        return_grad=calc_grad,
                        loss_class=loss_class,
                    )
                    # a, b = train_ds[0]
                    eval_ds = MIAOutGrad(
                        attack_eval_weights[eval_idcs],
                        attack_eval_targets[eval_idcs],
                        sample_labels=eval_labels,
                        transform=transforms[1],
                        model_architecture=orig_model,
                        deterministic=True,
                        return_grad=calc_grad,
                        loss_class=loss_class,
                    )
                    test_ds = MIAOutGrad(
                        attack_eval_weights[test_idcs],
                        attack_eval_targets[test_idcs],
                        sample_labels=test_labels,
                        transform=transforms[2],
                        model_architecture=orig_model,
                        deterministic=True,
                        return_grad=calc_grad,
                        loss_class=loss_class,
                    )
            case other:
                raise ValueError(f"Attack Type {other} not supported yet")

        return train_ds, eval_ds, test_ds
