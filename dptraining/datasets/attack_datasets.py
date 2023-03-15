import numpy as np
from typing import Union, Optional, Callable
from torch import load as torchload
from torch.utils.data import Dataset

from dptraining.datasets.base_creator import DataLoaderCreator
from dptraining.config import Config, AttackType
from dptraining.utils.attack_utils import rescale_and_shrink_network_params


class MIADataset(Dataset):
    def __init__(
        self,
        model_weights: np.array,
        differing_samples: np.array,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        deterministic: bool = False,  # Always compare the same two samples
    ) -> None:
        super().__init__()
        self.model_weights = model_weights
        self.differing_samples = differing_samples
        self.transform: Callable = transform if transform else lambda _: _
        self.label_transform: Callable = (
            label_transform if label_transform else lambda _: _
        )
        self.deterministic = deterministic

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
        # This order so label 0 means first element is correct
        return (
            [
                model_weights,
                false_sample if order else true_sample,
                true_sample if order else false_sample,
            ],
            int(order),
        )


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

        if config.dataset.attack.rescale_params or config.dataset.attack.pca_dim:
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
                train_ds = MIADataset(
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
                eval_ds = MIADataset(
                    attack_eval_weights[eval_idcs],
                    attack_eval_targets[eval_idcs],
                    transform=transforms[1],
                    deterministic=True,
                )
                test_ds = MIADataset(
                    attack_eval_weights[test_idcs],
                    attack_eval_targets[test_idcs],
                    transform=transforms[2],
                    deterministic=True,
                )
            case other:
                raise ValueError(f"Attack Type {other} not supported yet")

        return train_ds, eval_ds, test_ds
