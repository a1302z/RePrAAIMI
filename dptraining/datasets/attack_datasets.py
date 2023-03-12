import numpy as np
from typing import Union, Optional, Callable
from torch import load as torchload
from torch.utils.data import Dataset

from dptraining.datasets.base_creator import DataLoaderCreator
from dptraining.config import Config, AttackType
from dptraining.utils.attack_utils import rescale_and_shrink_network_params


class AttackDataset(Dataset):
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

        attack_train_weights, attack_eval_weights = rescale_and_shrink_network_params(
            config, attack_train_weights, attack_eval_weights
        )

        if config.attack.type == AttackType.RECON_INFORMED:
            train_ds = AttackDataset(
                attack_train_weights, attack_train_targets, transform=transforms[0]
            )
            test_split = int(
                round(config.dataset.test_split * attack_eval_weights.shape[0])
            )
            random_samples = np.arange(attack_eval_weights.shape[0])
            rng = np.random.Generator(np.random.PCG64(config.dataset.datasplit_seed))
            rng.shuffle(random_samples)
            eval_idcs, test_idcs = (
                random_samples[test_split:],
                random_samples[:test_split],
            )
            eval_ds = AttackDataset(
                attack_eval_weights[eval_idcs],
                attack_eval_targets[eval_idcs],
                transform=transforms[1],
            )
            test_ds = AttackDataset(
                attack_eval_weights[test_idcs],
                attack_eval_targets[test_idcs],
                transform=transforms[2],
            )
        else:
            raise ValueError(f"Attack Type {config.attack.type} not supported yet")

        return train_ds, eval_ds, test_ds
