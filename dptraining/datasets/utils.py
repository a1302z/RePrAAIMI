from typing import Union
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def collate_np_classification(
    list_of_data_tuples: list[tuple[np.array, list[Union[int, float]]]]
) -> tuple[np.array, np.array]:
    return (
        np.stack([b[0] for b in list_of_data_tuples]),
        np.array([b[1] for b in list_of_data_tuples], dtype=int),
    )


def collate_np_reconstruction(list_of_samples):
    if len(list_of_samples) > 1:
        list_of_outputs = tuple(
            np.stack([s[i] for s in list_of_samples], axis=0)
            for i in range(len(list_of_samples[0]))
        )
    else:
        list_of_outputs = tuple(list_of_samples)
    return list_of_outputs


def calc_mean_std(dataset: DataLoader):
    mean = 0.0
    for images, _ in tqdm(
        dataset, total=len(dataset), desc="calculating mean", leave=False
    ):
        batch_samples = images.shape[0]
        images = images.reshape((batch_samples, images.shape[1], -1))
        mean += images.mean(2).sum(0)
    mean = mean / len(dataset.dataset)

    var = 0.0
    reshaped_mean = mean[np.newaxis, ..., np.newaxis] if len(mean.shape) == 1 else mean
    for images, _ in tqdm(
        dataset, total=len(dataset), desc="calculating std", leave=False
    ):
        batch_samples = images.shape[0]
        images = images.reshape(batch_samples, images.shape[1], -1)
        var += ((images - reshaped_mean) ** 2).sum(2).sum(0)
    std = np.sqrt(var / (len(dataset.dataset) * 224 * 224))
    return mean, std
