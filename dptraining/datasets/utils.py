from typing import Union, Callable
import numpy as np


def collate_np_classification(
    list_of_data_tuples: list[tuple[np.array, list[Union[int, float]]]]
) -> tuple[np.array, np.array]:
    return (
        np.stack([b[0] for b in list_of_data_tuples]),
        np.array([b[1] for b in list_of_data_tuples], dtype=int),
    )


def collate_np_reconstruction(
    list_of_samples: list[tuple[np.array]],
) -> tuple[np.array]:
    list_of_outputs: tuple[np.array]
    if len(list_of_samples) > 1:
        list_of_outputs = tuple(
            np.stack([s[i] for s in list_of_samples], axis=0)
            for i in range(len(list_of_samples[0]))
        )
    else:
        list_of_outputs = tuple(l[np.newaxis, ...] for l in list_of_samples[0])
    return list_of_outputs


def create_collate_fn_lists(
    collate_fn: Callable,
) -> list[tuple[np.array, Union[int, float, np.array]]]:
    def collate_fn_list(list_of_samples: list[list[tuple]]):
        return_value: list[tuple[np.array, Union[int, float, np.array]]] = []
        for i in range(len(list_of_samples[0])):  # TODO this can probably be vectorized
            return_value.append(
                collate_fn([list_of_samples[j][i] for j in range(len(list_of_samples))])
            )
        return return_value

    return collate_fn_list
