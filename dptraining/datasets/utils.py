from typing import Union
import numpy as np


def collate_np_arrays(
    list_of_data_tuples: list[tuple[np.array, list[Union[int, float]]]]
) -> tuple[np.array, np.array]:
    return (
        np.stack([b[0] for b in list_of_data_tuples]),
        np.array([b[1] for b in list_of_data_tuples], dtype=int),
    )
