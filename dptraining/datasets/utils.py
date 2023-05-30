from typing import Union
import numpy as np


def collate_np_classification(
    list_of_data_tuples: list[tuple[np.array, list[Union[int, float]]]]
) -> tuple[np.array, np.array]:
    return (
        np.stack([b[0] for b in list_of_data_tuples]), # inputs
        np.array([b[1] for b in list_of_data_tuples], dtype=int),  #targets
    )

def collate_np_classification_attributes(
    list_of_data_tuples: list[tuple[np.array, list[Union[int, float]], list[Union[int, float]]]] 
)-> tuple[np.array, np.array, np.array]:
    return(
        np.stack([b[0] for b in list_of_data_tuples]), # inputs
        np.array([b[1] for b in list_of_data_tuples], dtype=int), # attributes
        np.array([b[2] for b in list_of_data_tuples], dtype=int) # targets
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
