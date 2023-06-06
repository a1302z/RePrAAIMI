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
    list_of_data_tuples: list[tuple[np.array, list[dict], list[Union[int, float]]]] 
)-> tuple[np.array, np.array, np.array]:
    """ This function turns a list of data tuples into a tuple of input array, attribute dictionary and target array.
        Args:
            list_of_data_tuples: list of tuples of the form (input, attribute dictionary, target)
                                The attribute dictionary is a list of dictionaries of the form: {attr_name: (actual_val, possible_vals)}
        Returns:
            tuple of the form (input array, attribute dictionary, target array)
            where the attribute dictionary is a dictionary of the form {attr_name_actual_val: [indices of samples with that attribute]}
    """
    # convert the list of attribute dictionaries into a single dictionary
    attr_dict = {}
    num_attrs = len(list_of_data_tuples[0][1].keys())
    for i, (_, sample_attr_dict, _) in enumerate(list_of_data_tuples):
        if num_attrs != len(sample_attr_dict.keys()):
            raise ValueError("All samples must have the same number of attributes.")
        for attr_name, attr_vals in sample_attr_dict.items():
            actual_val, possible_vals = attr_vals
            # set up the dictionary for the possible values of atrr_name
            if i==0:
                for val in possible_vals:
                    attr_dict[f"{attr_name}_{val}"] = []
            # add index to the corresponding list
            attr_dict[f"{attr_name}_{actual_val}"].append(i)
    # convert attr_idcs to numpy arrays
    for attr_name, attr_idcs in attr_dict.items():
        attr_dict[attr_name] = np.array(attr_idcs)
            
    return(
        np.stack([b[0] for b in list_of_data_tuples]), # inputs
        attr_dict, # converted attribute dictionary
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
