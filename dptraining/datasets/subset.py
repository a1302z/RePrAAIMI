from typing import Any
from torch.utils.data import Dataset


class DataSubset(Dataset):
    def __init__(self, total_dataset: Dataset, indices: list[int]) -> None:
        super().__init__()
        self.total_dataset: Dataset = total_dataset
        self.indices: list[int] = indices
        assert max(self.indices) < len(
            self.total_dataset
        ), "Indices cover larger range than total dataset"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: Any) -> Any:
        return self.total_dataset[self.indices[index]]


class FixedAndShadowDatasetFromOneSet(DataSubset):
    def __init__(
        self, total_dataset: Dataset, fixed_indices: list[int], shadow_idcs: list[int]
    ) -> None:
        super().__init__(total_dataset, fixed_indices)
        self.shadow_indices: list[int] = shadow_idcs
        self.N_copies: int = len(shadow_idcs)

    def __len__(self):
        return super().__len__() + 1

    def get_shadow_size(self) -> int:
        return self.N_copies

    def __getitem__(self, index: Any) -> Any:
        data = None
        index = index % (super().__len__() + 1)
        if index == super().__len__():
            data = [self.total_dataset[idx] for idx in self.shadow_indices]
        elif index < len(self.total_dataset):
            data = [super().__getitem__(index)] * self.N_copies
        else:
            raise ValueError(f"Index out of dataset size")
        return data


class FixedAndShadowDatasetFromTwoSets(DataSubset):
    def __init__(
        self,
        fixed_dataset: Dataset,
        indices: list[int],
        shadow_dataset: Dataset,
        shadow_idcs: list[int],
    ) -> None:
        super().__init__(fixed_dataset, indices)
        self.shadow_dataset: Dataset = shadow_dataset
        self.shadow_indices: list[int] = shadow_idcs
        self.N_copies: int = len(shadow_idcs)

    def __len__(self):
        return super().__len__() + 1

    def get_shadow_size(self) -> int:
        return self.N_copies

    def __getitem__(self, index: Any) -> Any:
        data = None
        if index == super().__len__():
            data = [self.shadow_dataset[idx] for idx in self.shadow_indices]
        elif index < len(self.total_dataset):
            data = [super().__getitem__(index)] * self.N_copies
        else:
            raise ValueError(f"Index out of dataset size")
        return data


class DataSubsetPlusOne(DataSubset):
    def __init__(
        self, total_dataset: Dataset, indices: list[int], shadow_idx: int
    ) -> None:
        super().__init__(total_dataset, indices)
        self.shadow_idx: int = shadow_idx

    def __len__(self):
        return super().__len__() + 1

    def __getitem__(self, index: Any) -> Any:
        if index == super().__len__():
            return self.total_dataset[self.shadow_idx]
        elif index < len(self.total_dataset):
            return super().__getitem__(index)
        else:
            raise ValueError(f"Index out of dataset size")
