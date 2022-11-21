import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from typing import Tuple, Union, Callable, Optional, List, Dict, cast, Any
from bisect import bisect_right
from torch.utils.data import random_split
from splitfolders import ratio
from warnings import warn

from PIL import Image


from enum import Enum
from tqdm import tqdm

from pathlib import Path

import sys

sys.path.insert(0, str(Path.cwd()))


from dptraining.datasets.base_creator import DataLoaderCreator

DATA_OUTPUT = Tuple[np.array, Union[int, np.array]]


SUPPORTED_MODALITIES = ("mr", "ct", "us")
SUPPORTED_TASKS = ("classification", "reconstruction")


class Task(Enum):
    CLASSIFICATION = 1
    RECONSTRUCTION = 2


def find_classes(directory: str) -> Tuple[Dict[str, Path], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    directory = directory if isinstance(directory, Path) else Path(directory)
    class_paths = [
        dirs
        for dirs in tqdm(
            directory.rglob("*"), desc="Searching data folder", leave=False
        )
        if dirs.is_dir()
        and next(dirs.iterdir()) is not None  # non empty
        and len([f for f in dirs.iterdir() if f.is_dir()]) == 0
    ]
    classes = dict(
        sorted({str(dirs.relative_to(directory)): dirs for dirs in class_paths}.items())
    )

    if not classes or len(classes) == 0:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ExtendedImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        is_valid_class: Optional[Callable[[str], bool]] = None,
    ):
        super(DatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        classes, class_to_idx = self.find_classes(self.root)
        if is_valid_class is None:
            self.classes, self.class_to_idx = classes, class_to_idx
        else:
            self.classes, self.class_to_idx = {}, {}
            idx = 0
            for c_name, c_path in classes.items():
                if is_valid_class(c_path):
                    self.classes[c_name] = c_path
                    self.class_to_idx[c_name] = idx
                    idx += 1

        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        samples = self.make_dataset(
            self.root,
            extensions,
            is_valid_file,
        )

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def make_dataset(
        self,
        directory: str,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:

        directory = Path(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: Path) -> bool:
                return x.is_file() and x.suffix in extensions

        is_valid_file = cast(Callable[[Path], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class, target_dir in tqdm(
            self.classes.items(),
            total=len(self.classes),
            desc="building dataset",
            leave=False,
        ):
            class_index = self.class_to_idx[target_class]
            if not target_dir.is_dir():
                continue
            for data_file in sorted(target_dir.iterdir()):
                if is_valid_file(data_file):
                    item = data_file, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(self.classes.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


# class ConcatExtendedImageFolder(ConcatDataset):
#     def __init__(self, datasets: List[ExtendedImageFolder]) -> None:
#         super().__init__(datasets)
#         self.classes: Dict[str, Path] = {
#             f"{Path(d.root).name}/{class_name}": class_path
#             for d in datasets
#             for class_name, class_path in d.classes.items()
#         }
#         self.class_to_idx: Dict[str, int] = {}
#         base_index: int = 0
#         self.dataset_label_base: List[int] = []
#         for dataset in self.datasets:
#             for name, idx in dataset.class_to_idx.items():
#                 self.class_to_idx[f"{Path(dataset.root).name}/{name}"] = (
#                     idx + base_index
#                 )
#             base_index = len(self.class_to_idx)
#             self.dataset_label_base.append(base_index)

#     def __getitem__(self, index: int) -> DATA_OUTPUT:
#         if index < 0:
#             if -index > len(self):
#                 raise ValueError(
#                     "absolute value of index should not exceed dataset length"
#                 )
#             index = len(self) + index
#         dataset_idx = bisect_right(self.cumulative_sizes, index)
#         if dataset_idx == 0:
#             sample_idx = index
#         else:
#             sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
#         img, label = self.datasets[dataset_idx][sample_idx]
#         label += self.dataset_label_base[dataset_idx - 1] if dataset_idx > 0 else 0
#         return img, label


class RadImageNet(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        task: str = "classification",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        modality: str = "all",
        allowed_body_regions: Union[str, List[str]] = "all",
        allowed_labels: Union[str, List[str]] = "all",
    ) -> None:
        assert modality.lower() in SUPPORTED_MODALITIES or modality.lower() in ("all",)
        assert task in SUPPORTED_TASKS
        super().__init__()
        self.root_dir: Path = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.task: Task = Task[task.upper()]
        self.modality: str = modality
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        is_valid_class: Optional[Callable]
        if (
            isinstance(allowed_labels, str)
            and allowed_labels == "all"
            and isinstance(allowed_body_regions, str)
            and allowed_body_regions == "all"
        ):
            is_valid_class = None
        else:
            is_valid_label: Callable
            if allowed_labels == "all":
                is_valid_label = lambda x: True
            else:
                if isinstance(allowed_labels, str):

                    def is_valid_label(x: Path) -> bool:
                        return allowed_labels == x.name

                elif isinstance(allowed_labels, list):

                    def is_valid_label(x: Path) -> bool:
                        return x.name in allowed_labels

                else:
                    raise ValueError(
                        f"Subclass {allowed_labels} must either be str or list[str]"
                    )

            is_valid_region: Callable
            if allowed_body_regions == "all":
                is_valid_region = lambda x: True
            else:
                if isinstance(allowed_body_regions, str):

                    def is_valid_region(x: Path) -> bool:
                        return allowed_body_regions == x.parent.name

                elif isinstance(allowed_body_regions, list):

                    def is_valid_region(x: Path) -> bool:
                        return x.parent.name in allowed_body_regions

                else:
                    raise ValueError(
                        f"Subclass {allowed_body_regions} must either be str or list[str]"
                    )

            def is_valid_class(x: Path) -> bool:
                return is_valid_label(x) and is_valid_region(x)

        def loader(path: Path):
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("L")
                img = np.array(img).astype(np.float32) / 255.0
                return img[np.newaxis, ...]

        if modality == "all":
            self.dataset = ExtendedImageFolder(
                self.root_dir, is_valid_class=is_valid_class, loader=loader
            )
        else:
            self.dataset = ExtendedImageFolder(
                self.root_dir / self.modality.upper(),
                is_valid_class=is_valid_class,
                loader=loader,
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_OUTPUT:
        image, label = self.dataset[index]
        match self.task:
            case Task.CLASSIFICATION:
                return self.transform(image), self.target_transform(label)
            case Task.RECONSTRUCTION:
                return self.transform(image), self.target_transform(image)
        raise ValueError("Task undefined")


def mk_subdirectories(path: Path, subdirs: List[str]) -> List[Path]:
    new_dirs = []
    for subdir in subdirs:
        new_path: Path = path / subdir
        if not new_path.is_dir():
            new_path.mkdir()
        new_dirs.append(new_path)
    return new_dirs


class RadImageNetCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: dict, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        task = (
            config["dataset"]["task"]
            if "task" in config["dataset"]
            else "classification"
        )
        match config["dataset"]["split"]:
            case "random":
                warn(
                    f"We strongly recommend to use stratified split mode instead of random"
                )
                if any([tf is not None for tf in transforms]):
                    raise ValueError(
                        "No transforms supported for RadImageNet in random split mode. "
                        "We recommend to use augmentations."
                    )
                total_dataset = RadImageNet(
                    config["dataset"]["root"],
                    task=task,
                    transform=np.array,
                    target_transform=np.array if task == "reconstruction" else None,
                    modality=config["dataset"]["modality"]
                    if "modality" in config["dataset"]
                    else "all",
                    allowed_body_regions=config["dataset"]["allowed_body_regions"]
                    if "allowed_body_regions" in config["dataset"]
                    else "all",
                    allowed_labels=config["dataset"]["allowed_labels"]
                    if "allowed_labels" in config["dataset"]
                    else "all",
                )
                L_train_val = int(
                    round(len(total_dataset) * config["dataset"]["test_split"])
                )
                train_val_set, test_set = random_split(
                    total_dataset,
                    lengths=[L_train_val, len(total_dataset) - L_train_val],
                )
                L_train = int(
                    round(len(train_val_set) * config["dataset"]["train_val_split"])
                )
                train_set, val_set = random_split(
                    train_val_set, lengths=[L_train, len(train_val_set) - L_train]
                )
            case "stratified":
                root_folder = Path(config["dataset"]["root"])
                train_split, test_split = (
                    config["dataset"]["train_val_split"],
                    config["dataset"]["test_split"],
                )
                val_split = 1.0 - train_split - test_split
                assert val_split > 0, f"Train and test split are combined larger than 1"
                seed = (
                    config["dataset"]["datasplit_seed"]
                    if "datasplit_seed" in config["dataset"]
                    else config["general"]["seed"]
                )
                copy_folder = (
                    config["dataset"]["split_folder"]
                    if "split_folder" in config["dataset"]
                    else root_folder.parent
                    / f"{root_folder.name}_dataset_split_{train_split}_{val_split:.2f}_{test_split}_seed={seed}"
                )
                if copy_folder.is_dir():
                    train_val_test_dirs = [
                        copy_folder / subdir for subdir in ["train", "val", "test"]
                    ]
                    assert all([subdir.is_dir() for subdir in train_val_test_dirs])
                    print(
                        f"Folder {copy_folder} already exists and will not be rebuilt. "
                        "If changes happen which require a rebuild please delete manually."
                    )
                else:
                    copy_folder.mkdir()
                    classes, _ = find_classes(root_folder)
                    out_class_paths = []
                    parent_paths = set()
                    for class_name, class_path in tqdm(
                        classes.items(),
                        total=len(classes),
                        desc="building split copy of data",
                        leave=False,
                    ):
                        parent_path = class_path.parent
                        if parent_path in parent_paths:
                            continue
                        parent_paths.add(parent_path)
                        out_class_path = copy_folder
                        for subfolder in class_name.split("/")[:-1]:
                            out_class_path /= subfolder
                        ratio(
                            parent_path,
                            output=out_class_path,
                            ratio=(train_split, val_split, test_split),
                            seed=seed,
                        )
                        out_class_paths.append(out_class_path)
                    train_val_test_dirs = mk_subdirectories(
                        copy_folder, ["train", "val", "test"]
                    )
                    for out_class_path in tqdm(
                        out_class_paths,
                        total=len(out_class_paths),
                        leave=False,
                        desc="moving classes",
                    ):
                        diff = out_class_path.relative_to(copy_folder)
                        for subdir in tqdm(
                            train_val_test_dirs,
                            total=len(train_val_test_dirs),
                            leave=False,
                            desc=f"moving {diff}",
                        ):
                            new_dir = subdir / diff
                            new_dir.mkdir(parents=True, exist_ok=False)
                            data_dir = out_class_path / subdir.name
                            data_dir.rename(new_dir)
                        diff_parts = diff.parts
                        for i in tqdm(
                            range(len(diff_parts), 0, -1),
                            total=len(diff_parts),
                            desc="Delete empty dirs",
                            leave=False,
                        ):
                            del_folder = copy_folder
                            for j in range(i):
                                del_folder /= diff_parts[j]
                            if (
                                len(
                                    [
                                        file
                                        for file in del_folder.rglob("*")
                                        if file.is_file()
                                    ]
                                )
                                == 0
                            ):
                                del_folder.rmdir()
                            else:
                                break
                (train_set, val_set, test_set) = (
                    RadImageNet(
                        new_root,
                        transform=tf,
                        task=config["dataset"]["task"],
                        modality=config["dataset"]["modality"]
                        if "modality" in config["dataset"]
                        else "all",
                        allowed_body_regions=config["dataset"]["allowed_body_regions"]
                        if "allowed_body_regions" in config["dataset"]
                        else "all",
                        allowed_labels=config["dataset"]["allowed_labels"]
                        if "allowed_labels" in config["dataset"]
                        else "all",
                    )
                    for (new_root, tf) in zip(train_val_test_dirs, transforms)
                )

            case other:
                raise ValueError(
                    f"Data split {other} not supported. "
                    "Supported are [random, stratified]"
                )

        return train_set, val_set, test_set


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from dptraining.datasets.utils import collate_np_arrays

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
        reshaped_mean = mean[np.newaxis, ...]
        for images, _ in tqdm(
            dataset, total=len(dataset), desc="calculating std", leave=False
        ):
            batch_samples = images.shape[0]
            images = images.reshape(batch_samples, images.shape[1], -1)
            var += ((images - reshaped_mean) ** 2).sum(2).sum(0)
        std = np.sqrt(var / (len(dataset.dataset) * 224 * 224))
        return mean, std

    ds1 = RadImageNet(
        "/media/alex/NVME/radiology_ai", task="classification", transform=None
    )
    ds1[0]
    print(len(ds1))

    data_loader1 = DataLoader(
        ds1,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_np_arrays,
        num_workers=16,
        prefetch_factor=8,
    )
    data_mean, data_std = calc_mean_std(data_loader1)
    print(f"Total: Mean: {data_mean}\t Std: {data_std}")

    for modality in ["CT", "MR", "US"]:
        ds2 = RadImageNet(
            "/media/alex/NVME/radiology_ai",
            task="classification",
            transform=None,
            modality=modality,
        )
        ds2[0]
        print(len(ds2))
        data_loader2 = DataLoader(
            ds2,
            batch_size=512,
            shuffle=False,
            collate_fn=collate_np_arrays,
            num_workers=16,
            prefetch_factor=8,
        )
        data_mean, data_std = calc_mean_std(data_loader2)
        print(f"{modality}: Mean: {data_mean}\t Std: {data_std}")
    exit()
    ds3 = RadImageNet(
        "/media/alex/NVME/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_body_regions="knee",
    )
    ds3[0]
    print(len(ds3))
    ds4 = RadImageNet(
        "/media/alex/NVME/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_labels="normal",
    )
    ds4[0]
    print(len(ds4))
    ds5 = RadImageNet(
        "/media/alex/NVME/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_labels="normal",
        allowed_body_regions="knee",
    )
    ds5[0]
    print(len(ds5))
