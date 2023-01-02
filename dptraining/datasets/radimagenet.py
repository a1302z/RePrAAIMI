import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image
from splitfolders import split_class_dir_ratio
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))


from dptraining.datasets.base_creator import (
    DataLoaderCreator,
    mk_subdirectories,
)
from dptraining.utils.transform import NormalizeNumpyImg
from dptraining.config import DatasetTask

# from dptraining.datasets.utils import calc_mean_std

DATA_OUTPUT_TYPE = Tuple[np.array, Union[int, np.array]]  # pylint:disable=invalid-name


SUPPORTED_MODALITIES = ("mr", "ct", "us")

STATS = {
    "all": ([0.22039941], [0.24865805]),
    "ct": ([0.33009225], [0.32522697]),
    "mr": ([0.21530229], [0.22644264]),
    "us": ([0.1469403], [0.18063141]),
}


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
        """Extension of torchvision ImageFolder structure.
        Can handle deeper data structures.

        Args:
            root (str): Path to root folder.
            transform (Optional[Callable], optional):
                Optional callable to be applied to images. Defaults to None.
            target_transform (Optional[Callable], optional):
                Optional callable to be applied to labels. Defaults to None.
            loader (Callable[[str], Any], optional):
                Function to load data from disc. Defaults to default_loader.
            is_valid_file (Optional[Callable[[str], bool]], optional):
                Function to ensure a certain file fulfills some property.
                Defaults to None.
            is_valid_class (Optional[Callable[[str], bool]], optional):
                Function to in-/exclude classes in the dataset.
                Defaults to None.
        """
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

    def make_dataset(  # pylint:disable=arguments-renamed
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

            def is_valid_file(file: Path) -> bool:  # pylint:disable=function-redefined
                return file.is_file() and file.suffix in extensions

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
                msg += (
                    "Supported extensions are: "
                    f"{extensions if isinstance(extensions, str) else ', '.join(extensions)}"
                )
            raise FileNotFoundError(msg)

        return instances


class ConcatExtendedImageFolder(ConcatDataset):
    def __init__(self, datasets: List[ExtendedImageFolder]) -> None:
        super().__init__(datasets)
        self.classes: Dict[str, Path] = {
            f"{Path(d.root).name}/{class_name}": class_path
            for d in datasets
            for class_name, class_path in d.classes.items()
        }
        self.class_to_idx: Dict[str, int] = {}
        base_index: int = 0
        self.dataset_label_base: List[int] = []
        for dataset in self.datasets:
            for name, idx in dataset.class_to_idx.items():
                self.class_to_idx[f"{Path(dataset.root).name}/{name}"] = (
                    idx + base_index
                )
            base_index = len(self.class_to_idx)
            self.dataset_label_base.append(base_index)

    def __getitem__(self, index: int) -> DATA_OUTPUT_TYPE:
        if index < 0:
            if -index > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            index = len(self) + index
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        img, label = self.datasets[dataset_idx][sample_idx]
        label += self.dataset_label_base[dataset_idx - 1] if dataset_idx > 0 else 0
        return img, label


class RadImageNet(Dataset):

    NORMLIZATION_TRANSFORMS = {
        modality: NormalizeNumpyImg(*stats) for modality, stats in STATS.items()
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        task: DatasetTask = DatasetTask.classification,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        modality: str = "all",  # pylint:disable=redefined-outer-name
        allowed_body_regions: Union[str, List[str]] = "all",
        allowed_labels: Union[str, List[str]] = "all",
        normalize_by_modality: bool = False,
    ) -> None:
        """
        Wrapper for RadImageNet dataset structure.

        Args:
            root_dir (Union[str, Path]):
                Path to directory where RadImageNet is stored
            task DatasetTask:
                If set to reconstruction returns images also as labels.
                Defaults to DatasetTask.classification,.
            transform (Optional[Callable], optional):
                Callable to be applied to images. Defaults to None.
            target_transform (Optional[Callable], optional):
                Callable to be applied to labels. Defaults to None.
            modality (str, optional):
                Whether to use all radiological modalities (CT, MR, US).
                Defaults to "all".
            allowed_body_regions (Union[str, List[str]], optional):
                Whether to restrict to certain body parts in the dataset.
                Defaults to "all".
            allowed_labels (Union[str, List[str]], optional):
                Whether to restrict to certain diagnoses. Defaults to "all".
            normalize_by_modality (bool, optional):
                Normalize images with stats of each modality for the cost of
                a slight performance loss. Defaults to False.

        Raises:
            ValueError: allowed_body_regions must be either a string or list of strings
            ValueError: allowed_labels must be either a string or list of strings

        Returns:
            None: No return value
        """
        assert modality.lower() in SUPPORTED_MODALITIES or modality.lower() in ("all",)
        super().__init__()
        self.root_dir: Path = root_dir if isinstance(root_dir, Path) else Path(root_dir)
        self.task: DatasetTask = task
        self.modality: str = modality
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        self.normalize_by_modality: bool = normalize_by_modality
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
                is_valid_label = (
                    lambda _: True  # pylint:disable=unnecessary-lambda-assignment
                )
            else:
                if isinstance(allowed_labels, str):

                    def is_valid_label(file: Path) -> bool:
                        return allowed_labels == file.name

                elif isinstance(allowed_labels, list):

                    def is_valid_label(file: Path) -> bool:
                        return file.name in allowed_labels

                else:
                    raise ValueError(
                        f"Subclass {allowed_labels} must either be str or list[str]"
                    )

            is_valid_region: Callable
            if allowed_body_regions == "all":
                is_valid_region = (
                    lambda _: True  # pylint:disable=unnecessary-lambda-assignment
                )
            else:
                if isinstance(allowed_body_regions, str):

                    def is_valid_region(file: Path) -> bool:
                        return allowed_body_regions == file.parent.name

                elif isinstance(allowed_body_regions, list):

                    def is_valid_region(file: Path) -> bool:
                        return file.parent.name in allowed_body_regions

                else:
                    raise ValueError(
                        f"Subclass {allowed_body_regions} must either be str or list[str]"
                    )

            def is_valid_class(file: Path) -> bool:
                return is_valid_label(file) and is_valid_region(file)

        def loader(path: Path):
            with open(path, "rb") as file:
                img = Image.open(file)
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
        self.idx_to_class = {
            idx: class_name for class_name, idx in self.dataset.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> DATA_OUTPUT_TYPE:
        image, label = self.dataset[index]
        if self.normalize_by_modality:
            if self.modality == "all":
                modality = (  # pylint:disable=redefined-outer-name
                    self.idx_to_class[label].split("/")[0].lower()
                )
            else:
                modality = self.modality
            image = RadImageNet.NORMLIZATION_TRANSFORMS[modality.lower()](image)
        if self.task == DatasetTask.reconstruction:
            label = image
        return self.transform(image), self.target_transform(label)


class RadImageNetCreator(DataLoaderCreator):
    @staticmethod
    def make_datasets(
        config: dict, transforms: Tuple
    ) -> Tuple[Dataset, Dataset, Dataset]:
        task = config.dataset.task
        root_folder = Path(config.dataset.root)
        train_split, test_split = (
            config.dataset.train_val_split,
            config.dataset.test_split,
        )
        val_split = 1.0 - train_split - test_split
        assert val_split > 0, "Train and test split are combined larger than 1"
        seed = (
            config.dataset.radimagenet.datasplit_seed
            if config.dataset.radimagenet.datasplit_seed
            else config.general.seed
        )
        copy_folder = (
            config.dataset.radimagenet.split_folder
            if config.dataset.radimagenet.split_folder
            else root_folder.parent
            / (
                f"{root_folder.name}_dataset_split_{train_split}_"
                f"{val_split:.2f}_{test_split}_seed={seed}"
            )
        )
        if copy_folder.is_dir():
            train_val_test_dirs = [
                copy_folder / subdir for subdir in ["train", "val", "test"]
            ]
            assert all((subdir.is_dir() for subdir in train_val_test_dirs))
            print(
                f"Folder {copy_folder} already exists and will not be rebuilt. "
                "If changes happen which require a rebuild please delete manually."
            )
        else:
            copy_folder.mkdir()
            classes, _ = find_classes(root_folder)
            out_class_paths = []
            for _, class_path in tqdm(
                classes.items(),
                total=len(classes),
                desc="building split copy of data",
                leave=False,
            ):
                out_class_path = copy_folder
                for subfolder in class_path.relative_to(root_folder).parts:
                    out_class_path /= subfolder
                split_class_dir_ratio(
                    class_path,
                    output=out_class_path,
                    ratio=(train_split, val_split, test_split),
                    seed=seed,
                    prog_bar=None,
                    group_prefix=None,
                    move=False,
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
                        len([file for file in del_folder.rglob("*") if file.is_file()])
                        == 0
                    ):
                        del_folder.rmdir()
                    else:
                        break
        (train_set, val_set, test_set) = (
            RadImageNet(
                new_root,
                transform=tf,
                task=task,
                modality=config.dataset.radimagenet.modality,
                allowed_body_regions=config.dataset.radimagenet.allowed_body_regions,
                allowed_labels=config.dataset.radimagenet.allowed_labels,
                normalize_by_modality=config.dataset.radimagenet.normalize_by_modality,
            )
            for (new_root, tf) in zip(train_val_test_dirs, transforms)
        )

        return train_set, val_set, test_set


if __name__ == "__main__":

    # from collections import Counter

    from dptraining.datasets.utils import collate_np_classification

    ds1 = ExtendedImageFolder("./data/radiology_ai/CT")
    ds2 = ExtendedImageFolder("./data/radiology_ai/MR")

    ds = ConcatExtendedImageFolder([ds1, ds2])
    print(ds.classes)
    print(ds.class_to_idx)
    all_labels = {f"CT/{k}" for k in ds1.classes.keys()}.union(
        {f"MR/{k}" for k in ds2.classes.keys()}
    )
    concat_labels = set(ds.classes.keys())
    label_idcs = set(ds.class_to_idx.values())
    assert len(all_labels - concat_labels) == 0
    assert len(concat_labels - all_labels) == 0
    assert len(set(range(len(ds1.classes) + len(ds2.classes))) - label_idcs) == 0
    assert len(label_idcs - set(range(len(ds1.classes) + len(ds2.classes)))) == 0
    # labels = [
    #     label
    #     for _, label in tqdm(
    #         ds, total=len(ds), leave=False, desc="iterate concat dataset"
    #     )
    # ]
    # print(Counter(labels))

    ds1 = RadImageNet(
        "./data/radiology_ai",
        task="classification",
        transform=None,
        normalize_by_modality=True,
    )
    _ = ds1[0]
    print(len(ds1))

    data_loader1 = DataLoader(
        ds1,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_np_classification,
        num_workers=16,
        prefetch_factor=8,
    )
    # data_mean, data_std = calc_mean_std(data_loader1)
    # print(f"Total: Mean: {data_mean}\t Std: {data_std}")

    for modality in ["CT", "MR", "US"]:
        ds2 = RadImageNet(
            "./data/radiology_ai",
            task="classification",
            transform=None,
            modality=modality,
            normalize_by_modality=True,
        )
        _ = ds2[0]
        print(len(ds2))
        data_loader2 = DataLoader(
            ds2,
            batch_size=512,
            shuffle=False,
            collate_fn=collate_np_classification,
            num_workers=16,
            prefetch_factor=8,
        )
        # data_mean, data_std = calc_mean_std(data_loader2)
        # print(f"{modality}: Mean: {data_mean}\t Std: {data_std}")
    ds3 = RadImageNet(
        "./data/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_body_regions="knee",
    )
    _ = ds3[0]
    print(len(ds3))
    ds4 = RadImageNet(
        "./data/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_labels="normal",
    )
    _ = ds4[0]
    print(len(ds4))
    ds5 = RadImageNet(
        "./data/radiology_ai",
        task="reconstruction",
        transform=None,
        target_transform=None,
        allowed_labels="normal",
        allowed_body_regions="knee",
    )
    _ = ds5[0]
    print(len(ds5))
