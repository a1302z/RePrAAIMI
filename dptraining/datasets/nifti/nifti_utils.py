from typing import Optional


import nibabel as nib
import numpy as np
from nilearn.image import resample_img

# sys.path.insert(0, str(Path.cwd()))

from dptraining.config import Normalization, CTWindow


def scale_array_zero_one(array: np.array) -> np.array:
    """Scales a numpy array from 0 to 1. Works in 3D
    Return np.array"""
    assert array.max() - array.min() > 0

    return ((array - array.min()) / (array.max() - array.min())).astype(np.float32)


def scale_array_unit_gaussian(
    array: np.array,
    mean: np.array,
    std: np.array,
) -> np.array:
    return (array - mean) / std


def rotate_label(label_volume) -> np.array:
    """Rotates and flips the label in the same way the scans were rotated and flipped
    Return: np.array"""

    label_volume = np.rot90(label_volume)
    label_volume = np.fliplr(label_volume)

    return label_volume.astype(np.float32)


def preprocess_scan(
    scan: np.array,
    ct_window: CTWindow,
    normalization: Normalization,
    mean_std: tuple[float],
) -> np.array:
    """Performs Preprocessing:
    - clips vales to -150 to 200,
    - peforms rotations and flipping to move patient into reference position
    Return: np.array"""
    if ct_window is not None and (
        ct_window.low is not None or ct_window.high is not None
    ):
        scan = np.clip(scan, a_min=ct_window.low, a_max=ct_window.high)
    match normalization:
        case Normalization.zeroone:
            scan = scale_array_zero_one(scan)
        case Normalization.gaussian:
            scan = scale_array_unit_gaussian(scan, *mean_std)
        case Normalization.consecutive:
            # not really necessary but may be useful if
            # stats were calculated over 0,1 interval
            scan = scale_array_zero_one(scan)
            scan = scale_array_unit_gaussian(scan, *mean_std)
    scan = np.rot90(scan)
    scan = np.fliplr(scan)

    return scan


def preprocess_and_convert_to_numpy(
    nifti_scan: nib.Nifti1Image,
    nifti_mask: nib.Nifti1Image,
    ct_window: CTWindow,
    normalization: Normalization,
    mean_std: tuple[float],
) -> list:
    """Convert scan and label to numpy arrays and perform preprocessing
    Return: Tuple(np.array, np.array)"""
    np_scan = nifti_scan.get_fdata()
    np_label = nifti_mask.get_fdata()
    nifti_mask.uncache()
    nifti_scan.uncache()
    np_scan = preprocess_scan(np_scan, ct_window, normalization, mean_std)
    np_label = rotate_label(np_label)
    assert np_scan.shape == np_label.shape, (
        f"Scan has shape {np_scan.shape} while label has {np_label.shape}"
        f"\nNifti data: {nifti_scan.header.get_data_shape()}"
        f"\t {nifti_mask.header.get_data_shape()}"
    )

    return np_scan, np_label


def resize_scan(
    scan: nib.Nifti1Header,
    label: nib.Nifti1Header,
    slice_thickness: Optional[float] = None,
    resolution: Optional[float] = None,
    n_slices: Optional[float] = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    data_shape = scan.header.get_data_shape()
    zooms = scan.header.get_zooms()
    # print(
    #     f"Actual size: {data_shape[0]*zooms[0]:.1f}mm "
    #     f"x {data_shape[1]*zooms[1]:.1f}mm "
    #     f"x {data_shape[2]*zooms[2]:.1f}mm"
    # )
    z_shape = data_shape[2]
    if slice_thickness:
        z_shape = int(data_shape[2] * zooms[2] / slice_thickness)
    if n_slices:
        z_shape = n_slices
    new_shape = (
        resolution if resolution else data_shape[0],
        resolution if resolution else data_shape[1],
        z_shape,
    )
    new_affine = np.copy(scan.affine)
    for i, new_shape_i in enumerate(new_shape):
        new_affine[i, i] *= data_shape[i] / new_shape_i
    # print(zooms)
    if slice_thickness:
        new_affine[2, 2] = slice_thickness
    elif n_slices:
        new_affine[2, 2] = (data_shape[2] * zooms[2]) / n_slices
    else:
        new_affine[2, 2] = data_shape[2]
    scan = resample_img(
        scan,
        target_affine=new_affine,
        target_shape=new_shape,
        interpolation="continuous",
    )
    label = resample_img(
        label,
        target_affine=new_affine,
        target_shape=new_shape,
        interpolation="nearest",
    )
    # if self.n_slices:
    #     new_zooms = scan.header.get_zooms()
    # print(
    #     f"Old num slices: {data_shape[2]}\tOld thickness: {zooms[2]:.2f}"
    #     f"\tNew num slices: {self.n_slices}\tNew thickness: {new_zooms[2]:.2f}"
    # )

    # print(
    #     f"New real size: {new_shape[0]*new_zooms[0]:.1f}mm "
    #     f"x {new_shape[1]*new_zooms[1]:.1f}mm "
    #     f"x {new_shape[2]*new_zooms[2]:.1f}mm"
    # )
    return scan, label
