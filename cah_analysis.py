# # When the Curious Abandon Honesty: Federated Learning is not Private

# %% [markdown]
# This notebook shows an example for the threat model and attack described in "When the Curious Abandon Honesty: Federated Learning is not Private". This example deviates from the other "honest-but-curious" server models and investigates an actively malicious model. As such, the attack applies to any model architecture, but its impact is more or less obvious (or not at all) depending on the already present architecture onto which the malicious "Imprint" block is grafted.
#
# **Important:** This is the opposite framing from the original work, where this is conceptualized as a "malicious parameters" attack which is valid against models with contain a layer structure of, for example, only fully connected layers or only convolutions without strides and fully connected layers. Both viewpoints are equally valid, but we like the "malicious-model" view point which allows the attack to be applied to any model, instead of having to preselect vulnerable architectures.
#
#
# Paper URL: https://arxiv.org/abs/2112.02918

# %% [markdown]
# ### Abstract:
#
# In federated learning (FL), data does not leave personal devices when they are jointly training a machine learning model. Instead, these devices share gradients with a central party (e.g., a company). Because data never "leaves" personal devices, FL is presented as privacy-preserving. Yet, recently it was shown that this protection is but a thin facade, as even a passive attacker observing gradients can reconstruct data of individual users. In this paper, we argue that prior work still largely underestimates the vulnerability of FL. This is because prior efforts exclusively consider passive attackers that are honest-but-curious. Instead, we introduce an active and dishonest attacker acting as the central party, who is able to modify the shared model's weights before users compute model gradients. We call the modified weights "trap weights". Our active attacker is able to recover user data perfectly and at near zero costs: the attack requires no complex optimization objectives. Instead, it exploits inherent data leakage from model gradients and amplifies this effect by maliciously altering the weights of the shared model. These specificities enable our attack to scale to models trained with large mini-batches of data. Where attackers from prior work require hours to recover a single data point, our method needs milliseconds to capture the full mini-batch of data from both fully-connected and convolutional deep neural networks. Finally, we consider mitigations. We observe that current implementations of differential privacy (DP) in FL are flawed, as they explicitly trust the central party with the crucial task of adding DP noise, and thus provide no protection against a malicious central party. We also consider other defenses and explain why they are similarly inadequate. A significant redesign of FL is required for it to provide any meaningful form of data privacy to users.

# %% [markdown]
# ### Startup

# %%
import numpy as np
import torch
import math
from omegaconf import open_dict
import os
from warnings import warn
from copy import deepcopy
import lpips
from matplotlib import pyplot as plt, colors as mplcolors, ticker as mtick
from matplotlib import image as mpimg
from pathlib import Path
from tqdm import tqdm, trange
from datetime import datetime
import cv2
import pandas as pd


from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

import seaborn as sn

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-cn", "--config", required=True, help="Path to training config.")
parser.add_argument(
    "-b", "--num_bins", default=None, help="Number of reconstruction images", type=int
)
parser.add_argument(
    "-nb", "--N_batches", default=5, help="How many batches to reconstruct", type=int
)
parser.add_argument(
    "-ni",
    "--N_images",
    default=5,
    help="How many images to visualize in summary",
    type=int,
)
parser.add_argument(
    "-l2",
    "--use_l2_distance",
    action="store_true",
    help="Use l2 distance as comparison of reconstructions and original "
    "images instead of perceptual distance",
)
parser.add_argument(
    "-eb",
    "--eps_range_begin",
    type=int,
    default=9,
    help="Exponential factor for first epsilon",
)
parser.add_argument(
    "-ee",
    "--eps_range_end",
    type=int,
    default=18,
    help="Exponential factor for last epsilon",
)
parser.add_argument(
    "-es",
    "--eps_range_steps",
    type=int,
    default=3,
    help="Step size between eps exp factors",
)
parser.add_argument(
    "--jax_threshold",
    type=float,
    default=1e10,
    help="Use iterative jax implementation to calculate L2 distance between vectors above this threshold",
)
parser.add_argument(
    "--allow_torch_cuda",
    action="store_true",
    help="Allow torch to access gpu (for perc dist)",
)
parser.add_argument(
    "--privacy_for_one_step",
    action="store_true",
    help="Calculate noise as if only one step was performed",
)
parser.add_argument(
    "--reevaluate_folder",
    type=Path,
    default=None,
    help="Use previous reconstructions to reevaluate results (e.g. if figures changed)",
)
args = parser.parse_args()


sn.set_theme(
    context="notebook",
    style="white",
    font="Times New Roman",
    font_scale=1,
    palette="viridis",
)
sn.despine()
sn.set(rc={"figure.figsize": (12, 12)})
colors = {
    "red": "firebrick",
    "blue": "steelblue",
    "green": "forestgreen",
    "purple": "darkorchid",
    "orange": "darkorange",
    "gray": "lightslategray",
    "black": "black",
}

from breaching import breaching
from breachingobjaxutils.initialize import make_configs


def flatten_features(ftrs):
    return np.stack(
        [np.concatenate([v.flatten().cpu().numpy() for v in d.values()]) for d in ftrs],
        axis=0,
    )


calc_rero = lambda dist, thresh: 100.0 * (dist < thresh).sum(axis=0) / dist.shape[0]

# %%
cfg, config = make_configs(
    ["attack=imprint", "case/server=malicious-model-cah"],
    # "breaching/test_train_config.yaml",
    # "configs/ham10000.yaml",
    args.config,
)
# if train_config.general.cpu:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from jax import numpy as jn


from dptraining.privacy import setup_privacy, analyse_epsilon
from dptraining.datasets import make_loader_from_config
from dptraining.utils.training_utils import fix_seeds
from breachingobjaxutils.objaxbasedfunctions import (
    make_make_fns,
    get_transform_normalization,
    get_aug_normalization,
)

fix_seeds(config)

# from jax import vmap, numpy as jn


# mean_diff = jax.pmap(
#     lambda x, y: jn.mean(jn.square(jn.subtract(x, y)), axis=1), (0, None), 0
# )
# def mean_diff(x, y):
#     results = []
#     for element in tqdm(x, total=x.shape[0], leave=False, desc="Calc dist"):
#         res = np.mean(np.square(np.subtract(element, y)), axis=1)
#         results.append(res)
#     return np.stack(results, axis=0)


class ProxySet(torch.utils.data.Dataset):
    def __init__(self, batches: list) -> None:
        super().__init__()
        self.batches: list = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i: int):
        return self.batches[i]


def mean_diff_jax_alongX(a1: jn.DeviceArray, a2: jn.DeviceArray):
    X, N = a1.shape
    X1, M = a2.shape
    assert X == X1
    out = jn.zeros((N, M))
    for l, r in tqdm(zip(a1, a2), total=X):
        out += jn.square(l[:, None] - r)
    out /= X
    return out


def mean_diff_np_alongNX(a1, a2, patchsize=1000):
    N, X = a1.shape
    M, X1 = a2.shape
    assert X == X1
    out = np.zeros((N, M))
    a1 = a1.T.copy()
    a2 = jn.array(a2.T.copy())
    N_patches = N // patchsize + (N % patchsize > 0) * 1
    for i in trange(N_patches):
        patch = jn.array(a1[:, i * patchsize : (i + 1) * patchsize])
        out[i * patchsize : (i + 1) * patchsize] = np.array(
            mean_diff_jax_alongX(patch, a2)
        )
        del patch
    return out


def save_imgs_to_path(imgs, batch_p_recon):
    counter = 0
    if not batch_p_recon.is_dir():
        batch_p_recon.mkdir(parents=True)
    for img in imgs:
        cv2.imwrite(str(batch_p_recon / f"{counter}.png"), img.astype(np.uint8))
        counter += 1


def convert_to_cv2(recon_imgs):
    recon_imgs = recon_imgs.transpose(0, 2, 3, 1)
    recon_imgs = recon_imgs - recon_imgs.min(
        axis=tuple(range(1, len(recon_imgs.shape))), keepdims=True
    )
    normaliser = recon_imgs.max(
        axis=tuple(range(1, len(recon_imgs.shape))), keepdims=True
    )
    normaliser = np.where(np.abs(normaliser) < 1e-9, 1.0, normaliser)
    recon_imgs /= normaliser
    recon_imgs *= 255.0
    if recon_imgs.shape[-1] == 3:  # convert RBG to BRG
        recon_imgs = recon_imgs[..., [2, 1, 0]]
    return recon_imgs


def normalise_array_to_zero_one(x, axis):
    x = x - x.min(axis=axis, keepdims=True)
    denom = x.max(axis=axis, keepdims=True)
    denom = np.where(denom < 1e-9, 1.0, denom)
    return x / denom


if args.reevaluate_folder:
    read_directory = Path(args.reevaluate_folder)
    save_directory = (
        read_directory.parent
        / f"{read_directory.stem}_reevaluated_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    save_directory.mkdir()
    privacy_level_dirs = [d for d in read_directory.iterdir() if d.is_dir()]
    privacy_level_dirs.sort(key=lambda p: p.stem.replace("eps=", ""))
    recon_data_per_eps = [
        pd.read_csv(d / "recon_assignment.csv") for d in privacy_level_dirs
    ]
    eps_values = [d.stem.replace("eps=", "") for d in privacy_level_dirs]
else:
    device = (
        torch.device(f"cuda")
        if args.allow_torch_cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    # device = torch.device("cpu")
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

    eps_values = [
        10**i
        for i in range(
            args.eps_range_begin, args.eps_range_end + 1, args.eps_range_steps
        )
    ] + ["Non-private"]
    # eps_values = [1, 1e6, 1e9, 1e12]
    # eps_values = ["Non-private"] + [1e15]

    recon_data_per_eps = []
    lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)
    save_directory = Path(
        f"reconstructions/{str(config.dataset.name).split('.')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    )

    train_loader, _, _ = make_loader_from_config(config)

    accountant = create_accountant(config.DP.mechanism)

    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i > args.N_batches * config.hyperparams.grad_acc_steps + 1:
            break

    proxy_dataset = ProxySet(batches)
    proxy_loader = torch.utils.data.DataLoader(
        proxy_dataset, batch_size=1, collate_fn=lambda _: _[0]
    )
    example_batch, example_label = proxy_dataset[0]
    data_dimensions = sum([s > 3 for s in example_batch.shape[1:]])

    aug_fn = None
    stats = get_transform_normalization(config)
    if stats is not None:
        dm, ds = stats
    else:
        stats = get_aug_normalization(config)
        if stats is not None:
            dm, ds, aug_fn = stats
        elif (
            config.dataset.nifti_seg_options is not None
            and config.dataset.nifti_seg_options.data_stats is not None
        ):
            dm, ds = (
                np.array([config.dataset.nifti_seg_options.data_stats.mean]),
                np.array([config.dataset.nifti_seg_options.data_stats.std]),
            )
        else:
            warn("Did not find stats. Will continue assuming unnormalized input.")
            dm, ds = np.array([0.0]), np.array([1.0])
    if (
        config.dataset.nifti_seg_options is not None
        and config.dataset.nifti_seg_options.ct_window is not None
    ):
        recon_clip_threshold = (
            config.dataset.nifti_seg_options.ct_window.low,
            config.dataset.nifti_seg_options.ct_window.high,
        )
    else:
        recon_clip_threshold = (0.0, 1.0)

    new_shape = [1] * len(example_batch.shape)
    new_shape[1] = ds.shape[0]
    dm, ds = dm.reshape(*new_shape), ds.reshape(*new_shape)
    stats = (dm, ds)

    # save_imgs_to_path(convert_to_cv2(np.array(example_batch)), Path("./test_path"))
    for eps in tqdm(
        eps_values, total=len(eps_values), desc="Privacy levels", leave=False
    ):
        eps_path = save_directory / (eps if isinstance(eps, str) else f"eps={eps:.0E}")
        # if eps == "Random noise":
        #     true_user_data = dict(data=example_batch, label=example_label)
        #     reconstructed_user_data = dict(
        #         data=np.random.randn(*example_batch.shape) * ds + dm, label=None
        #     )
        # else:
        train_config = deepcopy(config)
        if isinstance(eps, (float, int)):
            train_config.DP.epsilon = eps
            train_config.DP.eps_tol = eps * 10e-6
        else:
            train_config.DP = None

        make_model, make_loss_gv_from_model = make_make_fns(train_config)

        if train_config.DP:
            if args.privacy_for_one_step:
                effective_batch_size = (
                    config.hyperparams.grad_acc_steps * config.hyperparams.batch_size
                )
                batch_expansion_factor = config.hyperparams.grad_acc_steps
                sampling_rate: float = effective_batch_size / len(train_loader.dataset)
                delta = config.DP.delta
                steps = 1
                opt_func = lambda noise: analyse_epsilon(
                    accountant, steps, noise, sampling_rate, delta
                )
                sigma = get_noise_multiplier(
                    target_epsilon=train_config.DP.epsilon,
                    target_delta=delta,
                    sample_rate=sampling_rate,
                    steps=steps,
                    accountant=config.DP.mechanism,
                )
                total_noise = sigma * config.DP.max_per_sample_grad_norm
            else:
                (
                    grad_acc,
                    accountant,
                    sampling_rate,
                    delta,
                    sigma,
                    total_noise,
                    batch_expansion_factor,
                    effective_batch_size,
                ) = setup_privacy(train_config, train_loader)

                steps = (
                    len(train_loader) // batch_expansion_factor
                ) * train_config.hyperparams.epochs
            with open_dict(cfg):
                cfg.case.user.total_noise = total_noise
            print(
                f"actual epsilon: {analyse_epsilon(accountant,steps,sigma,sampling_rate,delta, config.DP.alphas):.2E} with noise multiplier {sigma}"
            )

        # %%
        cfg.case.user.num_data_points = config.hyperparams.batch_size
        # How many data points does this user own
        cfg.case.server.model_modification.type = "CuriousAbandonHonesty"  # What type of Imprint block will be grafted to the model
        cfg.case.server.model_modification.num_bins = (
            args.num_bins
            if args.num_bins
            else config.hyperparams.batch_size * config.hyperparams.grad_acc_steps * 2
        )  # How many bins are in the block

        cfg.case.server.model_modification.position = None  # '4.0.conv'
        cfg.case.server.model_modification.connection = "addition"
        # cfg.case.data.path = "/media/alex/NVME/ILSVRC2012/"

        # Unnormalized data:
        # cfg.case.data.normalize = False
        # cfg.case.server.model_modification.mu = 0
        # cfg.case.server.model_modification.sigma = 0.5
        # cfg.case.server.model_modification.scale_factor = 1 - 0.990
        # cfg.attack.breach_reduction = None # Will be done manually

        # Normalized data:
        cfg.case.data.normalize = True
        cfg.case.data.mean = None
        cfg.case.data.std = None
        cfg.case.data.shape = example_batch.shape[1:]
        cfg.case.server.model_modification.sigma = float(0.5 * np.mean(ds))
        cfg.case.server.model_modification.mu = float(
            -np.mean(dm) * math.sqrt(np.prod(example_batch.shape[1:])) * 0.5
        )
        cfg.case.server.model_modification.scale_factor = -0.9990
        cfg.attack.breach_reduction = None  # Will be done manually

        user, server, model_fn = breaching.cases.construct_case(
            train_config,
            cfg.case,
            make_model,
            dataloader=(proxy_loader, aug_fn),
            make_loss_grad=make_loss_gv_from_model,
        )
        attacker = breaching.attacks.prepare_attack(
            model_fn, make_loss_gv_from_model, cfg.attack, stats
        )
        server_payload = server.distribute_payload()
        min_dist_total = []
        recon_counter = 0
        gt_counter = 0
        stats_df = None
        for batch in tqdm(
            range(args.N_batches),
            total=args.N_batches,
            desc="reconstructing inputs",
            leave=False,
        ):
            shared_data, true_user_data = user.compute_local_updates(server_payload)

            reconstructed_user_data, recon_stats = attacker.reconstruct(
                [server_payload],
                [shared_data],
                server.secrets,
                dryrun=cfg.dryrun,
                rescale=config.dataset.nifti_seg_options is None,
            )
            true_user_data = {k: np.array(v) for k, v in true_user_data.items()}
            del shared_data
            reconstructed_user_data["data"] = np.minimum(
                np.maximum(
                    np.array(reconstructed_user_data["data"]) * ds + dm,
                    recon_clip_threshold[0],
                ),
                recon_clip_threshold[1],
            )
            true_user_data["data"] = np.minimum(
                np.maximum(
                    np.array(true_user_data["data"]) * ds + dm, recon_clip_threshold[0]
                ),
                recon_clip_threshold[1],
            )
            axis = tuple(range(1, len(true_user_data["data"].shape)))
            reconstructed_user_data["data"] = normalise_array_to_zero_one(
                reconstructed_user_data["data"], axis
            )
            true_user_data["data"] = normalise_array_to_zero_one(
                true_user_data["data"], axis
            )
            if args.use_l2_distance:
                features_gt = true_user_data["data"].reshape(
                    true_user_data["data"].shape[0], -1
                )
                features_rec = reconstructed_user_data["data"].reshape(
                    reconstructed_user_data["data"].shape[0], -1
                )
            else:
                rec_denormalized = (
                    torch.from_numpy(reconstructed_user_data["data"])
                    .to(**setup)
                    .float()
                )
                denormalized_images = (
                    torch.from_numpy(true_user_data["data"]).to(**setup).float()
                )

                features_gt = breaching.analysis.calc_perceptual_features(
                    lpips_scorer,
                    denormalized_images,
                )
                features_rec = breaching.analysis.calc_perceptual_features(
                    lpips_scorer,
                    rec_denormalized,
                )
                features_gt = flatten_features(features_gt)
                features_rec = flatten_features(features_rec)
                assert (
                    np.isnan(features_gt).sum() == 0
                    and np.isnan(features_rec).sum() == 0
                )

            N, M, X = features_gt.shape[0], features_rec.shape[0], features_gt.shape[1]
            if (
                N * M * X < args.jax_threshold
            ):  # this is just an empirical threshold and strongly depends on the hardware
                perc_dist = np.power(features_gt[:, None, ...] - features_rec, 2).mean(
                    axis=2
                )
            else:  # memory optimised version
                perc_dist = mean_diff_np_alongNX(features_rec, features_gt, 1500).T

            # l2_dists = np.power(
            #     np.array(true_user_data["data"][:, None, ...])
            #     - np.array(reconstructed_user_data["data"]),
            #     2,
            # ).mean(axis=(2, 3, 4))

            thresh = 1e-3

            min_dist, closest_recon_to_gt = np.min(perc_dist, axis=1), np.argmin(
                perc_dist, axis=1
            )
            min_idx_rec = np.argsort(np.min(perc_dist, axis=0))

            if data_dimensions > 2:
                recon_imgs = convert_to_cv2(
                    reconstructed_user_data["data"][
                        min_idx_rec, ..., reconstructed_user_data["data"].shape[-1] // 2
                    ]
                )
                gt_imgs = convert_to_cv2(
                    true_user_data["data"][..., true_user_data["data"].shape[-1] // 2]
                )
            else:
                recon_imgs = convert_to_cv2(
                    reconstructed_user_data["data"][min_idx_rec]
                )
                gt_imgs = convert_to_cv2(true_user_data["data"])
            batch_p = eps_path / f"{batch}"
            batch_p_recon = batch_p / "recon"
            batch_p_gt = batch_p / "gt"
            save_imgs_to_path(recon_imgs, batch_p_recon)
            save_imgs_to_path(gt_imgs, batch_p_gt)

            # np.save(str(batch_p / "distances_closest.npy"), min_dist)
            # np.save(str(batch_p / "idcs_closest.npy"), min_idx)
            def get_indices_of_values(larger_array, smaller_array):
                indices = []
                for value in smaller_array:
                    matching_indices = np.where(larger_array == value)[0]
                    indices.append(matching_indices)
                return np.concatenate(indices)

            batch_df = pd.DataFrame.from_dict(
                {
                    "min_distance": min_dist,
                    "gt_path": [
                        str(batch_p_gt / f"{i}.png") for i in range(min_dist.shape[0])
                    ],
                    "recon_path": [
                        str(batch_p_recon / f"{i}.png")
                        for i in get_indices_of_values(min_idx_rec, closest_recon_to_gt)
                    ],
                    "gt_id": [
                        batch * min_dist.shape[0] + i for i in range(min_dist.shape[0])
                    ],
                }
            )
            batch_df.to_csv(str(batch_p / "recon_assignment.csv"), index=False)
            # l2min_dist, l2min_idx = np.min(l2_dists, axis=1), np.argmin(l2_dists, axis=1)
            # reconstructed = reconstructed_user_data["data"][min_idx]

            # rero = calc_rero(min_dist, thresh)
            # print(f"ReRo@{thresh}: {rero:.2f}%")
            min_dist_total.append(min_dist)

            # idcs_pcpt = np.argsort(min_dist)
            # idcs_l2 = np.argsort(l2min_dist)
            if stats_df is None:
                stats_df = batch_df
            else:
                stats_df = pd.concat([stats_df, batch_df], ignore_index=True)
        stats_df.to_csv(eps_path / "recon_assignment.csv", index=True)
        stats_df["eps"] = eps
        recon_data_per_eps.append(stats_df)


def turn_axis_off(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


cmap = plt.cm.Dark2
cmaplist = [cmap(i) for i in range(cmap.N)]
# cmap = mplcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
# norm = mplcolors.BoundaryNorm(
#     [b - 1e-12 if i > 0 else b for i, b in enumerate(eps_values)]
#     + [2 * eps_values[-1] - eps_values[-2] for i in range(2)],
#     cmap.N,
# )
total_df = pd.concat(recon_data_per_eps, ignore_index=True)
total_df = total_df.sort_values(by="min_distance")
total_df.to_csv(str(save_directory / "all_reconstructions_by_recon_loss.csv"))
total_df = total_df.drop_duplicates(subset="gt_id")
total_df = total_df.rename(
    columns={"min_distance": "global_min_distance", "eps": "eps_best_recon"}
)
total_df.to_csv(str(save_directory / "best_recon_for_each_gt_img.csv"))
total_df = total_df.drop(columns=["recon_path", "gt_path"])


fig, axs = plt.subplots(
    len(eps_values),
    2 * args.N_images,
    figsize=(10 * args.N_images, 5 * len(eps_values)),
    sharey=True,
)
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
dist_metric_name: str
if args.reevaluate_folder:
    dist_metric_name = "Distance"
elif args.use_l2_distance:
    dist_metric_name = "MSE"
else:
    dist_metric_name = "Perceptual distance"
ax2.set_xlabel(dist_metric_name)
# ax2[1].set_xlabel("Mean squared distance")
ax2.set_ylabel("Images within distance")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
for ax, eps, stats_df, col in tqdm(
    zip(axs, eps_values, recon_data_per_eps, cmaplist),
    total=len(eps_values),
    desc="building figures",
    leave=False,
):
    eps_formatted = f"ε={eps:.0E}" if isinstance(eps, (float, int)) else eps
    stats_df = pd.merge(left=total_df, right=stats_df, on="gt_id")
    stats_df = stats_df.sort_values(by="global_min_distance")

    # stats_df = stats_df.iloc[stats_df.min_distance.argsort()]
    # reduced_stats_df = stats_df.drop_duplicates(subset="gt_path")
    # recon_data = recon_data.transpose(0, 2, 3, 1)
    # idcs_pcpt = np.argsort(perc_dist)
    # idcs_l2 = np.argsort(l2_dist)
    ax[0].set_ylabel(eps_formatted, fontsize=20)
    for i in range(args.N_images):
        row = stats_df.iloc[i]
        dist, gt_p, recon_p = (
            row.min_distance,
            row.gt_path,
            row.recon_path,
        )
        gt_img = mpimg.imread(gt_p)
        recon_img = mpimg.imread(recon_p)
        plt_kwargs = {}
        if len(gt_img.shape) == 2:
            plt_kwargs["cmap"] = "gray"
        ax[2 * i].imshow(gt_img, **plt_kwargs)
        ax[2 * i + 1].imshow(recon_img, **plt_kwargs)
        turn_axis_off(ax[2 * i])
        turn_axis_off(ax[2 * i + 1])
        ax[2 * i + 1].set_xlabel(f"d={dist:.0E}", fontsize=16)
        ax[2 * i].set_title("Original", fontsize=20)
        ax[2 * i + 1].set_title("Closest Reconstruction", fontsize=20)

    # col = cmap(norm(eps))
    x1 = stats_df.min_distance.to_numpy().copy()
    # x2 = l2_dist.copy()
    x1.sort()
    # x2.sort()
    y = 100.0 * np.linspace(0, x1.shape[0], num=x1.shape[0]) / x1.shape[0]
    ax2.step(x1, y, label=eps_formatted, color=col)
    # ax2[1].step(x2, y, color=col)
    # ax2[0].set_title("Perceptual")
    # ax2[1].set_title("MSE")
fig2.legend(loc="upper left", bbox_to_anchor=(0.12, 0.89))


fig.savefig(str(save_directory / "cah_analysis.png"), dpi=600, bbox_inches="tight")
fig.savefig(str(save_directory / "cah_analysis.svg"), bbox_inches="tight")
fig2.savefig(
    str(save_directory / "rero_thresholding.png"), dpi=600, bbox_inches="tight"
)
fig2.savefig(str(save_directory / "rero_thresholding.svg"), bbox_inches="tight")
