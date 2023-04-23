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
from tqdm import tqdm
from datetime import datetime
import cv2
import pandas as pd


from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

import seaborn as sn

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
        [np.concatenate([v.flatten().numpy() for v in d.values()]) for d in ftrs],
        axis=0,
    )


calc_rero = lambda dist, thresh: 100.0 * (dist < thresh).sum(axis=0) / dist.shape[0]

# Redirects logs directly into the jupyter notebook
# import logging, sys
# logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
# logger = logging.getLogger()

# %% [markdown]
# ### Initialize cfg object and system setup:

# %% [markdown]
# This will load the full configuration object. This includes the configuration for the use case and threat model as `cfg.case` and the hyperparameters and implementation of the attack as `cfg.attack`. All parameters can be modified below, or overriden with `overrides=` as if they were cmd-line arguments.

# %%
cfg, config = make_configs(
    ["attack=imprint", "case/server=malicious-model-cah"],
    # "breaching/test_train_config.yaml",
    # "configs/ham10000.yaml",
    "configs/imagefolder.yaml",
)
# if train_config.general.cpu:
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dptraining.privacy import setup_privacy, analyse_epsilon
from dptraining.datasets import make_loader_from_config
from dptraining.utils.training_utils import fix_seeds
from breachingobjaxutils.objaxbasedfunctions import (
    make_make_fns,
    get_transform_normalization,
    get_aug_normalization,
)

fix_seeds(config)

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))


# eps_values = [8, 1e6] + [10**i for i in range(9, 18, 1)] + ["Non-private"]
# eps_values = [1, 1e6, 1e9, 1e12]
eps_values = [10] + ["Non-private"]

recon_data_per_eps = []
lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)
p = Path(
    f"reconstructions/{str(config.dataset.name).split('.')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
)
N_batches = 1

train_loader, _, _ = make_loader_from_config(config)


# def make_compare_img(true_img, recon_img):
#     assert all([ts == rs for ts, rs in zip(true_img.shape, recon_img.shape)])
#     show_img = np.zeros(
#         (recon_img.shape[0] * 2, *recon_img.shape[1:]), dtype=recon_img.dtype
#     )
#     show_img[::2] = true_img
#     show_img[1::2] = recon_img
#     return show_img


# def double_labels(labels):
#     db_labels = np.zeros(2 * labels.shape[0])
#     db_labels[::2] = labels
#     db_labels[1::2] = labels
#     return db_labels

accountant = create_accountant(config.DP.mechanism)

PRIVACY_FOR_ENTIRE_TRAINING = True


class ProxySet(torch.utils.data.Dataset):
    def __init__(self, batches: list) -> None:
        super().__init__()
        self.batches: list = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i: int):
        return self.batches[i]


batches = []
for i, batch in enumerate(train_loader):
    batches.append(batch)
    if i > N_batches * config.hyperparams.grad_acc_steps + 1:
        break

proxy_dataset = ProxySet(batches)
proxy_loader = torch.utils.data.DataLoader(
    proxy_dataset, batch_size=1, collate_fn=lambda _: _[0]
)
example_batch, example_label = proxy_dataset[0]


aug_fn = None
tf_stats = get_transform_normalization(config)
if tf_stats is not None:
    dm, ds = tf_stats
else:
    aug_stats = get_aug_normalization(config)
    if aug_stats is None:
        warn("Did not find stats. Will continue assuming unnormalized input.")
        dm, ds = np.array([0.0]), np.array([1.0])
    else:
        dm, ds, aug_fn = aug_stats

new_shape = [1] * len(example_batch.shape)
new_shape[1] = ds.shape[0]
dm, ds = dm.reshape(*new_shape), ds.reshape(*new_shape)
stats = (dm, ds)


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
    recon_imgs /= recon_imgs.max(
        axis=tuple(range(1, len(recon_imgs.shape))), keepdims=True
    )
    recon_imgs *= 255.0
    if recon_imgs.shape[-1] == 3:  # convert RBG to BRG
        recon_imgs = recon_imgs[..., [2, 1, 0]]
    return recon_imgs


# save_imgs_to_path(convert_to_cv2(np.array(example_batch)), Path("./test_path"))

for eps in tqdm(eps_values, total=len(eps_values), desc="Privacy levels", leave=False):
    eps_path = p / (eps if isinstance(eps, str) else f"eps={eps:.0f}")
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
        if PRIVACY_FOR_ENTIRE_TRAINING:
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
        else:
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
            total_noise = get_noise_multiplier(
                target_epsilon=train_config.DP.epsilon,
                target_delta=delta,
                sample_rate=sampling_rate,
                steps=steps,
                accountant=config.DP.mechanism,
            )

        with open_dict(cfg):
            cfg.case.user.total_noise = total_noise
        print(
            f"actual epsilon: {analyse_epsilon(accountant,steps,cfg.case.user.total_noise,sampling_rate,delta, config.DP.alphas):.2f} with noise multiplier {total_noise}"
        )

    # %% [markdown]
    # ### Modify config options here

    # %% [markdown]
    # You can use `.attribute` access to modify any of these configurations for the attack, or the case:

    # %%
    cfg.case.user.num_data_points = config.hyperparams.batch_size
    # How many data points does this user own
    cfg.case.server.model_modification.type = "CuriousAbandonHonesty"  # What type of Imprint block will be grafted to the model
    cfg.case.server.model_modification.num_bins = 128  # How many bins are in the block

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

    # %% [markdown]
    # ### Instantiate all parties

    # %% [markdown]
    # The following lines generate "server, "user" and "attacker" objects and print an overview of their configurations.

    # %%
    # user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
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
    # breaching.utils.overview(server, user, attacker)

    # %% [markdown]
    # ### Simulate an attacked FL protocol

    # %% [markdown]
    # This exchange is a simulation of a single query in a federated learning protocol. The server sends out a `server_payload` and the user computes an update based on their private local data. This user update is `shared_data` and contains, for example, the parameter gradient of the model in the simplest case. `true_user_data` is also returned by `.compute_local_updates`, but of course not forwarded to the server or attacker and only used for (our) analysis.

    # %%
    server_payload = server.distribute_payload()
    min_dist_total = []
    recon_counter = 0
    gt_counter = 0
    stats_df = None
    for batch in tqdm(
        range(N_batches), total=N_batches, desc="reconstructing inputs", leave=False
    ):
        shared_data, true_user_data = user.compute_local_updates(server_payload)

        reconstructed_user_data, recon_stats = attacker.reconstruct(
            [server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun
        )

        rec_denormalized = torch.clamp(
            torch.from_numpy(np.array(reconstructed_user_data["data"])).to(**setup) * ds
            + dm,
            0,
            1,
        ).float()
        denormalized_images = torch.clamp(
            torch.from_numpy(np.array(true_user_data["data"])).to(**setup) * ds + dm,
            0,
            1,
        ).float()

        recon_imgs = convert_to_cv2(np.array(reconstructed_user_data["data"]))
        gt_imgs = convert_to_cv2(np.array(true_user_data["data"]))
        batch_p = eps_path / f"{batch}"
        batch_p_recon = batch_p / "recon"
        batch_p_gt = batch_p / "gt"
        save_imgs_to_path(recon_imgs, batch_p_recon)
        save_imgs_to_path(gt_imgs, batch_p_gt)

        features_gt_raw = breaching.analysis.calc_perceptual_features(
            lpips_scorer,
            denormalized_images,
        )
        features_rec_raw = breaching.analysis.calc_perceptual_features(
            lpips_scorer,
            rec_denormalized,
        )

        features_gt = flatten_features(features_gt_raw)
        features_rec = flatten_features(features_rec_raw)

        perc_dist = np.power(features_gt[:, None, ...] - features_rec, 2).mean(axis=2)

        # l2_dists = np.power(
        #     np.array(true_user_data["data"][:, None, ...])
        #     - np.array(reconstructed_user_data["data"]),
        #     2,
        # ).mean(axis=(2, 3, 4))

        thresh = 1e-3

        min_dist, min_idx = np.min(perc_dist, axis=1), np.argmin(perc_dist, axis=1)
        # np.save(str(batch_p / "distances_closest.npy"), min_dist)
        # np.save(str(batch_p / "idcs_closest.npy"), min_idx)

        batch_df = pd.DataFrame.from_dict(
            {
                "min_distance": min_dist,
                "gt_path": [
                    str(batch_p_gt / f"{i}.png") for i in range(min_dist.shape[0])
                ],
                "recon_path": [str(batch_p_recon / f"{i}.png") for i in min_idx],
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
            stats_df = pd.concat([stats_df, batch_df])
    stats_df.to_csv(eps_path / "recon_assignment.csv", index=False)
    recon_data_per_eps.append(stats_df)


def turn_axis_off(ax):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


TOP_N = 10
example_batch = example_batch.transpose(0, 2, 3, 1)
cmap = plt.cm.Dark2
cmaplist = [cmap(i) for i in range(cmap.N)]
# cmap = mplcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
# norm = mplcolors.BoundaryNorm(
#     [b - 1e-12 if i > 0 else b for i, b in enumerate(eps_values)]
#     + [2 * eps_values[-1] - eps_values[-2] for i in range(2)],
#     cmap.N,
# )
plt_kwargs = {}
if example_batch.shape[-1] == 1:
    plt_kwargs["cmap"] = "gray"

fig, axs = plt.subplots(
    len(eps_values), 2 * TOP_N, figsize=(10 * TOP_N, 5 * len(eps_values)), sharey=True
)
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
ax2.set_xlabel("Perceptual distance")
# ax2[1].set_xlabel("Mean squared distance")
ax2.set_ylabel("Images within distance")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
for ax, eps, stats_df, col in tqdm(
    zip(axs, eps_values, recon_data_per_eps, cmaplist),
    total=len(eps_values),
    desc="building figures",
    leave=False,
):
    eps_formatted = f"ε={eps:.0f}" if isinstance(eps, (float, int)) else eps

    stats_df = stats_df.iloc[stats_df.min_distance.argsort()]
    # reduced_stats_df = stats_df.drop_duplicates(subset="gt_path")
    # recon_data = recon_data.transpose(0, 2, 3, 1)
    # idcs_pcpt = np.argsort(perc_dist)
    # idcs_l2 = np.argsort(l2_dist)
    ax[0].set_ylabel(eps_formatted, fontsize=20)
    for i in range(TOP_N):
        dist, gt_p, recon_p = stats_df.iloc[i]
        ax[2 * i].imshow(mpimg.imread(gt_p), **plt_kwargs)
        ax[2 * i + 1].imshow(mpimg.imread(recon_p), **plt_kwargs)
        turn_axis_off(ax[2 * i])
        turn_axis_off(ax[2 * i + 1])
        ax[2 * i + 1].set_xlabel(f"d={dist:.3f}", fontsize=16)
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


fig.savefig(str(p / "cah_analysis.png"), dpi=600, bbox_inches="tight")
fig.savefig(str(p / "cah_analysis.svg"), bbox_inches="tight")
fig2.savefig(str(p / "rero_thresholding.png"), dpi=600, bbox_inches="tight")
fig2.savefig(str(p / "rero_thresholding.svg"), bbox_inches="tight")


# user.plot(
#     dict(
#         data=make_compare_img(
#             true_user_data["data"][idcs_pcpt], reconstructed[idcs_pcpt]
#         ),
#         labels=double_labels(min_dist[idcs_pcpt]).tolist(),
#     ),
#     savefile=f"cah_recon_sorted_by_pcpt",
#     stats=stats,
# )
# user.plot(
#     dict(
#         data=make_compare_img(
#             true_user_data["data"][idcs_l2], reconstructed[idcs_l2]
#         ),
#         labels=double_labels(l2min_dist[idcs_l2]).tolist(),
#     ),
#     savefile=f"cah_recon_sorted_by_l2",
#     stats=stats,
# )

# for rerothresh in [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3]:
#     reconstructed_user_data = dict(
#         data=np.where(
#             (min_dist < rerothresh)[:, None, None, None],
#             reconstructed,
#             np.zeros_like(reconstructed),
#         ),
#         labels=min_dist.tolist(),
#     )

#     user.plot(
#         reconstructed_user_data,
#         savefile=f"cah_recon_aligned@{rerothresh}.png",
#         stats=stats,
#     )

# x = np.logspace(-12, 0, num=1000)
# y = calc_rero(min_dist[:, None], x[None, :])

# x = min_dist.copy()
# x.sort()
# y = 100.0 * np.linspace(0, x.shape[0], num=x.shape[0]) / x.shape[0]

# plt.plot(x, y)
# plt.xscale("log")
# plt.savefig("rerothresholding")

# plt.clf()
# plt.scatter(min_dist, l2min_dist)
# # plt.xscale("log")
# # plt.yscale("log")
# plt.savefig("pcptvsl2")
