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
from pathlib import Path
from warnings import warn
from datetime import datetime
from tqdm import tqdm

from breaching import breaching
from breachingobjaxutils.initialize import make_configs
import lpips
import cv2
import matplotlib.pyplot as plt


def flatten_features(ftrs):
    return np.stack(
        [np.concatenate([v.flatten().numpy() for v in d.values()]) for d in ftrs],
        axis=0,
    )


def make_compare_img(true_img, recon_img):
    assert all([ts == rs for ts, rs in zip(true_img.shape, recon_img.shape)])
    show_img = np.zeros(
        (recon_img.shape[0] * 2, *recon_img.shape[1:]), dtype=recon_img.dtype
    )
    show_img[::2] = true_img
    show_img[1::2] = recon_img
    return show_img


def double_labels(labels):
    db_labels = np.zeros(2 * labels.shape[0])
    db_labels[::2] = labels
    db_labels[1::2] = labels
    return db_labels


# Redirects logs directly into the jupyter notebook
# import logging, sys
# logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
# logger = logging.getLogger()

# %% [markdown]
# ### Initialize cfg object and system setup:

# %% [markdown]
# This will load the full configuration object. This includes the configuration for the use case and threat model as `cfg.case` and the hyperparameters and implementation of the attack as `cfg.attack`. All parameters can be modified below, or overriden with `overrides=` as if they were cmd-line arguments.

# %%
cfg, train_config = make_configs(
    ["attack=imprint", "case/server=malicious-model-cah"],
    # "breaching/test_train_config.yaml",
    "configs/imagenet_dp.yaml",
    # "configs/radimagenet_dp.yaml",
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

fix_seeds(train_config)

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

make_model, make_loss_gv_from_model = make_make_fns(train_config)

train_loader, _, _ = make_loader_from_config(train_config)

if train_config.DP:
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
    with open_dict(cfg):
        cfg.case.user.total_noise = 1e-3
    print(
        f"actual epsilon: {analyse_epsilon(accountant,1,cfg.case.user.total_noise,sampling_rate,delta,train_config.DP.alphas)}"
    )

# %% [markdown]
# ### Modify config options here

# %% [markdown]
# You can use `.attribute` access to modify any of these configurations for the attack, or the case:
aug_fn = None
tf_stats = get_transform_normalization(train_config)
if tf_stats is not None:
    dm, ds = tf_stats
else:
    aug_stats = get_aug_normalization(train_config)
    if aug_stats is None:
        warn("Did not find stats. Will continue assuming unnormalized input.")
        dm, ds = np.array(0.0), np.array(1.0)
    else:
        dm, ds, aug_fn = aug_stats
example_batch, example_label = next(iter(train_loader))

new_shape = [1] * len(example_batch.shape)
new_shape[1] = ds.shape[0]
dm, ds = dm.reshape(*new_shape), ds.reshape(*new_shape)
stats = (dm, ds)

# %%
cfg.case.user.num_data_points = (
    train_config.hyperparams.batch_size
)  # How many data points does this user own
cfg.case.server.model_modification.type = (
    "CuriousAbandonHonesty"  # What type of Imprint block will be grafted to the model
)
cfg.case.server.model_modification.num_bins = (
    2 * train_config.hyperparams.batch_size * train_config.hyperparams.grad_acc_steps
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
    dataloader=(train_loader, aug_fn),
    make_loss_grad=make_loss_gv_from_model,
)
attacker = breaching.attacks.prepare_attack(
    model_fn, make_loss_gv_from_model, cfg.attack, stats
)
breaching.utils.overview(server, user, attacker)

# %% [markdown]
# ### Simulate an attacked FL protocol

# %% [markdown]
# This exchange is a simulation of a single query in a federated learning protocol. The server sends out a `server_payload` and the user computes an update based on their private local data. This user update is `shared_data` and contains, for example, the parameter gradient of the model in the simplest case. `true_user_data` is also returned by `.compute_local_updates`, but of course not forwarded to the server or attacker and only used for (our) analysis.

# %%

server_payload = server.distribute_payload()

lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)

p = Path(
    f"reconstructions/{str(train_config.dataset.name).split('.')[-1]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
)
N_batches = 5
min_dist_total = []
counter = 0
for batch in tqdm(
    range(N_batches), total=N_batches, desc="reconstructing inputs", leave=False
):
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    # %%
    # user.plot(true_user_data, savefile="cah_true_data.png", stats=stats)

    # %% [markdown]
    # ### Reconstruct user data:

    # %% [markdown]
    # Now we launch the attack, reconstructing user data based on only the `server_payload` and the `shared_data`.
    #
    # For this attack, we also share secret information from the malicious server with the attack (`server.secrets`), which here is the location and structure of the imprint block.

    # %%
    reconstructed_user_data, recon_stats = attacker.reconstruct(
        [server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun
    )

    # %% [markdown]
    # Next we'll evaluate metrics, comparing the `reconstructed_user_data` to the `true_user_data`.

    # %% [markdown]
    # ### Remove mixed images by direct GT comparison

    # %%
    # user.plot(reconstructed_user_data, savefile="cah_recon.png", stats=stats)

    # reconstructed = np.zeros_like(true_user_data["data"])
    # for sample in reconstructed_user_data["data"]:
    #     l2_dists = np.power(sample[None] - np.array(true_user_data["data"]), 2).mean(
    #         axis=(1, 2, 3)
    #     )
    #     min_dist, min_idx = np.min(l2_dists, axis=0), np.argmin(l2_dists, axis=0)
    #     # if min_dist < 1e-1:
    #     reconstructed[min_idx] = sample
    #     dists.append(min_dist)
    # for idx, sample in tqdm(
    #     enumerate(true_user_data["data"]),
    #     total=true_user_data["data"].shape[0],
    #     leave=False,
    #     desc="make aligned recon grid",
    # ):
    #     l2_dists = np.power(
    #         sample[None] - np.array(reconstructed_user_data["data"]), 2
    #     ).mean(axis=(1, 2, 3))
    #     min_dist, min_idx = np.min(l2_dists, axis=0), np.argmin(l2_dists, axis=0)
    #     # if min_dist < 1e-1:
    #     reconstructed[idx] = reconstructed_user_data["data"][min_idx]
    #     dists.append(min_dist)

    rec_denormalized = torch.clamp(
        torch.from_numpy(np.array(reconstructed_user_data["data"])).to(**setup) * ds
        + dm,
        0,
        1,
    ).float()
    imgs = rec_denormalized.numpy().transpose(0, 2, 3, 1)
    imgs = imgs - imgs.min(axis=tuple(range(1, len(imgs.shape))), keepdims=True)
    imgs /= imgs.max(axis=tuple(range(1, len(imgs.shape))), keepdims=True)
    imgs *= 255.0
    imgs = imgs[..., [2, 1, 0]]
    batch_p = p / f"{batch}/"
    if not batch_p.is_dir():
        batch_p.mkdir(parents=True)
    for img in imgs:
        cv2.imwrite(str(batch_p / f"{counter}.png"), img)
        # plt.imshow(img)
        # plt.save(f"reconstructions/{counter}.png")
        counter += 1

    denormalized_images = torch.clamp(
        torch.from_numpy(np.array(true_user_data["data"])).to(**setup) * ds + dm,
        0,
        1,
    ).float()
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
    # l2min_dist, l2min_idx = np.min(l2_dists, axis=1), np.argmin(l2_dists, axis=1)
    # reconstructed = reconstructed_user_data["data"][min_idx]

    min_dist_total.append(min_dist)

min_dist_total = np.concatenate(min_dist_total, axis=0)

calc_rero = (
    lambda distances, thresh: 100.0
    * (distances < thresh).sum(axis=0)
    / distances.shape[0]
)

print(f"ReRo@{thresh}: {calc_rero(min_dist_total, thresh):.2f}%")

# idcs_pcpt = np.argsort(min_dist)
# idcs_l2 = np.argsort(l2min_dist)

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
#         data=make_compare_img(true_user_data["data"][idcs_l2], reconstructed[idcs_l2]),
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


x = min_dist_total.copy()
x.sort()
y = 100.0 * np.linspace(0, x.shape[0], num=x.shape[0]) / x.shape[0]

plt.plot(x, y)
plt.xscale("log")
plt.savefig("rerothresholding.png")

# plt.clf()
# plt.scatter(min_dist, l2min_dist)
# # plt.xscale("log")
# # plt.yscale("log")
# plt.savefig("pcptvsl2")


# %%
# metrics = breaching.analysis.report(
#     dict(data=reconstructed, labels=None),
#     true_user_data,
#     [server_payload],
#     server.model,
#     order_batch=True,
#     compute_full_iip=False,
#     cfg_case=cfg.case,
#     setup=setup,
# )

# %% [markdown]
# And finally, we also plot the reconstructed data:

# %%

# %% [markdown]
# ### Notes:
# *
