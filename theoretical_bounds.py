# %%
from autodp.transformer_zoo import Composition, AmplificationBySampling
from autodp.mechanism_zoo import GaussianMechanism
from matplotlib import pyplot as plt, colors as mplcolors, colorbar
from matplotlib.lines import Line2D
from sklearn.metrics import auc
import seaborn as sn
from copy import deepcopy

sn.set_theme(
    context="notebook",
    style="white",
    font="Times New Roman",
    font_scale=1.25,
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

from omegaconf import OmegaConf

from dptraining.privacy import setup_privacy
from dptraining.datasets import make_loader_from_config
from dptraining.config import Config

import hydra
from pathlib import Path
from dptraining.config.config_store import load_config_store

load_config_store()


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(
    train_config: Config,
):
    # %%
    # base_config = OmegaConf.structured(Config)
    # config = OmegaConf.load("configs/imagenet_dp.yaml")
    # del config.defaults
    # config: Config = OmegaConf.merge(base_config, config)

    # %%

    key_values = []
    eps_vals = [1, 4, 8, 12, 16, 20]
    N_SAMPLES = 10

    train_loader, _, _ = make_loader_from_config(train_config)
    for eps in eps_vals:
        config = deepcopy(train_config)
        config.DP.epsilon = eps
        if config.DP:
            (
                grad_acc,
                accountant,
                sampling_rate,
                delta,
                sigma,
                total_noise,
                batch_expansion_factor,
                effective_batch_size,
            ) = setup_privacy(config, train_loader)
        else:
            raise ValueError(
                "This script is only designed for DP trainings. All other are critically non-private."
            )
        # %%
        # CLIP = config.DP.max_per_sample_grad_norm
        # # NOISE_MULTI = 0.6
        # sigma = total_noise

        # BS = config.hyperparams.batch_size
        # N = len(train_loader.dataset)
        # p = BS / N
        # p = sampling_rate

        STEPS = (
            len(train_loader) // batch_expansion_factor
        ) * config.hyperparams.epochs

        DELTA = config.DP.delta

        # %%
        subsample = AmplificationBySampling(PoissonSampling=True)
        mechanism = GaussianMechanism(sigma=sigma)
        sgm = subsample(mechanism, sampling_rate, improved_bound_flag=True)

        compose = Composition()

        comp_mech = compose((sgm,), (STEPS,))

        # %%
        eps = comp_mech.get_eps(DELTA)
        print(f"ε={eps:.2f} (Calculated RDP eps = {config.DP.epsilon:.2f})")
        # %%
        FPR, FNR = comp_mech.plot_fDP(length=N_SAMPLES)

        # %%
        # %%
        print(f"AUC: {auc(FPR, 1 - FNR):.2f}")
        # %%
        balanced_accuracy = 1 - (FNR + FPR) / 2
        max_acc, opt_cutoff = (
            balanced_accuracy.max(),
            balanced_accuracy.argmax() / balanced_accuracy.shape[0],
        )
        print(f"Maximum accuracy@eps={eps:.1f}: {100.0*max_acc:.2f} @ {opt_cutoff:.2f}")
        key_values.append((FPR, FNR, balanced_accuracy, max_acc, opt_cutoff))

    cmap = plt.cm.Dark2  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[-1] = cmap(cmap.N - 2)

    # create the new map
    cmap = mplcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    # define the bins and normalize
    bounds = eps_vals
    norm = mplcolors.BoundaryNorm(
        [b - 1e-12 if i > 0 else b for i, b in enumerate(bounds)]
        + [2 * bounds[-1] - bounds[-2]],
        cmap.N,
    )
    FPR = FPR[1:]
    FNR = FNR[1:]

    fig, ax = plt.subplots()
    ax.set_xlabel("P(FPR)")
    ax.set_ylabel("P(TPR)")
    ax2 = ax.twinx()
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_xlim(FPR[0], 1)
    ax.set_ylim(1 - FNR[0], 1)
    ax3 = fig.add_axes([1.0, 0.1, 0.03, 0.8])
    ax3.set_ylabel("epsilon")

    ax.plot(FPR, FPR, linestyle="dashed", color="gray")  # ,label="Reference",)
    ax.plot(FPR, 1 - FPR, linestyle="dashed", color="gray")  # ,label="Reference",)
    cb = colorbar.ColorbarBase(
        ax3,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        ticks=bounds,
        boundaries=bounds,
        format="%1i",
    )

    legend_handles = [
        Line2D([0], [0], color="black"),
        Line2D([0], [0], color="black", linestyle="dashed"),
    ]
    for eps, (FPR, FNR, balanced_accuracy, max_acc, opt_cutoff) in zip(
        eps_vals, key_values
    ):
        col = cmap(norm(eps))
        ax2.plot(FPR, balanced_accuracy, linestyle="dashed", color=col)
        FPR = FPR[1:]
        FNR = FNR[1:]
        ax.plot(FPR, 1 - FNR, color=col)
        ax2.set_ylabel("balanced accuracy")
        ax2.set_ylim(FPR[0], 1)
        # markerline, stemlines, baseline = ax2.stem(
        #     [opt_cutoff],
        #     [max_acc],
        #     linefmt=":",
        #     orientation="vertical",
        #     markerfmt="",
        #     # bottom=0,
        #     # color=col,
        # )
        # plt.setp(stemlines, "color", col)
        markerline, stemlines, baseline = ax2.stem(
            [max_acc],
            [opt_cutoff],
            orientation="horizontal",
            linefmt=":",
            markerfmt="",
            bottom=1.0,
            # color=col,
        )
        plt.setp(stemlines, "color", col)
        ax2.scatter([opt_cutoff], [max_acc], marker="x", color=col)
        # legend_handles.append(Line2D([0], [0], color=col))
    fig.legend(
        legend_handles,
        ["ROC Curve", "balanced acc"],  # + [f"ε={e}" for e in eps_vals],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.savefig(
        f"ROC_curve_DP_{str(config.dataset.name).split('.')[-1]}.svg",
        bbox_inches="tight",
        dpi=600,
    )
    fig.savefig(
        f"ROC_curve_DP_{str(config.dataset.name).split('.')[-1]}.png",
        bbox_inches="tight",
        dpi=600,
    )
    # plt.savefig(
    #     "balanced_accuracy_DP_{config.dataset.name}_eps={config.DP.epsilon}.png"
    # )
    # plt.show()


# %%
if __name__ == "__main__":
    main()
