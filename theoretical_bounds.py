# %%
import os
import seaborn as sn
import pandas as pd
import hydra
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt, colors as mplcolors, colorbar
from matplotlib.lines import Line2D
from sklearn.metrics import auc as calc_auc_score
from warnings import warn
from opacus.accountants.utils import get_noise_multiplier


os.environ["CUDA_VISIBLE_DEVICES"] = ""


sn.set_theme(
    context="notebook",
    style="white",
    font="Times New Roman",
    # font_scale=1,
    palette="viridis",
)
sn.despine()
sn.set(rc={"figure.figsize": (12, 12)}, font_scale=2)
colors = {
    "red": "firebrick",
    "blue": "steelblue",
    "green": "forestgreen",
    "purple": "darkorchid",
    "orange": "darkorange",
    "gray": "lightslategray",
    "black": "black",
}


from dptraining.config.config import Config
from dptraining.config.config_store import load_config_store
from dptraining.vulnerability.compute_rero_bound import rero_bound_sgm

load_config_store()


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(
    config: Config,
):
    # %%
    # base_config = OmegaConf.structured(Config)
    # config = OmegaConf.load("configs/imagenet_dp.yaml")
    # del config.defaults
    # config: Config = OmegaConf.merge(base_config, config)

    # %%

    key_values = []

    if "eps_values" in config.keys():
        eps_values = config.eps_values
    elif all([kwd in config.keys() for kwd in ["eps_min", "eps_max", "N_eps"]]):
        eps_values = np.linspace(
            config.eps_min,
            config.eps_max,
            config.N_eps,
            endpoint=True,
        ).tolist()
    else:
        eps_values = [0.5] + list(range(1, 21))
    if "DP" in config.keys() and "delta" in config.DP.keys():
        delta = config.DP.delta
    elif "delta" in config.keys():
        delta = config.delta
    else:
        warn("Delta should be set explicitly. Assuming delta of 1e-5")
        delta = 1e-5
    if "sampling_step" in config.keys():
        sampling_step = config.sampling_step
    else:
        sampling_step = 1e-3

    save_folder = (
        Path.cwd() / f"dp_auc/delta={delta}/eps=[{eps_values[0]}-{eps_values[-1]}]/"
    )

    # independent of exact values
    steps = 1e5
    sampling_rate = 0.1
    FPR = np.linspace(
        sampling_step,
        1 - sampling_step,
        int((1.0 - 2 * sampling_step) / sampling_step + 1),
    )
    sigmas = [
        get_noise_multiplier(
            target_epsilon=eps,
            target_delta=delta,
            sample_rate=sampling_rate,
            steps=steps,
        )
        for eps in tqdm(
            eps_values,
            total=len(eps_values),
            desc="Calculating sigmas",
            leave=False,
        )
    ]

    for sigma in tqdm(
        sigmas,
        total=len(sigmas),
        desc="Calculating ROC",
        leave=False,
    ):
        # %%
        TPR = rero_bound_sgm(FPR, 1.0 / sigma, sampling_rate, steps)
        balanced_accuracy = 1 - ((1.0 - TPR) + FPR) / 2
        max_acc, opt_cutoff = (
            balanced_accuracy.max(),
            balanced_accuracy.argmax() / balanced_accuracy.shape[0],
        )
        key_values.append(
            {
                "FPR": FPR,
                "TPR": TPR,
                "balanced_acc": balanced_accuracy,
                "max_acc": max_acc,
                "opt_cutoff": opt_cutoff,
                "auc": calc_auc_score(FPR, TPR),
            }
        )
    for eps, kv in zip(eps_values, key_values):
        print(f"Epsilon: {eps}")
        print(f"\tAUC: {kv['auc']*100.0:.2f}%")
        print(f"\tTPR@[FPR={kv['FPR'][1]:.1E}]={kv['TPR'][1]}")
        print(f"\tMaximum accuracy: {100.0*kv['max_acc']:.2f} @ {kv['opt_cutoff']:.2f}")

    if not save_folder.is_dir():
        save_folder.mkdir(parents=True)
    key_value_dict = {eps: kv for eps, kv in zip(eps_values, key_values)}
    key_value_df = pd.DataFrame.from_dict(key_value_dict)
    key_value_df.to_csv(str(save_folder / "results.csv"))
    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # cmaplist[-1] = cmap(cmap.N - 2)

    # create the new map
    cmap = mplcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    # define the bins and normalize
    bounds = eps_values
    norm = mplcolors.BoundaryNorm(
        [b - 1e-12 if i > 0 else b for i, b in enumerate(bounds)]
        + [2 * bounds[-1] - bounds[-2]],
        # np.linspace(eps_vals[0], eps_vals[-1], cmap.N),
        cmap.N,
    )

    fig_auc, ax = plt.subplots()
    fig_acc, ax4 = plt.subplots()
    fig_tpr, ax5 = plt.subplots()
    ax5.set_ylim(0, 1)
    ax5.set_xlim(0, eps_values[-1])
    ax5.plot(eps_values, [kv["TPR"][1] for kv in key_values])
    ax5.set_xlabel("epsilon")
    ax5.set_ylabel(f"TPR@(FPR={FPR[1]:.1E})")

    ax4.plot(eps_values, [kv["max_acc"] for kv in key_values])
    ax4.set_xlabel("epsilon")
    ax4.set_ylabel("maximal balanced accuracy")
    ax4.set_ylim(0, 1)
    ax4.set_xlim(0, eps_values[-1])
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("P(FPR)")
    ax.set_ylabel("P(TPR)")
    # ax2 = ax.twinx()
    # ax2.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xlim(FPR[0], 1)
    # ax.set_ylim(1 - key_values[0][1][0], 1)
    ax3 = fig_auc.add_axes([1.0, 0.1, 0.03, 0.8])
    ax3.set_ylabel(f"epsilon@[delta={delta}]")

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
    for eps, key_vals in zip(eps_values, key_values):
        FPR, TPR, balanced_accuracy, max_acc, opt_cutoff, _ = key_vals.values()
        col = cmap(norm(eps))
        # ax2.plot(FPR, balanced_accuracy, linestyle="dashed", color=col)
        FPR = FPR[1:]
        TPR = TPR[1:]
        ax.plot(FPR, TPR, color=col)
        # ax2.set_ylabel("balanced accuracy")
        # ax2.set_ylim(FPR[0], 1)
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
        # markerline, stemlines, baseline = ax2.stem(
        #     [max_acc],
        #     [opt_cutoff],
        #     orientation="horizontal",
        #     linefmt=":",
        #     markerfmt="",
        #     bottom=1.0,
        #     # color=col,
        # )
        # plt.setp(stemlines, "color", col)
        # ax2.scatter([opt_cutoff], [max_acc], marker="x", color=col)
        # legend_handles.append(Line2D([0], [0], color=col))
        # fig_auc.legend(
        #     legend_handles,
        #     ["ROC Curve", "balanced acc"],  # + [f"ε={e}" for e in eps_vals],
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, 0.02),
        # )
    fig_auc.savefig(
        str(save_folder / "ROC_curve_DP.svg"),
        bbox_inches="tight",
        dpi=600,
    )
    fig_auc.savefig(
        str(save_folder / "ROC_curve_DP.png"),
        bbox_inches="tight",
        dpi=600,
    )
    fig_acc.savefig(
        str(save_folder / "eps_acc.svg"),
        bbox_inches="tight",
        dpi=600,
    )
    fig_acc.savefig(
        str(save_folder / "eps_acc.png"),
        bbox_inches="tight",
        dpi=600,
    )
    fig_tpr.savefig(
        str(save_folder / "tpr_fail.svg"),
        bbox_inches="tight",
        dpi=600,
    )
    fig_tpr.savefig(
        str(save_folder / "tpr_fail.png"),
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
