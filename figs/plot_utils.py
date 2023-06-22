import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_disparate_impact(
    data_frame: pd.DataFrame,
    metric_name: str = "accuracy_score",
    group_identifier: str = "sex",
    group_vals: dict = {0: "Male", 1: "Female"},
    fontsize: int = 16,
    y_shift: float = 0.0,
):
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)
    plot_df = pd.DataFrame(columns=["epsilon", metric_name, "group_identifier", "seed"])
    epsila = data_frame["epsilon"].unique()
    for eps in epsila:
        sub_df = data_frame[data_frame["epsilon"] == eps]
        for idx, row in sub_df.iterrows():
            for g_val, group_name in group_vals.items():
                tmp_dict = {
                    "epsilon": [eps],
                    metric_name: [
                        row[f"test_{group_identifier}_{g_val}.{metric_name}"]
                    ],
                    "group_identifier": [group_name],
                    "seed": [row["general.seed"]],
                }
                entry_df = pd.DataFrame.from_dict(tmp_dict)
                plot_df = pd.concat([entry_df, plot_df], ignore_index=True)
    # disparate impact plot
    y_min = plot_df[metric_name].min()
    y_max = plot_df[metric_name].max()
    y_min -= y_shift
    fig = sns.catplot(
        x="epsilon", y=metric_name, hue="group_identifier", kind="box", data=plot_df
    )
    fig._legend.remove()
    plt.xlabel(r"$\varepsilon$", fontsize=fontsize)
    if metric_name == "accuracy_score":
        metric_name = "Accuracy"
    plt.ylabel(metric_name, fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc="lower center", ncol=len(group_vals))
    plt.ylim(y_min, y_max)
    plt.show()
