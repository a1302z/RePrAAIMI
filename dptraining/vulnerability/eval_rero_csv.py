# %%
import pandas as pd
from pathlib import Path

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--csv_file",
    type=Path,
    required=True,
    help="Path to csv file with all reconstructions and recon loss",
)
parser.add_argument(
    "--threshold", type=float, required=True, help="Threshold (eta) for ReRo"
)
args = parser.parse_args()
# %%
total_df = pd.read_csv(args.csv_file)
# %%
dfs = {eps: d for eps, d in total_df.groupby(total_df.eps)}
# %%
for eps_value, df in dfs.items():
    res = (df.min_distance < args.threshold).value_counts()
    correct = res[True] if True in res else 0
    incorrect = res[False] if False in res else 0
    print(
        f"Privacy: {eps_value}\tReRo@{args.threshold}: {100.0*correct / (incorrect + correct):.2f}%"
    )

# %%
