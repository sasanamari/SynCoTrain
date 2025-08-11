# %%
import pandas as pd

from plot_cotrain_recall_funcs import (
    synth_y_margin,
    plot_recall_synth,
    plot_recall_stability,
    df_by_series,
    stability_y_margin,
)

# %%
df = pd.read_csv("../results/synth/results.csv")
df015 = pd.read_csv("../results/stability015/results.csv")
df = df.dropna()
df.reset_index(drop=True, inplace=True)
df["exper"].replace(
    {
        "schnet0": "SchNet0",
        "alignn0": "Alignn0",
        "coSchnet1": "coSchNet1",
        "coSchnet2": "coSchNet2",
        "coSchnet3": "coSchNet3",
    },
    inplace=True,
)
df015 = df015.dropna()
df015.reset_index(drop=True, inplace=True)
df015["exper"].replace(
    {
        "schnet0": "SchNet0",
        "alignn0": "Alignn0",
        "coSchnet1": "coSchNet1",
        "coSchnet2": "coSchNet2",
        "coSchnet3": "coSchNet3",
    },
    inplace=True,
)

# %%
plot_recall_synth(
    df_by_series(df)[0],
    y_lims=synth_y_margin(df),
    title="SchNet0 Series Synthesizability Prediction Rates",
)
# %%
plot_recall_synth(
    df_by_series(df)[1],
    y_lims=synth_y_margin(df),
    title="ALIGNN0 Series Synthesizability Prediction Rates",
)
# %%
plot_recall_stability(
    df_by_series(df015)[0],
    y_lims=stability_y_margin(df015),
    title="SchNet0 Series Stability Recall",
)
# %%
plot_recall_stability(
    df_by_series(df015)[1],
    y_lims=stability_y_margin(df015),
    title="ALIGNN0 Series Stability Recall",
)

# %%
