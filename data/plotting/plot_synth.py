# %%
import pandas as pd
import os

# import seaborn as sns
from plot_synth_funcs import (
    midlabel_dist,
    scatter_hm_final_frac,
    final_labels,
    label_dist4_frames,
)

# %%
prop = "synth"

# %%
df = pd.read_pickle("../clean_data/synthDF")
df = df[df.energy_above_hull < 3.5]
df = df[df.formation_energy_per_atom > -4.5]
# cleaninig 7 outliers for better visualization.

# %%
schnet0 = pd.read_pickle("../results/synth/schnet0.pkl")
alignn0 = pd.read_pickle("../results/synth/alignn0.pkl")

coschnet1 = pd.read_pickle("../results/synth/coSchnet1.pkl")
coalignn1 = pd.read_pickle("../results/synth/coAlignn1.pkl")

coschnet2 = pd.read_pickle("../results/synth/coSchnet2.pkl")
coalignn2 = pd.read_pickle("../results/synth/coAlignn2.pkl")

coschnet3 = pd.read_pickle("../results/synth/coSchnet3.pkl")
coalignn3 = pd.read_pickle("../results/synth/coAlignn3.pkl")

synthlab = pd.read_pickle(
    os.path.join(os.path.dirname(__file__), "../results/synth/synth_labels_2")
)
# %%

midlabel_dist(
    schnet0, df, figtitle="Iteration '0' with SchNet", filename="schnet0_prop_dist.png"
)
midlabel_dist(
    alignn0, df, figtitle="Iteration '0' with ALIGNN", filename="alignn0_prop_dist.png"
)
# )
# %%
midlabel_dist(
    coalignn1,
    df,
    figtitle="Iteration '1' with ALIGNN",
    filename="coalignn1_prop_dist.png",
)
midlabel_dist(
    coschnet1,
    df,
    figtitle="Iteration '1' with SchNet",
    filename="coschnet1_prop_dist.png",
)
# %%
midlabel_dist(
    coalignn2,
    df,
    figtitle="Iteration '2' with ALIGNN",
    filename="coalignn2_prop_dist.png",
)
midlabel_dist(
    coschnet2,
    df,
    figtitle="Iteration '2' with SchNet",
    filename="coschnet2_prop_dist.png",
)
# %%
midlabel_dist(
    coalignn3,
    df,
    figtitle="Iteration '3' with ALIGNN",
    filename="coalignn3_prop_dist.png",
)
midlabel_dist(
    coschnet3,
    df,
    figtitle="Iteration '3' with SchNet",
    filename="coschnet3_prop_dist.png",
)

# %%
scatter_hm_final_frac(synthlab, df, prop=prop, filename="final_sctter_hm_frac_it2.png")

# %%
final_labels(
    synthlab,
    figtitle="Label Distribution After Averaging",
    filename="final_label_dist_it2.png",
)

# %%
label_dist4_frames(
    synthlab, df, pred_col="synth_preds", filename="final_label_class_dist_it2.png"
)

# %%
synthlab_t25 = pd.read_pickle(
    os.path.join(
        os.path.dirname(__file__), "../results/synth/synth_labels_2_threshold_25.pkl"
    )
)
synthlab_t75 = pd.read_pickle(
    os.path.join(
        os.path.dirname(__file__), "../results/synth/synth_labels_2_threshold_75.pkl"
    )
)
# %%
final_labels(
    synthlab_t75,
    figtitle="Label Distribution with 0.75 Threshold",
    filename="final_label_dist_it2_75.png",
    threshold=0.75,
)
final_labels(
    synthlab_t25,
    figtitle="Label Distribution with 0.25 Threshold",
    filename="final_label_dist_it2_25.png",
    threshold=0.25,
)
