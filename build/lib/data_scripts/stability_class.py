# %%
import numpy as np
import pandas as pd
import os
# %%
synthDF = pd.read_pickle('/data/clean_data/synthDF')

# %%
synthDF.loc[:, "stability"] = np.nan
synthDF.loc[:, "stability"] = np.where(synthDF.energy_above_hull <= 0.1, 1, 0)
synthDF.loc[:, "stability"] = synthDF.stability.astype(int)
synthDF = synthDF.sort_values(by='stability', ascending=False).reset_index(drop=True)
synthDF = synthDF.drop(columns = 'synth')

# %%
synthDF.to_pickle('/data/clean_data/stabilityDF')