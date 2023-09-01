# Run this script to produce the data required for ehull_test and small_data runs.
# %%
import numpy as np
import pandas as pd
import os
# %%
data_dir = 'data/clean_data/'
synthDF = pd.read_pickle(os.path.join(data_dir,'synthDF'))
stabilityDF = synthDF.copy()

experimental_df = synthDF[synthDF.synth==1]
theoretical_df = synthDF[synthDF.synth==0]
small_data_frac = 0.05
del synthDF
# %%
stabilityDF.loc[:, "stability"] = np.nan
stabilityDF.loc[:, "stability"] = np.where(stabilityDF.energy_above_hull <= 0.1, 1, 0)
stabilityDF.loc[:, "stability"] = stabilityDF.stability.astype(int)
stabilityDF = stabilityDF.sort_values(by='stability', ascending=False).reset_index(drop=True)
stabilityDF = stabilityDF.drop(columns = 'synth')

# %%
stabilityDF.to_pickle(os.path.join(data_dir,'stabilityDF'))
# %%
small_experimental_df = experimental_df.sample(frac = small_data_frac, 
                       random_state = 42, ignore_index = True)
small_theoretical_df = theoretical_df.sample(frac = small_data_frac, 
                       random_state = 42, ignore_index = True)
small_df = pd.concat([small_experimental_df, small_theoretical_df],
                     ignore_index=True)
small_df.to_pickle(os.path.join(data_dir, 'small_synthDF'))