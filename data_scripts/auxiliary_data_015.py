# Run this script to produce the data required for ehull015 and small_data runs.
# %%
import numpy as np
import pandas as pd
import os
# %%
np.random.seed(42)
data_dir = os.path.join(os.path.dirname(__file__),'../data/clean_data/')
synthDF = pd.read_pickle(os.path.join(data_dir,'synthDF'))
stabilityDF = synthDF.copy()

experimental_df = synthDF[synthDF.synth==1]
theoretical_df = synthDF[synthDF.synth==0]
small_data_frac = 0.05
del synthDF
# %%
stabilityDF.loc[:, "stability"] = np.nan
stabilityDF.loc[:, "stability"] = np.where(stabilityDF.energy_above_hull <= 0.015, 1, 0)
stabilityDF["stability_GT"] = stabilityDF["stability"].copy()
n_unlabel = stabilityDF["stability_GT"].sum() - len(experimental_df)
# we leave the same number of positive class in the data as the synthesizability.
materials_to_unlabel = stabilityDF[stabilityDF["stability_GT"]==1].sample(int(n_unlabel)).index
stabilityDF.loc[materials_to_unlabel, "stability"] = int(0)
# we unlabel the same number of data points as the 
stabilityDF.loc[:, "stability"] = stabilityDF.stability.astype(int)
stabilityDF.loc[:, "stability_GT"] = stabilityDF.stability_GT.astype(int)
stabilityDF = stabilityDF.sample(frac=1) #to mix synthesizability values, just in case.
stabilityDF = stabilityDF.sort_values(by='stability', ascending=False).reset_index(drop=True)
stabilityDF = stabilityDF.drop(columns = 'synth')
# %%
# # Uncomment if you want to start from scratch to set the specified columns to NaN
# experiments = ['schnet0', 'alignn0', 'coSchnet1', 'coAlignn1', 'coSchnet2',
#                'coAlignn2', 'coSchnet3', 'coAlignn3']
# stabilityDF[experiments] = np.nan
# %%
stabilityDF.to_pickle(os.path.join(data_dir,'stabilityDF015'))
# %%
small_experimental_df = experimental_df.sample(frac = small_data_frac, 
                       random_state = 42, ignore_index = True)
small_theoretical_df = theoretical_df.sample(frac = small_data_frac, 
                       random_state = 42, ignore_index = True)
small_df = pd.concat([small_experimental_df, small_theoretical_df],
                     ignore_index=True)
small_df.to_pickle(os.path.join(data_dir, 'small_synthDF'))
# %%
print(f"stabilityDF was saved in {os.path.join(data_dir,'stabilityDF015')}.")