# %%
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import sys
import argparse
from experiment_setup import current_setup, str_to_bool
import warnings
# %%
resdf = pd.read_csv('prediction_results_test_set.csv')
# %%
resdf.reset_index(inplace=True, drop=True)
resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])
# %%
propdf = pd.read_pickle("/home/samariam/projects/SynthCoTrain/data/clean_data/synthDF")
# %%
finaldf= pd.merge(propdf[["material_id", "synth"]], resdf, on="material_id", how="inner")
# %%
edf = finaldf[finaldf.synth==1]
tdf = finaldf[finaldf.synth==0]
# %%
edf.prediction.mean()
# %%
tdf.prediction.mean()
# %%
# finaldf.to_pickle("/home/samariam/projects/SynthCoTrain/predict_target/final_df")
# %%
