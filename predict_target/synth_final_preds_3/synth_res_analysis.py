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
experiment = 'final_avg'
ehull015 = False
small_data = False
cs = current_setup(small_data=small_data, experiment=experiment, ehull015 = ehull015)
propDFpath = cs["propDFpath"]
# %%
current_dir = os.path.dirname(os.path.abspath(__file__))
resfile = 'prediction_results_test_set.csv'
respath = os.path.join(current_dir, resfile)
resdf = pd.read_csv(respath)
# %%
resdf.reset_index(inplace=True, drop=True)
resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])
# %%
# propdf = pd.read_pickle("/home/samariam/projects/SynthCoTrain/data/clean_data/synthDF")
project_root = os.path.dirname(os.path.dirname(current_dir))
propdf = pd.read_pickle(os.path.join(project_root, propDFpath))
# change this to dynamic address!
# %%
finaldf= pd.merge(propdf[["material_id", "synth"]], resdf, on="material_id", how="inner")
# %%
edf = finaldf[finaldf.synth==1]
tdf = finaldf[finaldf.synth==0]
# %%
print(f"Recall for {len(edf)} test data points was {edf.prediction.mean()*100:.2f}%.")
# %%
print(f"{tdf.prediction.mean()*100:.2f}% of the {len(tdf)} theoretical test data were predicted to be synthesizable.")
# %%
# finaldf.to_pickle("~/projects/SynthCoTrain/predict_target/final_df")
# finaldf.to_pickle(os.path.join(os.path.dirname(current_dir), 'final_df'))
# %%
