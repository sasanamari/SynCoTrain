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
parser = argparse.ArgumentParser(
    description="Averaging the predictions and applying a threshold to label the data."
)
parser.add_argument(
    "--threshold", 
    type=float, 
    default=0.5, 
    help="The threshold for labeling the data.")
args = parser.parse_args(sys.argv[1:])
threshold = args.threshold
print(f'The labels are assigned using the threshold {threshold}.')
# %%
experiment = 'final_avg'
ehull015 = False
small_data = False
iteration = str(2)#choose which iteration to use
# %%
cs = current_setup(small_data=small_data, experiment=experiment, ehull015 = ehull015)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(current_dir))
data_dir = os.path.dirname(propDFpath)
# %%
coSchdf = pd.read_pickle(os.path.join(result_dir,f"coSchnet{iteration}.pkl"))
coAldf = pd.read_pickle(os.path.join(result_dir,f"coAlignn{iteration}.pkl"))
# %%
labeldf = pd.merge(coAldf[['material_id', 'avg_prediction', prop]],coSchdf[['material_id', 'predScore', prop]], on = "material_id", how = "outer")
labeldf[prop] = labeldf[f"{prop}_x"].fillna(labeldf[f"{prop}_y"])
labeldf = labeldf.drop(columns=[f"{prop}_x",f"{prop}_y"])
# %%
def classify_by_threshold(x, threshold):
# This also takes care of the boundaries, when threshold is exactly 0 or 1.
    if threshold == 0:
        if x == 0:
            return 0
        else:
            return 1
    elif x < threshold:
        return 0
    else:
        return 1
# %%
labeldf[f"{prop}_avg"]=labeldf[['avg_prediction','predScore']].apply(lambda x: np.nanmean(x), axis=1)
labeldf[f"{prop}_preds"]=labeldf[f"{prop}_avg"].map(lambda x: classify_by_threshold(x, threshold))
labeldf[f"{prop}_labels"]=labeldf.apply(lambda x: 1 if x[prop] == 1 else x[f"{prop}_preds"], axis=1)
# the last line corrects for the false negative data, to have healthy labels for training a 
# synthesizability predictor (all the experimental data will be labeled as class 1.)

# %%
crysdf = pd.read_pickle(propDFpath)
labeldf = labeldf.merge(crysdf[["material_id", "atoms"]], on="material_id", how='inner') 

# %%
# print(f"{prop}_labels_{iteration}_threshold_{str(threshold).replace('.', '_')}")
# print(labeldf.tail())
labeldf.to_pickle(os.path.join(result_dir,f"{prop}_labels_{iteration}_threshold_{str(threshold).replace('.', '_')}"))
# %%
