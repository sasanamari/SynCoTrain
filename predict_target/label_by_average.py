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
# parser = argparse.ArgumentParser(
#     description="Semi-Supervised ML for Synthesizability Prediction"
# )
# parser.add_argument(
#     "--experiment",
#     default="alignn0",
#     help="name of the experiment and corresponding config files.",
# )
# parser.add_argument(
#     "--ehull",
#     type=str_to_bool,
#     default=False,
#     help="Predicting stability to evaluate PU Learning's efficacy.",
# )
# parser.add_argument(
#     "--ehull015",
#     type=str_to_bool,
#     default=False,
#     help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
# )
# parser.add_argument(
#     "--small_data",
#     type=str_to_bool,
#     default=False,
#     help="This option selects a small subset of data for checking the workflow faster.",
# )
# # %%
# args = parser.parse_args(sys.argv[1:])
# experiment = args.experiment 
# ehull015 = args.ehull015
# ehull_test = args.ehull
# small_data = args.small_data
# %%
experiment = 'final_avg'
ehull015 = False
ehull_test = False
small_data = False
# %%
cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment, ehull015 = ehull015)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(current_dir))
data_dir = os.path.dirname(propDFpath)
# %%
coSchdf = pd.read_pickle(os.path.join(result_dir,"coSchAl3.pkl"))
coAldf = pd.read_pickle(os.path.join(result_dir,"coAlSch3.pkl"))
# %%
labeldf = pd.merge(coAldf[['material_id', 'avg_prediction', prop]],coSchdf[['material_id', 'predScore', prop]], on = "material_id", how = "outer")
labeldf[prop] = labeldf[f"{prop}_x"].fillna(labeldf[f"{prop}_y"])
labeldf = labeldf.drop(columns=[f"{prop}_x",f"{prop}_y"])
# %%
labeldf[f"{prop}_avg"]=labeldf[['avg_prediction','predScore']].apply(lambda x: np.nanmean(x), axis=1)
labeldf[f"{prop}_preds"]=labeldf[f"{prop}_avg"].map(lambda x: 0 if x<0.5 else 1)
labeldf[f"{prop}_labels"]=labeldf.apply(lambda x: 1 if x[prop] == 1 else x[f"{prop}_preds"], axis=1)
# %%
# tdf = labeldf[labeldf[prop]==0]
# edf = labeldf[labeldf[prop]==1]
# # %%
# edf[f"{prop}_labels"].hist()
# # %%
# tdf[f"{prop}_labels"].hist()
# # %%
# edf[f"{prop}_avg"].hist()
# # %%
# tdf[f"{prop}_avg"].hist()
# %%
crysdf = pd.read_pickle(propDFpath)
labeldf = labeldf.merge(crysdf[["material_id", "atoms"]], on="material_id", how='inner') 

# %%
labeldf.to_pickle(os.path.join(result_dir,f"{prop}_labels"))
# %%
