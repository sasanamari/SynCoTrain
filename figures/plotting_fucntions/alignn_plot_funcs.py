import numpy as np
import pandas as pd
import os
import re
from jarvis.db.jsonutils import loadjson, dumpjson
import matplotlib.pyplot as plt
import sys
import argparse
# %%
# the results (res_dir_list) can be found at:
# result_dir = 'data/results'
# report['agg_df'].to_pickle(os.path.join(
#     result_dir,f'{experiment}.pkl'))
# report['resdf'].to_pickle(os.path.join(
#     result_dir,f'{experiment}_resdf.pkl'))
# %%
def plot_accuracies(dirPath, metric, plot_condition):
    train_label, val_label = None, None
    if plot_condition:
        train_label = "Train"
        val_label = "Validation"
    histT = loadjson(os.path.join(dirPath,'history_train.json' ))
    history = loadjson(os.path.join(dirPath,metric+'.json' ))
    plt.plot(histT['accuracy'], '-b', alpha = .6, label=train_label)
    plt.plot(history['accuracy'], '-r', alpha = .6, label=val_label);
    plt.ylim(.3,1.02)
    plt.xlim(None,150)
    plt.ylabel('Acuuracy', fontsize =20)
    plt.xlabel('Epochs', fontsize =15)
    plt.title("Training/Validatoin History")    
    # plt.legend()
# %%
def plot_metric(resdir, metric = 'history_val'):
    history = loadjson(os.path.join(resdir,metric+'.json' ))
    plt.plot(history['accuracy'], '-', alpha = .8);
    plt.ylabel(metric, fontsize =20)
    plt.xlabel('Epochs', fontsize =15)
# %%
def show_plot(plot_condition):
    if plot_condition:
        plt.legend()
        plt.show();
# %%
metric = 'history_val'
nruns = 10
# output_dir = 'PUOutput'+'_debug'
# output_dir = 'PUOutput'+'_fulldata_2'
output_dir = 'PUOutput'+'longlightDBug'

report = pu_report(output_dir = output_dir)

for i, PUiter in enumerate(report["res_dir_list"]):
    plot_condition = (i+1)%nruns==0
    plot_condition = True
    plot_metric(PUiter,metric=metric)
    plot_accuracies(PUiter, metric, plot_condition)
    show_plot(plot_condition)  