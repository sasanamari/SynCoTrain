# %%
import numpy as np
import pandas as pd
import os
import re
from jarvis.db.jsonutils import loadjson, dumpjson
import matplotlib.pyplot as plt
from deepdiff import DeepDiff
from tabulate import tabulate
import pprint 
# %%
# home_dir = "/home/samariam/projects/chemheuristics"
# os.chdir("/home/samariam/projects/chemheuristics/alignn")
origdatapath = '/home/samariam/projects/synth/data/clean_data/synthDF'
home_dir = "/home/samariam/projects/synth"
alignn_dir = "/home/samariam/projects/synth/alignn"
result_dir = '/home/samariam/projects/synth/data/results'
os.chdir(alignn_dir)
synthDF = pd.read_pickle(origdatapath)

# #With a single test-set, the code always uses the new labels for co-training 
# and does not use them in prediction. We could decide to always have a random 
# portion of them in the test-set if we choose. 
# %%
def pu_report(output_dir : str = None, cotraining = False, synthDF=synthDF):
    res_df_list = []
    res_dir_list = []
    for PUiter in os.listdir(output_dir):
        resdir = os.path.join(output_dir,PUiter)
        try:   #for incomplete experiments, when the last prediction is not ready.
            res_dir_list.append(resdir)
            res = pd.read_csv(resdir+'/prediction_results_test_set.csv')
            res_df_list.append(res)
        except:
            pass
    resdf = pd.concat(res_df_list)
    resdf.reset_index(inplace=True, drop=True)
    resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])
    
    agg_df = pd.DataFrame()
    agg_df['avg_prediction'] = resdf.groupby('material_id').prediction.mean()
    agg_df['prediction'] = agg_df['avg_prediction'].map(round)
    agg_df['target'] = resdf.groupby('material_id').target.first()
    
    agg_df = agg_df.merge(synthDF[['material_id', 'synth']], on='material_id', how='left')
    positive_data = agg_df[agg_df['synth']==1] #needs to change for cotrain?
    unlabeled_data = agg_df[agg_df['synth']==0]   
       
    if output_dir.endswith("ehull"):
        true_positive_rate = "Need to compared to 'ehull_test_ref' in data dir."
        predicted_positive_rate = true_positive_rate
    else:
        true_positive_rate = positive_data['prediction'].sum()/len(positive_data)
        predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)
    
    cotrain_df = synthDF[['material_id', 'synth']].merge(
        agg_df[['material_id','prediction']], on='material_id', how='left')
    cotrain_df = cotrain_df.rename(columns={'prediction': 'new_labels'}) #for clarity
# We'll overwritre the NaNs from test-set and the co-train-set separately,
# to show that there is no unwarranted NaN value left.
    cotrain_df.loc[cotrain_df.synth == 1, 'new_labels'] = 1 #filling up the NaN used for training.
    cotrain_df.new_labels = cotrain_df.new_labels.astype(np.int16)
    
    report = {'res_dir_list':res_dir_list, 'resdf':resdf,
              'true_positive_rate':round(true_positive_rate, 4), 
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'agg_df':agg_df,
              'cotrain_df': cotrain_df} 
    return report

# %%
exp_dict = { #for publishing, just use the value names for running experiments.
    'PUOutput_100runs':'alignn0',
    'PUOutput_coAlSch1':'coAlSch1',
    'PUOutput_coAlSch2':'coAlSch2',
}
# %%
print("stop")
'PUOutput_coAlSch1'
# experiment = "PUOutput_100runs"
experiment = 'PUOutput_coAlSch1'
experiment_path =  os.path.join(alignn_dir,experiment)
report = pu_report(output_dir = experiment_path)
# %%
report['agg_df'].to_pickle(os.path.join(
    result_dir,exp_dict[experiment]+'.pkl'))
report['resdf'].to_pickle(os.path.join(
    result_dir,exp_dict[experiment]+'_resdf.pkl'))

synthDF[exp_dict[experiment]]=report['cotrain_df'].new_labels #just need the labels
synthDF.to_pickle(origdatapath)
# %%
resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
                        index_col=0)
resultcsv.loc[exp_dict[experiment], 
            'true_positive_rate'] = report['true_positive_rate']
resultcsv.loc[exp_dict[experiment], 
            'predicted_positive_rate'] = report['predicted_positive_rate']
resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))

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
# %%
# ht = loadjson(os.path.join(PUiter,'history_val.json' ))
# plt.plot(ht["rocauc"])

