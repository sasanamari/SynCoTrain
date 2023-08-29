# only updated the top  of the file, not inside the function yet.
# %%
import numpy as np
import pandas as pd
import os
import re
from jarvis.db.jsonutils import loadjson, dumpjson
import matplotlib.pyplot as plt
import sys
import argparse
from experiment_setup import current_setup
# %%
parser = argparse.ArgumentParser(
    description="PU-ALIGNN result analysis"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)
parser.add_argument(
    "--ehull",
    default=False,
    help="Predicting stability to evaluate PU Learning's efficacy.",
)
parser.add_argument(
    "--small_data",
    default=False,
    help="This option selects a small subset of data for checking the workflow faster.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment 
ehull_test = args.ehull
small_data = args.small_data
# %%
cs = current_setup(ehull_test=ehull_test, small_data=small_data)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]

# if small_data:
#     propDFpath = '/data/clean_data/small_synthDF'
#     result_dir = 'data/results/small_data_synth'
# elif ehull_test:
#     propDFpath = '/data/clean_data/stabilityDF' 
#     result_dir = 'data/results/stability'
# else:
#     propDFpath = '/data/clean_data/synthDF'
#     result_dir = 'data/results/synth'
        
alignn_dir = "alignn"

os.chdir(alignn_dir)

# %%
def pu_report_alignn(experiment: str = None, prop: str = None,
              propDFpath='data/clean_data/synthDF',
              pseudo_label_threshold = 0.75,ehull_test = False,
              small_data = False, ):
    data_prefix = "small_" if small_data else ""
    
    output_dir = os.path.join(alignn_dir, f'PUOutput_{data_prefix}{experiment}')
    if ehull_test:
        output_dir = os.path.join(alignn_dir, f'PUehull_{experiment}')
    propDF = pd.read_pickle(propDFpath)
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
    agg_df['new_labels'] = agg_df['avg_prediction'].map(lambda x: 1 if x >= pseudo_label_threshold else 0)
    agg_df['target'] = resdf.groupby('material_id').target.first()
    
    agg_df = agg_df.merge(propDF[['material_id', prop]], on='material_id', how='left')
    positive_data = agg_df[agg_df[prop]==1] #needs to change for cotrain?
    unlabeled_data = agg_df[agg_df[prop]==0]   
       
    true_positive_rate = positive_data['prediction'].sum()/len(positive_data)
    predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)
    
    cotrain_df = propDF[['material_id', prop]].merge(
        agg_df[['material_id','new_labels']], on='material_id', how='left')
        # agg_df[['material_id','prediction']], on='material_id', how='left')
    # cotrain_df = cotrain_df.rename(columns={'prediction': 'new_labels'}) #for clarity
    label_source = { #output_dir: training_label_column
            'alignn0':prop,
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
            'coAlSch3':'coSchAl2',
            'coAlSch4':'coSchAl3',
            'coAlSch5':'coSchAl4',
    }
    cotrain_index = propDF[propDF[prop]!=propDF[label_source[
        output_dir.split('_')[-1]]]].index 
    cotrain_df.loc[cotrain_index, 'new_labels'] = 1 #used in cotraining, not predicted. does nothing at step 0.
    cotrain_df.loc[cotrain_df[prop] == 1, 'new_labels'] = 1 #filling up the NaN used for training.
    if output_dir.endswith('small'):
        cotrain_df = cotrain_df.dropna() #in small data set-up, there will be NaN values for unsed data.
    cotrain_df.new_labels = cotrain_df.new_labels.astype(np.int16)
       
    report = {'res_dir_list':res_dir_list, 'resdf':resdf,
              'true_positive_rate':round(true_positive_rate, 4), 
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'agg_df':agg_df,
              'cotrain_df': cotrain_df} 
    
    return report, propDF

# %%
report, propDF = pu_report_alignn(output_dir = experiment, prop = prop,
                            propDFpath=propDFpath, ehull_test = ehull_test,
                            small_data=small_data)
print(f"The True positive rate was {report['true_positive_rate']} and the "
      f"predicted positive rate was {report['predicted_positive_rate']}.")

# %%    
report['agg_df'].to_pickle(os.path.join(
    result_dir,f'{experiment}.pkl'))
report['resdf'].to_pickle(os.path.join(
    result_dir,f'{experiment}_resdf.pkl'))
# %%
propDF[experiment]=report['cotrain_df'].new_labels #just need the labels
# %%
propDF.to_pickle(propDFpath)
# %%
resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
                        index_col=0)
new_rates = {'true_positive_rate':report['true_positive_rate'],
     'predicted_positive_rate':report['predicted_positive_rate']}
resultcsv.loc[experiment.split('PUOutput_')[-1]] = new_rates

# %%
resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))
       
# %%
# ht = loadjson(os.path.join(PUiter,'history_val.json' ))
# plt.plot(ht["rocauc"])



# # %%
# import numpy as np
# import pandas as pd
# import os
# import re
# from jarvis.db.jsonutils import loadjson, dumpjson
# import matplotlib.pyplot as plt
# # from deepdiff import DeepDiff
# from tabulate import tabulate
# import pprint 
# # %%
# origdatapath = 'data/clean_data/synthDF'
# alignn_dir = "alignn/"
# result_dir = 'data/results'
# os.chdir(alignn_dir)
# synthDF = pd.read_pickle(origdatapath)

# # #With a single test-set, the code always uses the new labels for co-training 
# # and does not use them in prediction. We could decide to always have a random 
# # portion of them in the test-set if we choose. 
# # %%
# def pu_report(output_dir : str = None, cotraining = False, synthDF=synthDF):
#     res_df_list = []
#     res_dir_list = []
#     for PUiter in os.listdir(output_dir):
#         resdir = os.path.join(output_dir,PUiter)
#         try:   #for incomplete experiments, when the last prediction is not ready.
#             res_dir_list.append(resdir)
#             res = pd.read_csv(resdir+'/prediction_results_test_set.csv')
#             res_df_list.append(res)
#         except:
#             pass
#     resdf = pd.concat(res_df_list)
#     resdf.reset_index(inplace=True, drop=True)
#     resdf['material_id'] = resdf['id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])
    
#     agg_df = pd.DataFrame()
#     agg_df['avg_prediction'] = resdf.groupby('material_id').prediction.mean()
#     agg_df['prediction'] = agg_df['avg_prediction'].map(round)
#     agg_df['target'] = resdf.groupby('material_id').target.first()
    
#     agg_df = agg_df.merge(synthDF[['material_id', 'synth']], on='material_id', how='left')
#     positive_data = agg_df[agg_df['synth']==1] #needs to change for cotrain?
#     unlabeled_data = agg_df[agg_df['synth']==0]   
       
#     if output_dir.endswith("ehull"):
#         true_positive_rate = "Need to compared to 'ehull_test_ref' in data dir."
#         predicted_positive_rate = true_positive_rate
#     else:
#         true_positive_rate = positive_data['prediction'].sum()/len(positive_data)
#         predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)
    
#     cotrain_df = synthDF[['material_id', 'synth']].merge(
#         agg_df[['material_id','prediction']], on='material_id', how='left')
#     cotrain_df = cotrain_df.rename(columns={'prediction': 'new_labels'}) #for clarity
# # We'll overwritre the NaNs from test-set and the co-train-set separately,
# # to show that there is no unwarranted NaN value left.
#     experiment_train_match = { #output_dir: training_label_column
#             'alignn0':'synth',
#             'coAlSch1':'schnet0',
#             'coAlSch2':'coSchAl1',
#     }
#     cotrain_index = synthDF[synthDF.synth!=synthDF[experiment_train_match[
#         output_dir.split('_')[-1]]]].index
#     cotrain_df.loc[cotrain_index, 'new_labels'] = 1 #used in training
#     cotrain_df.loc[cotrain_df.synth == 1, 'new_labels'] = 1 #filling up the NaN used for training.
#     cotrain_df.new_labels = cotrain_df.new_labels.astype(np.int16)
    
    
    
#     report = {'res_dir_list':res_dir_list, 'resdf':resdf,
#               'true_positive_rate':round(true_positive_rate, 4), 
#               'predicted_positive_rate': round(predicted_positive_rate, 4),
#               'agg_df':agg_df,
#               'cotrain_df': cotrain_df} 
#     return report

# # %%
# exp_dict = { #for publishing, just use the value names for running experiments.
#     'PUOutput_100runs':'alignn0',
#     'PUOutput_coAlSch1':'coAlSch1',
#     'PUOutput_coAlSch2':'coAlSch2',
# }
# # %%
# print("stop")
# # 'PUOutput_coAlSch1'
# # experiment = "PUOutput_100runs"
# experiment = 'PUOutput_coAlSch1'
# # experiment = 'PUOutput_coAlSch2'
# experiment_path =  os.path.join(alignn_dir,experiment)
# # %%
# report = pu_report(output_dir = experiment_path)
# # %%
# report['agg_df'].to_pickle(os.path.join(
#     result_dir,exp_dict[experiment]+'.pkl'))
# report['resdf'].to_pickle(os.path.join(
#     result_dir,exp_dict[experiment]+'_resdf.pkl'))

# synthDF[exp_dict[experiment]]=report['cotrain_df'].new_labels #just need the labels
# synthDF.to_pickle(origdatapath)
# # %%
# resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
#                         index_col=0)
# resultcsv.loc[exp_dict[experiment], 
#             'true_positive_rate'] = report['true_positive_rate']
# resultcsv.loc[exp_dict[experiment], 
#             'predicted_positive_rate'] = report['predicted_positive_rate']
# resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))

# # %%
# def plot_accuracies(dirPath, metric, plot_condition):
#     train_label, val_label = None, None
#     if plot_condition:
#         train_label = "Train"
#         val_label = "Validation"
#     histT = loadjson(os.path.join(dirPath,'history_train.json' ))
#     history = loadjson(os.path.join(dirPath,metric+'.json' ))
#     plt.plot(histT['accuracy'], '-b', alpha = .6, label=train_label)
#     plt.plot(history['accuracy'], '-r', alpha = .6, label=val_label);
#     plt.ylim(.3,1.02)
#     plt.xlim(None,150)
#     plt.ylabel('Acuuracy', fontsize =20)
#     plt.xlabel('Epochs', fontsize =15)
#     plt.title("Training/Validatoin History")    
#     # plt.legend()
# # %%
# def plot_metric(resdir, metric = 'history_val'):
#     history = loadjson(os.path.join(resdir,metric+'.json' ))
#     plt.plot(history['accuracy'], '-', alpha = .8);
#     plt.ylabel(metric, fontsize =20)
#     plt.xlabel('Epochs', fontsize =15)
# # %%
# def show_plot(plot_condition):
#     if plot_condition:
#         plt.legend()
#         plt.show();
# # %%
# metric = 'history_val'
# nruns = 10
# # output_dir = 'PUOutput'+'_debug'
# # output_dir = 'PUOutput'+'_fulldata_2'
# output_dir = 'PUOutput'+'longlightDBug'

# report = pu_report(output_dir = output_dir)

# for i, PUiter in enumerate(report["res_dir_list"]):
#     plot_condition = (i+1)%nruns==0
#     plot_condition = True
#     plot_metric(PUiter,metric=metric)
#     plot_accuracies(PUiter, metric, plot_condition)
#     show_plot(plot_condition)         
# # %%
# # ht = loadjson(os.path.join(PUiter,'history_val.json' ))
# # plt.plot(ht["rocauc"])

