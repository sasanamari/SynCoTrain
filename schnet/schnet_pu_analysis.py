# %%
import os
import json
from pathlib import Path  #recommended path library for python3
import pandas as pd
import numpy as np
import sys
import argparse
from experiment_setup import current_setup
# %%
parser = argparse.ArgumentParser(
    description="PU-Schnet result analysis"
)
parser.add_argument(
    "--experiment",
    default="schnet0",
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
# small_data = False
# experiment = 'coSchAl1'
# ehull_test = False

cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]

# if small_data:
#     propDFpath = '/data/clean_data/small_synthDF'
#     result_dir = 'data/results/small_data_synth'
# elif ehull_test:
#     propDFpath = '/data/clean_data/stabilityDF' 
#     result_dir = 'data/results/stability'
# else:
#     propDFpath = '/data/clean_data/synthDF'
#     result_dir = 'data/results/synth'
need to check on new_target vs TARGET; the rest might be cleaned already.
def pu_report_schnet(experiment : str = None,
              propDFpath='data/clean_data/synthDF', 
              pseudo_label_threshold = 0.75,
              ehull_test = False, small_data = False):
    if small_data and ehull_test:
        error_message = "small_data and ehull_test are not allowed at the same time."
        raise Exception(error_message)
        sys.exit()

    data_prefix = "small_" if small_data else ""
    
    def scoreFunc(x):
        trial_num = sum(x.notna())
        if trial_num == 0:
            return np.nan, trial_num
        res = x.sum()
        score = res/trial_num
        return score, trial_num
    
    
    schnet_config_dir = "schnet/schnet_configs"
    config_path = os.path.join(schnet_config_dir, f'pu_config_{experiment}.json')
    with open(config_path, "r") as read_file:
        print("Read Experiment configuration")
        config = json.load(read_file)

    schnetDirectory = config["schnetDirectory"]
    if not os.getcwd().endswith(schnetDirectory):
        os.chdir(schnetDirectory)
    print(os.getcwd())
    print(Path().resolve())  #current working directory
    print(Path().absolute()) #file path in jupyter

    # cotraining = config["cotraining"]
    new_target = config["new_target"]
    epoch_num = config["epoch_num"]
    start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
    num_iter = config["num_iter"]
    # batch_size = config["batch_size"]
    experiment = config["experiment"]
    if small_data:
        epoch_num = int(epoch_num*0.5)
                
    # res_df_fileName = data_prefix+experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)
    res_df_fileName = f'{data_prefix}{experiment}_{str(start_iter)}_{str(num_iter)}ep{str(epoch_num)}'
    save_dir = os.path.join(schnetDirectory,f'{experiment}_res_log')
    # data_dir = config["data_dir"]
    # data_dir = 'data/clean_data'
    
    half_way_analysis = False
    # half_way_analysis = True
    if half_way_analysis:
        crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',f'{res_df_fileName}tmp'))   #saving results at each iteration
    else:
        crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',res_df_fileName))
   
    pred_columns = []
    score_columns = []
    if half_way_analysis:
        iterations_so_far = [int(col[-2:].translate(str.maketrans('', '', '_'))) for col in crysdf.columns if col.startswith("pred")]
        # The above returns the number of iterations already carried out.
        num_iter = max(iterations_so_far)+1 #for the exclusive looping below
    for it in range(0, num_iter):  #always start at 0 because we want to average prediction over all the iterations.
        pred_col_name = 'pred_'+str(it)
        pred_columns.append(pred_col_name)
        
        score_col_name = 'pred_score'+str(it)
        score_columns.append(score_col_name)

    Preds = crysdf[pred_columns]
    crysdf['predScore'] = Preds.apply(scoreFunc, axis=1)
    crysdf[['predScore', 'trial_num']] = crysdf.predScore.tolist()
    crysdf["prediction"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else round(x))
    crysdf["new_labels"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else 1 if x >= pseudo_label_threshold else 0)

    res_df = crysdf[crysdf.predScore.notna()][[
        'material_id',prop, 'TARGET','prediction', 'predScore', 'trial_num']]  #selecting data with prediction values

    experimental_df = res_df[res_df[prop]==1]
    theoretical_df = res_df[res_df[prop]==0]

    true_positive_rate = sum(experimental_df.prediction)/experimental_df.shape[0]
    predicted_positive_rate = sum(theoretical_df.prediction)/theoretical_df.shape[0]
    
    print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
    print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))    
    
    crysdf = crysdf.drop(columns=Preds)
    
    propDF = pd.read_pickle(propDFpath)
    
    merged_df = propDF[['material_id', [prop]]].merge(
        theoretical_df[['material_id', 'new_labels']], on='material_id', how='left')
    # merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity
    # if new_target:
    #     # new_target2 = f"{config['new_target']}_{test_strategy}"
    #     cotrain_index = propDF[(propDF[prop] != propDF[new_target]) &
    #     pd.notna(propDF[new_target])].index
    #     merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training
    # else:
    #     pass
# The condition above seems unnecessary with separate dfs, and wrong for checking pd.notna.
# The data used in training will have nan values. We simplify below and check in practice.
    cotrain_index = propDF[(propDF[prop] != propDF[new_target])].index
    merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training, , not predicted. empty index at step 0.
  
    merged_df.loc[merged_df[prop] == 1, 'new_labels'] = 1 #used in training
    # changed below, only for small data.
    # merged_df = merged_df.dropna()
    merged_df.new_labels = merged_df.new_labels.astype(pd.Int16Dtype())
    
    propDF[experiment] = merged_df.new_labels
    
    report = {'resdf':Preds, 'agg_df':crysdf, 
              'true_positive_rate':round(true_positive_rate, 4), 
              'predicted_positive_rate': round(predicted_positive_rate, 4)}
    return report, propDF
# %%
report, propDF = pu_report_schnet(experiment = experiment,
              propDFpath=propDFpath, ehull_test = ehull_test, small_data = small_data)
report['agg_df'].to_pickle(os.path.join( #goes one directory up from schnet to main dir.
    f'../{result_dir}',f'{experiment}.pkl'))
report['resdf'].to_pickle(os.path.join( 
    f'../{result_dir}',f'{experiment}_resdf.pkl'))
# %%
propDF.to_pickle(propDFpath)
# %%
csv_path =os.path.join(f'../{result_dir}', 'results.csv')
resultcsv = pd.read_csv(csv_path,
                        index_col=0)
new_rates = {'true_positive_rate':report['true_positive_rate'],
     'predicted_positive_rate':report['predicted_positive_rate']}
resultcsv.loc[experiment] = new_rates
# %%
resultcsv.to_csv(csv_path)
# %%
"""
schnet_config_dir = "schnet/schnet_configs"
config_path = os.path.join(schnet_config_dir, 'pu_config_'+experiment+'.json')
with open(config_path, "r") as read_file:
    print("Read Experiment configuration")
    config = json.load(read_file)

    
schnetDirectory = config["schnetDirectory"]
if not os.getcwd().endswith(schnetDirectory):
    os.chdir(schnetDirectory)
print(os.getcwd())
# tensorboard command is tensorboard --logdir={experiment}_res_log/lightning_logs/
# %%
print(Path().resolve())  #current working directory
print(Path().absolute()) #file path in jupyter
# print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# time.sleep(600)
# %%
# ###quick settings
cotraining = config["cotraining"]
new_target = config["new_target"]
epoch_num = config["epoch_num"]
start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
num_iter = config["num_iter"]
batch_size = config["batch_size"]
experiment = config["experiment"]
if small_data:
    epoch_num = int(epoch_num*0.5)
    
if small_data:
    propDFpath = '/data/clean_data/small_synthDF'
    result_dir = 'data/results/small_data_synth'
    # prop = "synth" can add this in the final function.
elif ehull_test:
    propDFpath = '/data/clean_data/stabilityDF' 
    result_dir = 'data/results/stability'
    prop = "stability"
else:
    propDFpath = '/data/clean_data/synthDF'
    result_dir = 'data/results/synth'
    prop = "synth"
        
res_df_fileName = data_prefix+experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)

save_dir = os.path.join(schnetDirectory,f'{experiment}_res_log')

data_dir = config["data_dir"]
# data_dir = 'data/clean_data'
# %%
half_way_analysis = False
# half_way_analysis = True
if half_way_analysis:
    crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',f'{res_df_fileName}tmp'))   #saving results at each iteration
else:
    crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',res_df_fileName))
# %%
pred_columns = []
score_columns = []
if half_way_analysis:
    iterations_so_far = [int(col[-2:].translate(str.maketrans('', '', '_'))) for col in crysdf.columns if col.startswith("pred")]
    # The above returns the number of iterations already carried out.
    num_iter = max(iterations_so_far)+1 #for the exclusive looping below
for it in range(0, num_iter):  #always start at 0 because we want to average prediction over all the iterations.
    pred_col_name = 'pred_'+str(it)
    pred_columns.append(pred_col_name)
    
    score_col_name = 'pred_score'+str(it)
    score_columns.append(score_col_name)


Preds = crysdf[pred_columns]



# %%
def scoreFunc(x):
    trial_num = sum(x.notna())
    if trial_num == 0:
        return np.nan, trial_num
    res = x.sum()
    score = res/trial_num
    return score, trial_num

# %%
crysdf['predScore'] = Preds.apply(scoreFunc, axis=1)
# %%
crysdf[['predScore', 'trial_num']] = crysdf.predScore.tolist()
# %%
crysdf["prediction"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else round(x))
# crysdf["prediction_0.75"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else 1 if x >= 0.75 else 0)

# %%
res_df = crysdf[crysdf.predScore.notna()][[
    'material_id',prop, 'TARGET','prediction', 'predScore', 'trial_num']]  #selecting data with prediction values

# %%
experimental_df = res_df[res_df[prop]==1]
theoretical_df = res_df[res_df[prop]==0]
# for co-training we need more analysis, but the experimental vs theoretical distinction is more important.

# %%
true_positive_rate = sum(experimental_df.prediction)/experimental_df.shape[0]
predicted_positive_rate = sum(theoretical_df.prediction)/theoretical_df.shape[0]
# %%
print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))

# %%
crysdf = crysdf.drop(columns=Preds)
# %%
crysdf.to_pickle(os.path.join( #goes one directory up from schnet to main dir.
    f'../{result_dir}',f'{experiment}_.pkl'))
# %%
propDF = pd.read_pickle(propDFpath)
# %%
merged_df = propDF[['material_id', [prop]]].merge(
    theoretical_df[['material_id', 'prediction']], on='material_id', how='left')
merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity
if config['new_target']:
    # new_target2 = f"{config['new_target']}_{test_strategy}"
    cotrain_index = propDF[(propDF[prop] != propDF[config['new_target']]) &
    pd.notna(propDF[config['new_target']])].index
    merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training
else:
    pass
merged_df.loc[merged_df[prop] == 1, 'new_labels'] = 1 #used in training
# changed below, only for small data.
# merged_df = merged_df.dropna()
merged_df.new_labels = merged_df.new_labels.astype(pd.Int16Dtype())
# %%
# synthDF[exp_dict[experiment]] = merged_df.new_labels
propDF[experiment] = merged_df.new_labels
# %%
propDF.to_pickle(propDFpath)
# %%
csv_path =os.path.join(f'../{result_dir}', 'results.csv')
resultcsv = pd.read_csv(csv_path,
                        index_col=0)
new_rates = {'true_positive_rate':true_positive_rate,
     'predicted_positive_rate':predicted_positive_rate}
resultcsv.loc[experiment] = new_rates
# %%
resultcsv.to_csv(csv_path)
# %%
"""
# %%
# # %%
# import os
# import json
# from pathlib import Path  #recommended path library for python3
# import pandas as pd
# import numpy as np

# # current_config = "25runTest_config.json"
# # current_config = "debugsch_config.json"
# # current_config = "longDebug_config.json"
# # current_config = "100runs_config.json"
# # current_config = "cotrain_debug_config.json"
# current_config = "coSchAl1_config.json"

# result_dir = 'data/results'

# config_dir = "schnet/schnet_configs/"
# with open(os.path.join(config_dir, current_config), "r") as read_file:
#     print("Read Experiment configuration")
#     config = json.load(read_file)
    
# schnetDirectory = config["schnetDirectory"]
# os.chdir(schnetDirectory)
# print(os.getcwd())
# # tensorboard command is tensorboard --logdir={experiment}_res_log/lightning_logs/
# # %%
# print(Path().resolve())  #current working directory
# print(Path().absolute()) #file path in jupyter
# # print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# # time.sleep(600)
# # %%
# # ###quick settings
# cotraining = config["cotraining"]
# new_target = config["new_target"]
# epoch_num = config["epoch_num"]
# start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
# num_iter = config["num_iter"]
# batch_size = config["batch_size"]
# experiment = config["experiment"]
# res_df_fileName = experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)


# save_dir = os.path.join(schnetDirectory,experiment+'_res_log')
# fulldatapath = config["fulldatapath"]
# # exp_dict = { #for publishing, just use the value names for running experiments.
# #     '100runs':'schnet0',
# #     'coSchAl1':'coSchAl1',
# #     'coSchAl2':'coSchAl2',
# # }
# # %%
# half_way_analysis = False
# # half_way_analysis = True
# if half_way_analysis:
#     crysdf=pd.read_pickle(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration
# else:
#     crysdf=pd.read_pickle(save_dir+'/res_df/'+res_df_fileName)
# # I chanegd crysdf to it_test_df!!!
# # from here on is the analysis, I should add to a separate file.
# # %%
# pred_columns = []
# score_columns = []
# if half_way_analysis:
#     iterations_so_far = [int(col[-2:].translate(str.maketrans('', '', '_'))) for col in crysdf.columns if col.startswith("pred")]
#     # The above returns the number of iterations already carried out.
#     num_iter = max(iterations_so_far)+1 #for the exclusive looping below
# for it in range(0, num_iter):  #always start at 0 because we want to average prediction over all the iterations.
#     pred_col_name = 'pred_'+str(it)
#     pred_columns.append(pred_col_name)
    
#     score_col_name = 'pred_score'+str(it)
#     score_columns.append(score_col_name)


# Preds = crysdf[pred_columns]


# # %%
# def scoreFunc(x):
#     trial_num = sum(x.notna())
#     if trial_num == 0:
#         return np.nan, trial_num
#     res = x.sum()
#     score = res/trial_num
#     return score, trial_num

# # %%
# crysdf['predScore'] = Preds.apply(scoreFunc, axis=1)
# # %%
# crysdf[['predScore', 'trial_num']] = crysdf.predScore.tolist()
# # %%
# crysdf["prediction"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else round(x))
# # %%
# res_df = crysdf[crysdf.predScore.notna()][[
#     'material_id','synth', 'TARGET','prediction', 'predScore', 'trial_num']]  #selecting data with prediction values

# # %%
# experimental_df = res_df[res_df.synth==1]
# theoretical_df = res_df[res_df.synth==0]
# # for co-training we need more analysis, but the experimental vs theoretical distinction is more important.

# # %%
# # true_positive_rate = sum(experimental_df.predScore>=.5)/experimental_df.shape[0]
# # predicted_positive_rate = sum(theoretical_df.predScore>=.5)/theoretical_df.shape[0]
# true_positive_rate = sum(experimental_df.prediction)/experimental_df.shape[0]
# predicted_positive_rate = sum(theoretical_df.prediction)/theoretical_df.shape[0]
# # %%
# print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
# print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))

# # %%
# crysdf = crysdf.drop(columns=Preds)
# # %%
# crysdf.to_pickle(os.path.join(
#     result_dir,experiment+'.pkl'))
#     # result_dir,exp_dict[experiment]+'.pkl'))
# # %%
# synthDF = pd.read_pickle(fulldatapath)
# # synthDF = synthDF.reset_index(drop=True)
# # %%
# merged_df = synthDF[['material_id', 'synth']].merge(
#     theoretical_df[['material_id', 'prediction']], on='material_id', how='left')
# merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity
# cotrain_index = synthDF[
#     synthDF.synth!=synthDF[config['new_target']]].index
# merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training
# merged_df.loc[merged_df.synth == 1, 'new_labels'] = 1 #used in training
# merged_df.new_labels = merged_df.new_labels.astype(np.int16)
# # %%
# # synthDF[exp_dict[experiment]] = merged_df.new_labels
# synthDF[experiment] = merged_df.new_labels
# # %%
# synthDF.to_pickle(fulldatapath)
# # %%
# resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
#                         index_col=0)
# # resultcsv.loc[exp_dict[experiment], 
# resultcsv.loc[experiment, 'true_positive_rate'] = true_positive_rate
# # resultcsv.loc[exp_dict[experiment], 
# resultcsv.loc[experiment, 'predicted_positive_rate'] = predicted_positive_rate
# resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))
# # %%
