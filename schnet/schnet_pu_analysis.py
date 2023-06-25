# %%
import os
import json
from pathlib import Path  #recommended path library for python3
import pandas as pd
import numpy as np

# current_config = "25runTest_config.json"
# current_config = "debugsch_config.json"
# current_config = "longDebug_config.json"
# current_config = "100runs_config.json"
# current_config = "cotrain_debug_config.json"
current_config = "coSchAl1_config.json"

result_dir = '/home/samariam/projects/synth/data/results'

config_dir = "/home/samariam/projects/synth/schnet/schnet_configs/"
with open(os.path.join(config_dir, current_config), "r") as read_file:
    print("Read Experiment configuration")
    config = json.load(read_file)
    
schnetDirectory = config["schnetDirectory"]
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
res_df_fileName = experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)


save_dir = os.path.join(schnetDirectory,experiment+'_res_log')
fulldatapath = config["fulldatapath"]
# exp_dict = { #for publishing, just use the value names for running experiments.
#     '100runs':'schnet0',
#     'coSchAl1':'coSchAl1',
#     'coSchAl2':'coSchAl2',
# }
# %%
half_way_analysis = False
# half_way_analysis = True
if half_way_analysis:
    crysdf=pd.read_pickle(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration
else:
    crysdf=pd.read_pickle(save_dir+'/res_df/'+res_df_fileName)
# I chanegd crysdf to it_test_df!!!
# from here on is the analysis, I should add to a separate file.
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
# %%
res_df = crysdf[crysdf.predScore.notna()][[
    'material_id','synth', 'TARGET','prediction', 'predScore', 'trial_num']]  #selecting data with prediction values

# %%
experimental_df = res_df[res_df.synth==1]
theoretical_df = res_df[res_df.synth==0]
# for co-training we need more analysis, but the experimental vs theoretical distinction is more important.

# %%
# true_positive_rate = sum(experimental_df.predScore>=.5)/experimental_df.shape[0]
# predicted_positive_rate = sum(theoretical_df.predScore>=.5)/theoretical_df.shape[0]
true_positive_rate = sum(experimental_df.prediction)/experimental_df.shape[0]
predicted_positive_rate = sum(theoretical_df.prediction)/theoretical_df.shape[0]
# %%
print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))

# %%
crysdf = crysdf.drop(columns=Preds)
# %%
crysdf.to_pickle(os.path.join(
    result_dir,exp_dict[experiment]+'.pkl'))
# %%
synthDF = pd.read_pickle(fulldatapath)
# synthDF = synthDF.reset_index(drop=True)
# %%
merged_df = synthDF[['material_id', 'synth']].merge(
    theoretical_df[['material_id', 'prediction']], on='material_id', how='left')
merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity
cotrain_index = synthDF[synthDF.synth!=synthDF.alignn0].index
merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training
merged_df.loc[merged_df.synth == 1, 'new_labels'] = 1 #used in training
merged_df.new_labels = merged_df.new_labels.astype(np.int16)
# %%
synthDF[exp_dict[experiment]] = merged_df.new_labels
# %%
synthDF.to_pickle(fulldatapath)
# %%
resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
                        index_col=0)
resultcsv.loc[exp_dict[experiment], 
            'true_positive_rate'] = true_positive_rate
resultcsv.loc[exp_dict[experiment], 
            'predicted_positive_rate'] = predicted_positive_rate
# resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))
# %%
