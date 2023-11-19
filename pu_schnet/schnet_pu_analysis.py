# %%
import os
import json
from pathlib import Path  #recommended path library for python3
import pandas as pd
import numpy as np
import sys
import argparse
from experiment_setup import current_setup, str_to_bool
# %%
# For each round, we need a separate prediction column and a cotrain label.
# The final round only gets a prediction label.
parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--experiment",
    default="schnet0",
    help="name of the experiment and corresponding config files.",
)
parser.add_argument(
    "--ehull",
    type=str_to_bool,
    default=False,
    help="Predicting stability to evaluate PU Learning's efficacy.",
)
parser.add_argument(
    "--ehull015",
    type=str_to_bool,
    default=False,
    help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
)
parser.add_argument(
    "--hw",
    type=str_to_bool,
    default=False,
    help="Analysis before the final iteration.",
)
# parser.add_argument(
#     "--schnettest",
#     type=str_to_bool,
#     default=False,
#     help="Predicting stability and checking results.",
# )
parser.add_argument(
    "--small_data",
    type=str_to_bool,
    default=False,
    help="This option selects a small subset of data for checking the workflow faster.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment 
ehull015 = args.ehull015
ehull_test = args.ehull
small_data = args.small_data
half_way_analysis = args.hw
# schnettest = args.schnettest

cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment, ehull015=ehull015)
# cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment, schnettest=schnettest)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]
print(f'experiment is {experiment}, small_data {small_data} & ehull {ehull_test}.')
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(current_dir))
data_dir = os.path.dirname(propDFpath)
split_id_dir = f"{data_prefix}{TARGET}_{prop}"
split_id_dir_path = os.path.join(data_dir, split_id_dir)
LOTestPath = os.path.join(split_id_dir_path, 'leaveout_test_id.txt')
with open(LOTestPath, "r") as ff:
        id_LOtest = [int(line.strip()) for line in ff] 


def pu_report_schnet(experiment: str = None, prop=prop,
              propDFpath = propDFpath, 
              TARGET = TARGET,
              id_LOtest = id_LOtest,
              pseudo_label_threshold = 0.75,
              ehull_test = False, small_data = False, data_prefix = data_prefix,
              max_iteration = 60):
    
    def scoreFunc(x):
        trial_num = sum(x.notna())
        if trial_num == 0:
            return np.nan, trial_num
        res = x.sum()
        score = res/trial_num
        return score, trial_num
    
    
    schnet_config_dir = "pu_schnet/schnet_configs"
    config_path = os.path.join(schnet_config_dir, 'pu_config_schnetpack.json')
    # if half_way_analysis:
    #     config_path = os.path.join(schnet_config_dir, 'pu_config_schnetpack_hw.json')

    # if schnettest:
    #     config_path = os.path.join(schnet_config_dir, 'pu_config_schnetpackTest.json')
        
    with open(config_path, "r") as read_file:
        print("Read Experiment configuration")
        config = json.load(read_file)

    schnetDirectory = config["schnetDirectory"]

    print(os.getcwd())
    print(Path().resolve())  #current working directory
    print(Path().absolute()) #file path in jupyter

    epoch_num = config["epoch_num"]
    start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
    num_iter = config["num_iter"]
    # start_iter = 20 #fix later
    # num_iter = 100 #fix later

    if small_data:
        epoch_num = int(epoch_num*0.5)
                
    res_df_fileName = f'{data_prefix}{experiment}_{str(start_iter)}_{str(num_iter)}ep{str(epoch_num)}'
    save_dir = os.path.join(schnetDirectory,f'PUOutput_{data_prefix}{experiment}')
    if ehull015:
        save_dir = os.path.join(schnetDirectory,f'PUehull015_{experiment}')
    elif ehull_test:
        save_dir = os.path.join(schnetDirectory,f'PUehull_{experiment}')

    if half_way_analysis:
        crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',f'{res_df_fileName}tmp'))   #saving results at each iteration
    else:
        crysdf=pd.read_pickle(os.path.join(save_dir,'res_df',res_df_fileName))
        # crysdf=pd.read_pickle('pu_schnet/PUOutput_coSchAl1/res_df/coSchAl1_17_60ep150')
        
    # crysdf = crysdf.loc[:, ~crysdf.columns.duplicated()] # drops duplicated props at round zero.
   
    pred_columns = []
    score_columns = []
    excess_iters = []

    for it in range(0, num_iter):  #always start at 0 because we want to average prediction over all the iterations.
        pred_col_name = 'pred_'+str(it)
        if half_way_analysis:
            if pred_col_name not in crysdf.columns:
                continue
        if it>max_iteration:
            excess_iters.append(pred_col_name)
        else:            
            pred_columns.append(pred_col_name)
            
            score_col_name = 'pred_score'+str(it)
            score_columns.append(score_col_name)

    Preds = crysdf[pred_columns]
    crysdf['predScore'] = Preds.apply(scoreFunc, axis=1)
    crysdf[['predScore', 'trial_num']] = pd.DataFrame(crysdf.predScore.tolist())
    crysdf["prediction"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else round(x))
    crysdf["new_labels"] = crysdf.predScore.map(lambda x: x if np.isnan(x) else 1 if x >= pseudo_label_threshold else 0)

    res_df = crysdf[crysdf.predScore.notna()][[
        'material_id',prop, TARGET,'prediction', 'predScore', 'trial_num']]  #selecting data with prediction values
    res_df = res_df.loc[:, ~res_df.columns.duplicated()] # drops duplicated props at round zero.

    crysdf = crysdf.drop(columns=Preds)
    crysdf = crysdf.drop(columns=excess_iters) #might require a df like Preds
    
    

    experimental_df = res_df[res_df[prop]==1]
    theoretical_df = res_df[res_df[prop]==0]
    
    true_positive_rate = experimental_df.prediction.mean()
    predicted_positive_rate = theoretical_df.prediction.mean()
    
    propDF = pd.read_pickle(propDFpath)
    
    LO_test = propDF.loc[id_LOtest] 
    LO_test = pd.merge(LO_test, crysdf, on='material_id', how="inner")
    LO_true_positive_rate = LO_test.prediction.mean()
    
    print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
    print('Our LO true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(LO_true_positive_rate*100, num_iter, epoch_num))
    print('and {:.1f}% of currenly unlabeled data have been predicted to belong to the positive class.'.format(predicted_positive_rate*100))    
    
    # merged_df = propDF[['material_id', [prop]]].merge(
    merged_df = propDF[['material_id', prop]].merge(
        theoretical_df[['material_id', 'prediction']], on='material_id', how='left')
    merged_df = merged_df.rename(columns={'prediction': 'new_labels'}) #for clarity

# The data used in training will have nan values. We simplify below and check in practice.
    cotrain_index = propDF[(propDF[prop] != propDF[TARGET])].index
    merged_df.loc[cotrain_index, 'new_labels'] = 1 #used in training, , not predicted. empty index at step 0.
  
    merged_df.loc[merged_df[prop] == 1, 'new_labels'] = 1 #used in training

    merged_df.new_labels = merged_df.new_labels.astype(pd.Int16Dtype())
    
    propDF[experiment] = merged_df.new_labels
    
    report = {'resdf':Preds, 'agg_df':crysdf, 
              'true_positive_rate':round(true_positive_rate, 4), 
              'LO_true_positive_rate':round(LO_true_positive_rate, 4),
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'GT_true_positive_rate':'',
              'false_positive_rate':'',}
    if ehull_test or ehull015:
        GT_stable = propDF[propDF["stability_GT"]==1] 
        GT_stable = pd.merge(GT_stable, crysdf, on='material_id', how="inner")
        GT_unstable = propDF[propDF["stability_GT"]==0]
        GT_unstable = pd.merge(GT_unstable, crysdf, on='material_id', how="inner") 
        GT_tpr = GT_stable['prediction'].mean()
        false_positive_rate = GT_unstable['prediction'].mean()
        report['GT_true_positive_rate'] = round(GT_tpr, 4)
        report['false_positive_rate'] = round(false_positive_rate, 4)
        print(f"The Groud Truth true-positive-rate was {report['GT_true_positive_rate']*100}% and the "
      f" False positive rate was {100*report['false_positive_rate']}%.")
        
    return report, propDF
# %%
report, propDF = pu_report_schnet(experiment = experiment,
                                TARGET = TARGET,
                                propDFpath=propDFpath, ehull_test = ehull_test, 
                                small_data = small_data)
if half_way_analysis:
    pass 
else:
    report['agg_df'].to_pickle(os.path.join( #goes one directory up from schnet to main dir.
        f'{result_dir}',f'{experiment}.pkl'))
    report['resdf'].to_pickle(os.path.join( 
        f'{result_dir}',f'{experiment}_resdf.pkl'))
    
    # print(propDF.head(3))
    # print(propDF.tail(3))
    propDF.to_pickle(propDFpath)
    print(propDFpath)
    # # %%
    csv_path =os.path.join(f'{result_dir}', 'results.csv')
    resultcsv = pd.read_csv(csv_path,
                            index_col=0)
    new_rates = {'true_positive_rate':report['true_positive_rate'],
                'LO_true_positive_rate':report['LO_true_positive_rate'],
                'predicted_positive_rate':report['predicted_positive_rate'],
                'GT_true_positive_rate':report['GT_true_positive_rate'],
                'false_positive_rate':report['false_positive_rate']}
    resultcsv.loc[experiment] = new_rates
    # %%
    resultcsv.to_csv(csv_path)
    print(resultcsv)
# %%