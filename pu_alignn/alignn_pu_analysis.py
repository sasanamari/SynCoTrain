# only updated the top  of the file, not inside the function yet.
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
# For each round, we need a separate prediction column and a cotrain label.
# The final round only gets a prediction label.
parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)

parser.add_argument(
    "--ehull015",
    type=str_to_bool,
    default=False,
    help="Predicting stability to evaluate PU Learning's efficacy with 0.015eV cutoff.",
)
parser.add_argument(
    "--small_data",
    type=str_to_bool,
    default=False,
    help="This option selects a small subset of data for checking the workflow faster.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment 
ehull015 = args.ehull015
small_data = args.small_data
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
split_id_dir = f"{data_prefix}{TARGET}_{prop}"
split_id_dir_path = os.path.join(data_dir, split_id_dir)
LOTestPath = os.path.join(split_id_dir_path, 'leaveout_test_id.txt')
with open(LOTestPath, "r") as ff:
        id_LOtest = [int(line.strip()) for line in ff]  
alignn_dir = "pu_alignn"
# os.chdir(alignn_dir)        
# %%
def pu_report_alignn(experiment: str = None, prop: str = None,
              propDFpath=propDFpath,
              TARGET = TARGET,
              id_LOtest = id_LOtest,
              pseudo_label_threshold = 0.75,ehull015 = False,
              small_data = False, data_prefix = data_prefix, 
              max_iter=60):
    print(f'experiment is {experiment}, ehull015 is {ehull015} and small data is {small_data}.')
    output_dir = os.path.join(alignn_dir, f'PUOutput_{data_prefix}{experiment}')
    if ehull015:
        output_dir = os.path.join(alignn_dir, f'PUehull015_{experiment}')
    propDF = pd.read_pickle(propDFpath)
    res_df_list = []
    res_dir_list = []
    for iter, PUiter in enumerate(os.listdir(output_dir)):
        if iter>max_iter:
            break
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
    #lowercase target is produced by ALIGNN
    
    agg_df = agg_df.merge(propDF[['material_id', prop]], on='material_id', how='left')
    experimental_data = agg_df[agg_df[prop]==1] #needs to change for cotrain?
    unlabeled_data = agg_df[agg_df[prop]==0]   
    
    # LO_test = experimental_data.loc[id_LOtest] 
    LO_test = propDF.loc[id_LOtest] 
    LO_test = pd.merge(LO_test, agg_df, on='material_id', how="inner")
    LO_true_positive_rate = LO_test['prediction'].sum()/len(LO_test)
    
    true_positive_rate = experimental_data['prediction'].sum()/len(experimental_data)
    predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)
    
    cotrain_df = propDF[['material_id', prop]].merge(
        agg_df[['material_id','new_labels']], on='material_id', how='left')

    cotrain_index = propDF[propDF[prop]!=propDF[TARGET]].index 
        # output_dir.split('_')[-1]]]].index 
    cotrain_df.loc[cotrain_index, 'new_labels'] = 1 #used in cotraining, not predicted. does nothing at step 0.
    cotrain_df.loc[cotrain_df[prop] == 1, 'new_labels'] = 1 #filling up the NaN used for training.
    # if small_data:
        # cotrain_df = cotrain_df.dropna() #in small data set-up, there will be NaN values for unsed data.
    # if cotrain_df['new_labels'].isna().mean() > 0.02:
    #     raise ValueError("Too many NaN values remaining in 'new_labels'.")
    # else:
    if cotrain_df['new_labels'].isna().mean() > 0.02:
        warn_str = f"{round(cotrain_df['new_labels'].isna().mean(),3)*100}% of the 'new_labels' are NaN."
        warnings.warn(warn_str, RuntimeWarning)
    cotrain_df['new_labels'].fillna(cotrain_df[prop], inplace=True)
            
    cotrain_df.new_labels = cotrain_df.new_labels.astype(np.int16)
       
    report = {'res_dir_list':res_dir_list, 'resdf':resdf,
              'true_positive_rate':round(true_positive_rate, 4), 
              'LO_true_positive_rate':round(LO_true_positive_rate, 4), 
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'GT_true_positive_rate':'',
              'false_positive_rate':'',
              'agg_df':agg_df,
              'cotrain_df': cotrain_df} 
    if ehull015:
        GT_stable = propDF[propDF["stability_GT"]==1] 
        GT_stable = pd.merge(GT_stable, agg_df, on='material_id', how="inner")
        GT_unstable = propDF[propDF["stability_GT"]==0]
        GT_unstable = pd.merge(GT_unstable, agg_df, on='material_id', how="inner") 
        GT_tpr = GT_stable['prediction'].sum()/len(GT_stable)
        false_positive_rate = GT_unstable['prediction'].sum()/len(GT_unstable)
        report['GT_true_positive_rate'] = round(GT_tpr, 4)
        report['false_positive_rate'] = round(false_positive_rate, 4)
    
    return report, propDF

# %%
report, propDF = pu_report_alignn(experiment=experiment, prop=prop,
                                propDFpath=propDFpath,
                                TARGET = TARGET,
                                ehull015 = ehull015,
                                small_data = small_data, data_prefix = data_prefix)
print(f"The True positive rate was {report['true_positive_rate']} and the "
      f"predicted positive rate was {report['predicted_positive_rate']}.")
if ehull015:
    print(f"The Groud Truth true-positive-rate was {report['GT_true_positive_rate']} and the "
      f" False positive rate was {report['false_positive_rate']}.")

# %%    
report['agg_df'].to_pickle(os.path.join(
    result_dir,f'{experiment}.pkl'))
report['resdf'].to_pickle(os.path.join(
    result_dir,f'{experiment}_resdf.pkl'))
# %%
propDF[experiment]=report['cotrain_df'].new_labels #just need the labels
# %%
propDF.to_pickle(propDFpath)
print(propDFpath)
# %%
resultcsv = pd.read_csv(os.path.join(result_dir, 'results.csv'),
                        index_col=0)
new_rates = {'true_positive_rate':report['true_positive_rate'],
             'LO_true_positive_rate':report['LO_true_positive_rate'],
             'predicted_positive_rate':report['predicted_positive_rate'],
             'GT_true_positive_rate':report['GT_true_positive_rate'],
             'false_positive_rate':report['false_positive_rate']}
resultcsv.loc[experiment.split('PUOutput_')[-1]] = new_rates

# %%
resultcsv.to_csv(os.path.join(result_dir, 'results.csv'))
print(new_rates)
print(resultcsv)

# %%