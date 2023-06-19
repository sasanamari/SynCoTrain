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
home_dir = "/home/samariam/projects/synth"
os.chdir("/home/samariam/projects/synth/alignn")
# output_dir = 'PUOutput_test6'
# old_output_to_compare = 'PUOutput_test2'
# output_dir = 'PUOutput_test7_onlyAlignn_layers'
# %%
def pu_report(output_dir : str = None, cotraining = False):
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
    
    if cotraining:
        this might need changing:
        # new_labels_path = '/home/samariam/projects/chemheuristics/data/schnet_0.pkl'
        cotrain_df0 = pd.read_pickle(cotraining) 
        agg_df = agg_df.join(cotrain_df0.original_target)        
        positive_data = agg_df[agg_df['original_target']==1]
        unlabeled_data = agg_df[agg_df['original_target']==0]
    else:
        positive_data = agg_df[agg_df['target']==1]
        unlabeled_data = agg_df[agg_df['target']==0]
    
    if output_dir.endswith("ehull"):
        true_positive_rate = "Need to compared to 'ehull_test_ref' in data dir."
        predicted_positive_rate = true_positive_rate
    else:
        true_positive_rate = positive_data['prediction'].sum()/len(positive_data)
        predicted_positive_rate = unlabeled_data['prediction'].sum()/len(unlabeled_data)
    
    report = {'res_dir_list':res_dir_list, 'resdf':resdf,
              'true_positive_rate':round(true_positive_rate, 4), 
              'predicted_positive_rate': round(predicted_positive_rate, 4),
              'agg_df':agg_df}
    return report
# %%
# co_report = pu_report(output_dir = "/home/samariam/projects/chemheuristics/alignn/PUOutput_fulldata_cotraining_0",
#                    cotraining = '/home/samariam/projects/chemheuristics/data/schnet_0.pkl')
# cotraining_res_dir = "/home/samariam/projects/chemheuristics/reports/cotraining_0"
# data_path = "/home/samariam/projects/chemheuristics/data"
# cotrain_res_path = os.path.join(cotraining_res_dir, "cotrain_alignn_from_schnet_0.pkl")
# cotrain_data_path = os.path.join(data_path, "cotrain_alignn_from_schnet_0.pkl")
# co_report["resdf"].to_pickle(cotrain_res_path)
# co_report["agg_df"].to_pickle(cotrain_data_path)
# %%
# report = pu_report(output_dir = output_dir)
print("stop")
experiment_path =  "/home/samariam/projects/synth/alignn/PUOutput_100runs"
# report = pu_report(output_dir = "/home/samariam/projects/chemheuristics/alignn/PUOutput_fulldata_2")
report = pu_report(output_dir = experiment_path)
# %%
# res_dir = "/home/samariam/projects/chemheuristics/reports/1st_fulldata_schnet_alignn"
res_dir = "/home/samariam/projects/synth/reports/longlightDBug"
res_path = os.path.join(res_dir, "puAlignnFullData_2.pkl")
report["resdf"].to_pickle(res_path)
data_path = "/home/samariam/projects/synth/data"
experiment_name = "alignn_0.pkl"
report['agg_df'].to_pickle(os.path.join(data_path,experiment_name))

# %%
# print("True positive rate of this experiment:", 
#       report['true_positive_rate'])
# print("Positive prediction for the unlabeled data: ",
#       report['predicted_positive_rate'])
# %%
for direc in os.listdir():
    if direc.startswith("PU"):
        print(direc)
# %%
# get a list of all directories in the current working directory
dirs = [d for d in os.listdir() if os.path.isdir(d)]
# filter the list to include only directories that start with "PU"
dirs = [d for d in dirs if d.startswith('PUOutput')]
dirs = [d for d in dirs if not d.endswith('ehull')]  #separating  the ehull test
expers = [os.path.join(d, "2iter") for d in dirs]
# %%
configs = [loadjson(os.path.join(exper,"config.json")) for exper in expers]
configs[0]["output_dir"] = './alignn/sample_synth/../PUOutput_test1/2iter/'
# just a correction, name changed later

# %%
def move_up(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result.update(move_up(v))
        else:
            result[k] = v
    return result

# %%
configs = [move_up(d) for d in configs]
# %%
configs = [{k: v.split('/')[-3] if k=='output_dir' else v for k, v in conf.items()} for conf in configs]

# %%
# configs = [conf.update({"true_positive_rate": pu_report(conf["output_dir"])["true_positive_rate"]}) for conf in configs]
for conf in configs:
    conf["true_positive_rate"] = pu_report(conf["output_dir"])["true_positive_rate"]
for conf in configs:
    conf['predicted_positive_rate'] = pu_report(conf["output_dir"])['predicted_positive_rate']    
# %%
res_comp = pd.DataFrame(configs)
# %%
res_comp.drop(columns=['version', 'dataset', 'target', 'neighbor_strategy',
       'id_tag', 'n_val', 'n_test',
       'n_train', 'train_ratio', 'val_ratio', 'test_ratio',
       'target_multiplication_factor', 'filename', 
        'save_dataloader', 'write_checkpoint',
       'write_predictions', 'store_outputs', 'progress', 'log_tensorboard',
       'standard_scalar_and_pca', 
       'distributed', 'name', 'atom_input_features',
       'edge_input_features', 'triplet_input_features', 'embedding_features',
        'zero_inflated',
       ], inplace=True)

# %%
summary_df = res_comp.loc[:,res_comp.nunique()>1]
# %%
summary_df = summary_df.reindex(columns=[ 'true_positive_rate', 'predicted_positive_rate','batch_size', 'learning_rate', 'pin_memory', 'num_workers', 'cutoff',
       'n_early_stopping', 'output_dir', 'alignn_layers', 'gcn_layers',
       'hidden_features',])
summary_df.sort_values(["true_positive_rate"], ascending=False,inplace=True)
# summary_df.set_index("output_dir")
# %%
summary_df  #for interactive session
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


# # %%
# from alignn.utils import plot_learning_curve
# # %%
# plot_learning_curve(PUiter, key="rocauc", plot_train=True);
# %%



# def compare_config(dir_a, dir_b):
#     ""
#     dir_a_pu = os.path.join(dir_a, "1iter")
#     dir_b_pu = os.path.join(dir_b, "1iter")
#     x = loadjson(os.path.join(dir_a_pu,"config.json"))
#     y = loadjson(os.path.join(dir_b_pu,"config.json"))
#     diff = DeepDiff(x, y)
#     return diff    
# # %%
# old_report = pu_report(output_dir = old_output_to_compare)
# # %%
# compare_df = pd.DataFrame()
# compare_df["experiments"] = ["old", "current"]
# compare_df["predicted_positive"] = [round(old_report['predicted_positive_rate'], 3),
#                                round(report['predicted_positive_rate'], 3) ]
# compare_df["true_positive"] = [round(old_report['true_positive_rate'], 3),
#                                round(report['true_positive_rate'], 3) ]
# compare_df.set_index("experiments", inplace=True)
# print(tabulate(compare_df, headers='keys', tablefmt='fancy_outline'))
# # %%
# pp = pprint.PrettyPrinter(indent=1)
# comp = compare_config(old_output_to_compare, output_dir)
# del comp['values_changed']["root['output_dir']"]
# pp.pprint(comp)

# # our true positive rate is quite poor (60%) for 2*2 layer network and 15 epochs.
# # Need to train more. either on epochs (based on validation/history)
# # or increase network layers to 4?
# # The validation runs are just beginning to plateau. I'll try increasing both, see what happens.
# %%
# data_path = "/home/samariam/projects/chemheuristics/alignn/sample_synth"
# # %%
# orig_data_path = os.path.join(data_path, "synth_id_prop.csv")
# rev_data_path = os.path.join(data_path, "synth_id_prop_rev.csv")
# # %%
# rev_data = pd.read_csv(rev_data_path, names=["poscar_id", "target"])
# rev_data['material_id'] = rev_data['poscar_id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])

# # %%
# orig_data = pd.read_csv(orig_data_path, names=["poscar_id", "target"])
# orig_data['material_id'] = orig_data['poscar_id'].map(lambda material_id: re.split('CAR-|.vas', material_id)[1])

# %%
hv = loadjson("/home/samariam/projects/chemheuristics/alignn/pretrain_test/_history_val.json")
ht = loadjson("/home/samariam/projects/chemheuristics/alignn/pretrain_test/_history_train.json")

# %%
plt.plot(hv["recall"])
plt.plot(ht["recall"])
# %%
