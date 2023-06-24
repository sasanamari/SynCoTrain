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
home_dir = "/home/samariam/projects/chemheuristics"
os.chdir("/home/samariam/projects/chemheuristics/alignn")
output_dir = "/home/samariam/projects/chemheuristics/alignn/PUOutput_fulldata_ehull"
# %%
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

positive_data_pu_fromRef = agg_df[agg_df['target']==1]
unlabeled_data_pu_fromRef = agg_df[agg_df['target']==0]

tpr_pu = positive_data_pu_fromRef['prediction'].sum()/len(positive_data_pu_fromRef)
ppr_pu = unlabeled_data_pu_fromRef['prediction'].sum()/len(unlabeled_data_pu_fromRef)
# %%
data_path = "/home/samariam/projects/chemheuristics/data/alignn_full_data/"
# csv_path = os.path.join(data_path, "ehull_test.csv")
ref_path = os.path.join(data_path, "ehull_test_ref")
# source = pd.read_csv(csv_path, names = ["id", "target"])
refdf = pd.read_pickle(ref_path)
refdf.head()

refdf.set_index("material_id", inplace=True)
# agg_df.set_index("material_id", inplace=True)
# %%
t = agg_df.join(refdf,rsuffix="_s", lsuffix="_a")

# t.target_s.dropna(inplace=True)
# %%
positive_data_ref = t[t['ehull_class']==1]
unlabeled_data_ref = t[t['ehull_class']==0]
# %%
real_tpr = positive_data_ref['prediction'].sum()/len(positive_data_ref)
fpr = unlabeled_data_ref['prediction'].sum()/len(unlabeled_data_ref)
# %%

positive_data_pu_fromRef = t[t['pu_ehull']==1]
unlabeled_data_pu_fromRef = t[t['pu_ehull']==0]

tpr_pu_fromRef = positive_data_pu_fromRef['prediction'].sum()/len(positive_data_pu_fromRef)
ppr_pu_fromRef = unlabeled_data_pu_fromRef['prediction'].sum()/len(unlabeled_data_pu_fromRef)
# %%
