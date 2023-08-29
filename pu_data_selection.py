# %%
import numpy as np
import pandas as pd
import os
import sys
import argparse
from experiment_setup import current_setup
# %%
data_dir = '/home/samariam/projects/synth/data/clean_data/'

# %%
test_portion = 0.1
parser = argparse.ArgumentParser(
    description="Data preparation for ALIGNN"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)
parser.add_argument(
    "--ehull",
    default=False, #change this! manually defined cos lazy.
    help="Predicting stability to evaluate PU Learning's efficacy.",
)
parser.add_argument(
    "--small_data",
    default=False, #change this! manually defined cos lazy.
    help="Run the synthesizability experiment with smaller data to check the pipeline.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment
ehull_test = args.ehull
small_data = args.small_data
cs = current_setup(ehull_test=ehull_test, small_data=small_data)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
data_prefix = "small_" if small_data else ""
# %%
def data_id_selector(TARGET,
                     prop,
                     data_path = '/home/samariam/projects/synth/data/clean_data/synthDF',
                     num_iter = 100,
                     test_ratio = 0.1,
                     small_data = False,
                     
                     ):
        
    df = pd.read_pickle(data_path)
    df = df[['material_id', prop, TARGET]]
    df = df.loc[:, ~df.columns.duplicated()] # drops duplicated synth at round zero.
    df = df[~df[TARGET].isna()] #remving NaN values. for small_data
    data_dir = os.path.dirname(data_path)
    
    experiment_dir = f'{data_prefix}{TARGET}{prop}'
    dir_path = os.path.join(data_dir, experiment_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    os.chdir(dir_path)
    experimentalDataSize = df[prop].sum()
    
    with open(f"experimentalDataSize.txt", "w") as f:
        f.write(str(experimentalDataSize))
    # select validation set inside alignn/schnet        
    positive_df = df[df[TARGET]==1]
    
    for it in range(num_iter):
        testdf1 = positive_df.sample(frac = test_ratio, random_state =it)
        df_wo_test = df.drop(index=testdf1.index) #remove test data
        traindf1 = df_wo_test[df_wo_test[TARGET]==1].sample(frac=1, random_state = it+1)
        class_train_num = len(traindf1)
        traindf0 = df_wo_test[df_wo_test[TARGET]==0].sample(n=class_train_num,random_state=it+2) #a different 'negative' train-set at each iteration.
        testdf0 = df_wo_test[df_wo_test[TARGET]==0].drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
        
        it_traindf = pd.concat([traindf0,traindf1])
        it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
        it_traindf = it_traindf.sample(frac=1,random_state=it+3)
        it_testdf = it_testdf.sample(frac=1,random_state=it+4)
        
        with open(f"train_id_{it}.txt", "w") as f:
            for it_train_id in it_traindf.index:
                f.write(str(it_train_id) + "\n")
        with open(f"test_id_{it}.txt", "w") as f:
            for it_test_id in it_testdf.index:
                f.write(str(it_test_id) + "\n")
        # break
        # print(f'saving ids for iteration {it}.')
    
    return

# %%
data_id_selector(TARGET='synth',
                     data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
                     num_iter = 100,
                     test_ratio = 0.1,
                     )
# %%
data_id_selector(TARGET='synth',
                     data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
                     test_ratio = 0.1,
                     num_iter = 100,
                     )
# %%
# data_id_selector(TARGET='alignn0_constant',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='alignn0_dynamic',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='schnet0_constant',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='schnet0_dynamic',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='coSchAl1_constant',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='coSchAl1_dynamic',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# %%
# data_id_selector(TARGET='coAlSch1_constant',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
# # %%
# data_id_selector(TARGET='coAlSch1_dynamic',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      )
split_id_dir = f"{experiment_target_match[experiment]}_{data_prefix}data"
split_id_path = os.path.join(data_dir, split_id_dir)
the above is from the current alignn_pu_learning
also, want to call preparying_data_byfile as a function.

# # %%
# import numpy as np
# import pandas as pd
# import os
# import sys
# import shutil
# # %%
# small_data_frac = 0.05 
# data_dir = '/home/samariam/projects/synth/data/clean_data/'

# # %%
# test_portion = 0.1

# # %%
# def data_id_selector(TARGET,
#                      data_path = '/home/samariam/projects/synth/data/clean_data/synthDF',
#                     #  small_data_frac=None,
#                      num_iter = 100,
#                      test_ratio = 0.1
#                      ):
    
#     file_name = os.path.basename(data_path)
#     test_strategy = file_name.split('_')[-1]
#     small_data = file_name.startswith("small")  
        
#     df = pd.read_pickle(data_path)
#     df = df[['material_id', 'synth', TARGET]]
#     df = df.loc[:, ~df.columns.duplicated()] # drops duplicated synth at round zero.
#     df = df[~df[TARGET].isna()] #remving NaN values. for small_data
#     data_dir = os.path.dirname(data_path)
#     if small_data:
#         experiment_dir = TARGET.split(f'_{test_strategy}')[0]+'_small_data_'+test_strategy
#         dir_path = os.path.join(data_dir, experiment_dir)
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         os.chdir(dir_path)
#     else:
#         experiment_dir = TARGET.split(f'_{test_strategy}')[0]+'_'+test_strategy
#         dir_path = os.path.join(data_dir, experiment_dir)
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         os.chdir(dir_path)
        
#     experimentalDataSize = df.synth.sum()
    
#     with open(f"experimentalDataSize.txt", "w") as f:
#         f.write(str(experimentalDataSize))
    
    
#     # select validation set inside alignn/schnet
#     if test_strategy == 'constant':
#         experimental_df = df[df.synth==1]
        
#         testdf1 = experimental_df.sample(frac = test_ratio, random_state = 12345)   
#         df_wo_test = df.drop(index=testdf1.index) #remove test data
#         traindf1 = df_wo_test[df_wo_test[TARGET]==1].sample(frac=1, random_state = 54321)
#         class_train_num = len(traindf1)
        
#         for it in range(0, num_iter):
#             negative_df = df_wo_test[df_wo_test[TARGET]==0]
            
#             traindf0 = negative_df.sample(n=class_train_num,random_state=it) #a different 'negative' train-set at each iteration.
#             testdf0 = negative_df.drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
            
#             it_traindf = pd.concat([traindf0,traindf1])
#             it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
#             it_traindf = it_traindf.sample(frac=1,random_state=it+1)
#             it_testdf = it_testdf.sample(frac=1,random_state=it+2)
            
#             with open(f"train_id_{it}.txt", "w") as f:
#                 for it_train_id in it_traindf.index:
#                     f.write(str(it_train_id) + "\n")
#             with open(f"test_id_{it}.txt", "w") as f:
#                 for it_test_id in it_testdf.index:
#                     f.write(str(it_test_id) + "\n")
#             # break
#             # print(f'saving ids for iteration {it}.')
#     elif test_strategy=='dynamic':
        
#         positive_df = df[df[TARGET]==1]
#         negative_df = df[df[TARGET]==0]
        
#         for it in range(0, num_iter):
#             testdf1 = positive_df.sample(frac = test_ratio, random_state =it+2)
#             df_wo_test = df.drop(index=testdf1.index) #remove test data
#             traindf1 = df_wo_test[df_wo_test[TARGET]==1].sample(frac=1, random_state = it+1)
#             class_train_num = len(traindf1)
#             traindf0 = df_wo_test[df_wo_test[TARGET]==0].sample(n=class_train_num,random_state=it) #a different 'negative' train-set at each iteration.
#             testdf0 = df_wo_test[df_wo_test[TARGET]==0].drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
            
#             it_traindf = pd.concat([traindf0,traindf1])
#             it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
#             it_traindf = it_traindf.sample(frac=1,random_state=it+3)
#             it_testdf = it_testdf.sample(frac=1,random_state=it+4)
            
#             with open(f"train_id_{it}.txt", "w") as f:
#                 for it_train_id in it_traindf.index:
#                     f.write(str(it_train_id) + "\n")
#             with open(f"test_id_{it}.txt", "w") as f:
#                 for it_test_id in it_testdf.index:
#                     f.write(str(it_test_id) + "\n")
#             # break
#             # print(f'saving ids for iteration {it}.')
#     else:
#         print('invalide test strategy')
#         sys.exit()
    
#     return

# # %%
# data_id_selector(TARGET='synth',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
#                      num_iter = 100,
#                      test_ratio = 0.1,
#                      )
# # %%
# data_id_selector(TARGET='synth',
#                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
#                      test_ratio = 0.1,
#                      num_iter = 100,
#                      )
# # %%
# # data_id_selector(TARGET='alignn0_constant',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='alignn0_dynamic',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='schnet0_constant',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='schnet0_dynamic',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='coSchAl1_constant',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='coSchAl1_dynamic',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # %%
# # data_id_selector(TARGET='coAlSch1_constant',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_constant',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )
# # # %%
# # data_id_selector(TARGET='coAlSch1_dynamic',
# #                      data_path = '/home/samariam/projects/synth/data/clean_data/small_synthDF_dynamic',
# #                      num_iter = 100,
# #                      test_ratio = 0.1
# #                      )

