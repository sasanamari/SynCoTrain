# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(3) # use before loading lightning.gpu
from pathlib import Path 
import sys
import numpy as np
import random
import schnetpack as spk
from schnetpack.data import ASEAtomsData, AtomsDataModule
import schnetpack.transform as trn
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
import json
import pandas as pd
import pu_schnet.pu_learn.int2metric as int2metric
# from schnet.pu_learn.Datamodule4PU import *
# from schnet.pu_learn.schnet_funcs  import directory_setup, predProb
# changed imports here for this directory, revert back to commented versions for git.
from pu_learn.Datamodule4PU import *
from pu_learn.schnet_funcs  import directory_setup, predProb
import argparse
# import time

# %%
parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--experiment",
    default="schnet0",
    help="name of the experiment and corresponding config files.",
)
# !!!!!!!!!Change the default for small data after debugging!!!!!!
parser.add_argument(
    "--small_data",
    # default=False,
    # change this is final version.
    default=True,
    help="name of the experiment and corresponding config files.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment 
small_data = args.small_data
data_prefix = "small_" if small_data else ""
# print(os.path.exists('data/clean_data/synth_small_data_constant/train_id_24.txt'))
# time.sleep(20)

# change the abs path below, uncomment.
# schnet_config_dir = "schnet/schnet_configs"
# %%
test_strategy = 'constant'
# test_strategy = 'dynamic'
# change strategy manually, remove it in the final version.
# changed this bit cos lazy
experiment = "coSchAl1"
schnet_config_dir = "/home/samariam/projects/synth/schnet/schnet_configs"
config_path = os.path.join(schnet_config_dir, 'pu_config_'+experiment+'.json')
with open(config_path, "r") as read_file:
    print("Read Experiment configuration")
    config = json.load(read_file)
    
schnetDirectory = config["schnetDirectory"]
# os.chdir(schnetDirectory)
# commented out (changed) the above line for cleaner path management.
# Might need to keep it, if other paths are broken.

print(os.getcwd())
# tensorboard command is tensorboard --logdir={experiment}_res_log/lightning_logs/
# %%
print(Path().resolve())  #current working directory
print(Path().absolute()) #file path in jupyter
print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# time.sleep(600)
# %%
# ###quick settings
cotraining = config["cotraining"]
# changed here for dynamic/constant experimenmt
# new_target = config["new_target"]
new_target = f'{config["new_target"]}_{test_strategy}'
epoch_num = config["epoch_num"]
if small_data:
    epoch_num = int(epoch_num*0.5)
start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
num_iter = config["num_iter"]
batch_size = config["batch_size"]
experiment = config["experiment"]
# small_data = config["small_data"] #moved it to argparse
# res_df_fileName = data_prefix+experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)
res_df_fileName = f'{data_prefix}{experiment}_{str(start_iter)}_{str(num_iter)}ep{str(epoch_num)}'

# prev_fileName = 'blahblah/res_df/poster_iter70ep15tmp'
# save_dir = os.path.join(schnetDirectory,experiment+'_res_log')
# wanna change the test strategy to small_data (or not small data) in the final version.
save_dir = os.path.join(schnetDirectory,experiment+'_'+test_strategy+'_res_log')
data_dir = config["data_dir"]
res_dir = os.path.join(save_dir,'res_df')

# %%
np.random.seed(42)
# %%
# changed below for dynamic/constant experiment
# crysdf = pd.read_pickle(os.path.join(data_dir, 'synthDF'))

crysdf = pd.read_pickle(os.path.join(data_dir, f'{data_prefix}synthDF_{test_strategy}'))
crysdf["TARGET"] = crysdf.synth.copy()
# %%
experiment_target_match = { #output_dir: training_label_column
            'schnet0':'synth',
            'coSchAl1':'alignn0',
            'coSchAl2':'coAlSch1',
                            } needs to change to experiment setup
if small_data:
    split_id_dir = experiment_target_match[experiment]+'_small_data_'+test_strategy
else:
    split_id_dir = experiment_target_match[experiment]+'_'+test_strategy
split_id_path = os.path.join(data_dir, split_id_dir)
# %%
if cotraining:
    print("I haven't yet modified the cotrain_labeling_schnet for synthdf!")
    print("I should remove user input for running in the background.")
    if new_target == None:
        print("You forgot to assign new target column forcotraining!")
        sys.exit()
    else:
        print(f'You have assigned "{new_target}" as cotraining target.')
        crysdf["TARGET"] = crysdf[new_target].copy()
        
#%% 
#  also here, the test set should depend on Synth, not Target :/    
crysdf["targets"] = crysdf.TARGET.map(lambda target: np.array(target).flatten())
#we need the array to have the shape (1,), hence we use flatten()
crysdf["targets"] = crysdf.targets.map(lambda target: dict(synth=np.array(target)) )     
#The above changes targets fromat from array to dict with array val
iteration_results = crysdf[["material_id", 'synth', 'TARGET']]
# %%
# num_experimental = crysdf.synth.sum()
# num_theo = crysdf.shape[0]-num_experimental
# experimental_test_num = int(0.1*min(num_experimental, num_theo))
# positive_index = list(range(num_experimental))
# random.seed(42)
# random.shuffle(positive_index)
# testdf1 = crysdf[crysdf.synth==1].iloc[positive_index[:experimental_test_num]]

# crysdf = crysdf.drop(index=testdf1.index) # need to remove positive-test...

# traindf1 = crysdf[crysdf.TARGET==1].sample(frac=1,random_state=123)
# class_train_num = len(traindf1) #number of train-data from each class.

# if small_data:
#     traindf1 = traindf1.sample(frac = .05, random_state = 42)  #for quick computation and debugginng
#     testdf1 = testdf1.sample(frac = .05, random_state = 43)  #for quick computation and debugginng
# positivePredictionLength = len(testdf1)



# trainLength = round(2*len(traindf1)*.8)-5 #The size of taining data is twice the number of the training data with either label.
# valLength = round(2*len(traindf1)*.2)-5
# innerTestLength = (2*len(traindf1))-(trainLength+valLength)   #Fatal error without internal test set.

# %%
cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64
dataDir = os.path.join(schnetDirectory,"schnetDatabases")
# changed the database name below. Just uncomment and remove the lines below in the final version.
# testDatapath =os.path.join(dataDir,experiment+'_test_dataset.db')
# trainDataPath = os.path.join(dataDir,experiment+'_train_dataset.db')
testDatapath =os.path.join(dataDir,experiment+test_strategy+'_test_dataset.db')
trainDataPath = os.path.join(dataDir,experiment+test_strategy+'_train_dataset.db')
bestModelPath = os.path.join(save_dir,'best_inference_model')
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms

schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=3, radial_basis=radial_basis,    
    cutoff_fn = spk.nn.CosineCutoff(cutoff),
)

pred_synth = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='synth')

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_synth],
)

output_synth = int2metric.ModelOutput4ACC(
    name='synth',
    loss_fn=torch.nn.BCEWithLogitsLoss(), 
    loss_weight=1.,
    metrics={
        "Accur": torchmetrics.Accuracy("binary")   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_synth],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
# %%
for it in range(start_iter, num_iter):      
        
    print('we started iteration {}'.format(it))
    splitFilestring = directory_setup(res_dir = res_dir, 
                                      dataPath = trainDataPath, save_dir = save_dir,
                                      bestModelPath= bestModelPath, iteration_num=it)
    
    np.random.seed(it)
    
    # traindf0 = crysdf[crysdf.TARGET==0].sample(n=class_train_num,random_state=it) #a different 'negative' train-set at each iteration.
    # testdf0 = crysdf[crysdf.TARGET==0].drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
    # if small_data:
    #     traindf0 = traindf0.sample(frac = .05, random_state = 44)
    #     testdf0 = testdf0.sample(frac = .05, random_state = 44)
        
    # it_traindf = pd.concat([traindf0,traindf1])
    # it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
# !we need to read the train/test_id_path s below:    
    train_id_path = os.path.join(split_id_path, f'train_id_{it}.txt')
    test_id_path = os.path.join(split_id_path, f'test_id_{it}.txt')
    experimentalDataSize_path = os.path.join(split_id_path, 'experimentalDataSize.txt')
    
    
    with open(train_id_path, "r") as f:
        id_val_train = [int(line.strip()) for line in f]
    
    with open(test_id_path, "r") as f2:
        id_test = [int(line.strip()) for line in f2]
        
    with open(experimentalDataSize_path, "r") as f3:
        experimentalDataSize = [int(line.strip()) for line in f3][0]        
    
    it_traindf = crysdf.loc[id_val_train]
    it_testdf = crysdf.loc[id_test]
    total_data_length = len(it_testdf)+len(it_traindf)
    
    valLength = int(len(it_traindf)*.1)-5
    trainLength = int(len(it_traindf)*.9)-5 
    positivePredictionLength = it_testdf.TARGET.sum()
    unlabeledPredictionLength = len(it_testdf)-positivePredictionLength
    testLength = len(it_testdf)

    # del traindf0, testdf0 

    it_traindf = it_traindf.sample(frac=1,random_state=it, ignore_index=True) #shuffling for each iteration.
    # change here, reset index according to stackoverflow.
    it_traindf.reset_index(drop=True, inplace=True)
    it_testdf.reset_index(drop=True, inplace=True)
    
    if it==1:
        print('The #training data is {}, #validation data {} and #internal test data {}. '.format(trainLength, valLength, innerTestLength))
        print(f"The total number of test-set (predictions) is {testLength}, out of which {unlabeledPredictionLength} are unlabeled\
 and {positivePredictionLength} are labeled positive.")

    class_dataset = ASEAtomsData.create(trainDataPath, 
                                    distance_unit='Ang',
                                    property_unit_dict={'synth':int(1)}     #The unit is int(1); aka unitless.                             
                                     )
    print('adding systems to dataset')
    class_dataset.add_systems(np.array(it_traindf.targets), np.array(it_traindf.atoms))  
    print('creating data module')
    crysData = AtomsDataModule(datapath=trainDataPath,
                   batch_size=batch_size,
                    num_train=trainLength,
                    num_val=valLength,
                    transforms=[
                        trn.ASENeighborList(cutoff=5.),
                        trn.CastTo32(), 
                                ],
                    property_units={'synth':int(1)},
                    num_workers=4,    
                    split_file = splitFilestring, 
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=['synth'], 
    )
    
    crysData.prepare_data()
    crysData.setup()
    
    splitFilestringTest = directory_setup(res_dir = res_dir, 
                                          dataPath = testDatapath,save_dir = save_dir, 
                                          bestModelPath= bestModelPath, iteration_num=it)

    test_dataset = ASEAtomsData.create(testDatapath, 
                                    distance_unit='Ang',
                                    property_unit_dict={
                                        'synth':int(1),
                                        })
    print('adding systems to the test dataset')
    test_dataset.add_systems(np.array(it_testdf.targets), np.array(it_testdf.atoms))  

    print('creating data module')

    crysTest = DataModuleWithPred(
                    datapath=testDatapath,
                    batch_size=batch_size,
                    num_train=0,
                    num_val=0, 
                    num_test=len(it_testdf),
                    transforms=[
                        trn.ASENeighborList(cutoff=5.),
                        trn.CastTo32(), 
                                ],
                    property_units={'synth':int(1)},
                    num_workers=4,
                    split_file = splitFilestringTest, 
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=['synth'], 
                                )

    crysTest.prepare_data()
    crysTest.setup("test")
    
    means, stddevs = crysData.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True)
    
    meansTest, stddevsTest = crysTest.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True,
    mode = 'test')
    
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())
    # This doesn't work when no test data is given.    
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    callbacks = [
        # early_stopping,
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(save_dir, "best_inference_model"),
        save_top_k=1,
        # monitor="val_loss"
        monitor="val_synth_Accur"
    )
    ]
       

    trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    auto_select_gpus = True,
    strategy=None, 
    precision=16,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=save_dir,
    max_epochs=epoch_num, 
    )

    trainer.fit(task, datamodule=crysData)
    
    myaccuracy = trainer.callback_metrics["val_synth_Accur"]
    print(myaccuracy)
    
    predictions = trainer.predict(model=task, 
                    dataloaders= crysTest.predict_dataloader(),
                    return_predictions=True)

    # %%
    results = []
    for batch in predictions:    
        for datum in batch['synth']:
            results = results+[predProb(datum.float())]
            
    res_list = []
    for i, datum in enumerate(crysTest.test_dataset):
        groundTruth = int(datum['synth'].detach())
        ind = int(datum['_idx'])
        res_list.append([ind,groundTruth,results[i]])

    resdf = pd.DataFrame(res_list, columns=['testIndex','GT','pred_'+str(it)])  #GT is a duplicate
    resdf = resdf.set_index('testIndex').sort_index() 
        
    it_testdf = it_testdf[['material_id']] 
    it_testdf = it_testdf.merge(resdf['pred_'+str(it)],
                                left_index=True, right_index=True)
    iteration_results = iteration_results.merge(it_testdf,
                            left_on='material_id', right_on='material_id', how='outer')

    print("===the {}th iteration is done.".format(it))
    iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp'))   #overwriting results at each iteration
   
# %%
iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName))

# %%


# # %%
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = str(3) # use before loading lightning.gpu
# from pathlib import Path 
# import sys
# import numpy as np
# import random
# import schnetpack as spk
# from schnetpack.data import ASEAtomsData, BaseAtomsData, AtomsDataFormat, AtomsDataModule
# import schnetpack.transform as trn
# import torch
# from torch import nn
# import torchmetrics
# import pytorch_lightning as pl
# import json
# import pandas as pd
# import schnet.pu_learn.int2metric as int2metric
# # from schnet.pu_learn.Datamodule4PU import *
# # from schnet.pu_learn.schnet_funcs  import directory_setup, predProb
# # changed imports here for this directory, revert back to commented versions for git.
# from pu_learn.Datamodule4PU import *
# from pu_learn.schnet_funcs  import directory_setup, predProb
# import argparse
# # import time

# # %%
# parser = argparse.ArgumentParser(
#     description="Semi-Supervised ML for Synthesizability Prediction"
# )
# parser.add_argument(
#     "--experiment",
#     default="schnet0",
#     help="name of the experiment and corresponding config files.",
# )
# # !!!!!!!!!Change the default for small data after debugging!!!!!!
# parser.add_argument(
#     "--small_data",
#     # default=False,
#     # change this is final version.
#     default=True,
#     help="name of the experiment and corresponding config files.",
# )
# args = parser.parse_args(sys.argv[1:])
# experiment = args.experiment 
# small_data = args.small_data
# data_prefix = "small_" if small_data else ""
# # print(os.path.exists('data/clean_data/synth_small_data_constant/train_id_24.txt'))
# # time.sleep(20)

# # change the abs path below, uncomment.
# # schnet_config_dir = "schnet/schnet_configs"
# # %%
# test_strategy = 'constant'
# # test_strategy = 'dynamic'
# # change strategy manually, remove it in the final version.
# # changed this bit cos lazy
# experiment = "coSchAl1"
# schnet_config_dir = "/home/samariam/projects/synth/schnet/schnet_configs"
# config_path = os.path.join(schnet_config_dir, 'pu_config_'+experiment+'.json')
# with open(config_path, "r") as read_file:
#     print("Read Experiment configuration")
#     config = json.load(read_file)
    
# schnetDirectory = config["schnetDirectory"]
# # os.chdir(schnetDirectory)
# # commented out (changed) the above line for cleaner path management.
# # Might need to keep it, if other paths are broken.

# print(os.getcwd())
# # tensorboard command is tensorboard --logdir={experiment}_res_log/lightning_logs/
# # %%
# print(Path().resolve())  #current working directory
# print(Path().absolute()) #file path in jupyter
# print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# # time.sleep(600)
# # %%
# # ###quick settings
# cotraining = config["cotraining"]
# # changed here for dynamic/constant experimenmt
# # new_target = config["new_target"]
# new_target = f'{config["new_target"]}_{test_strategy}'
# epoch_num = config["epoch_num"]
# if small_data:
#     epoch_num = int(epoch_num*0.5)
# start_iter = config["start_iter"] #not sure about  directory setup for starting after 0.
# num_iter = config["num_iter"]
# batch_size = config["batch_size"]
# experiment = config["experiment"]
# # small_data = config["small_data"] #moved it to argparse
# # res_df_fileName = data_prefix+experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)
# res_df_fileName = f'{data_prefix}{experiment}_{str(start_iter)}_{str(num_iter)}ep{str(epoch_num)}'

# # prev_fileName = 'blahblah/res_df/poster_iter70ep15tmp'
# # save_dir = os.path.join(schnetDirectory,experiment+'_res_log')
# # wanna change the test strategy to small_data (or not small data) in the final version.
# save_dir = os.path.join(schnetDirectory,experiment+'_'+test_strategy+'_res_log')
# data_dir = config["data_dir"]
# res_dir = os.path.join(save_dir,'res_df')

# # %%
# np.random.seed(42)
# # %%
# # changed below for dynamic/constant experiment
# # crysdf = pd.read_pickle(os.path.join(data_dir, 'synthDF'))

# crysdf = pd.read_pickle(os.path.join(data_dir, f'{data_prefix}synthDF_{test_strategy}'))
# crysdf["TARGET"] = crysdf.synth.copy()
# # %%
# experiment_target_match = { #output_dir: training_label_column
#             'schnet0':'synth',
#             'coSchAl1':'alignn0',
#             'coSchAl2':'coAlSch1',
#                             }
# if small_data:
#     split_id_dir = experiment_target_match[experiment]+'_small_data_'+test_strategy
# else:
#     split_id_dir = experiment_target_match[experiment]+'_'+test_strategy
# split_id_path = os.path.join(data_dir, split_id_dir)
# # %%
# if cotraining:
#     print("I haven't yet modified the cotrain_labeling_schnet for synthdf!")
#     print("I should remove user input for running in the background.")
#     if new_target == None:
#         print("You forgot to assign new target column forcotraining!")
#         sys.exit()
#     else:
#         print(f'You have assigned "{new_target}" as cotraining target.')
#         crysdf["TARGET"] = crysdf[new_target].copy()
        
# #%% 
# #  also here, the test set should depend on Synth, not Target :/    
# crysdf["targets"] = crysdf.TARGET.map(lambda target: np.array(target).flatten())
# #we need the array to have the shape (1,), hence we use flatten()
# crysdf["targets"] = crysdf.targets.map(lambda target: dict(synth=np.array(target)) )     
# #The above changes targets fromat from array to dict with array val
# iteration_results = crysdf[["material_id", 'synth', 'TARGET']]
# # %%
# # num_experimental = crysdf.synth.sum()
# # num_theo = crysdf.shape[0]-num_experimental
# # experimental_test_num = int(0.1*min(num_experimental, num_theo))
# # positive_index = list(range(num_experimental))
# # random.seed(42)
# # random.shuffle(positive_index)
# # testdf1 = crysdf[crysdf.synth==1].iloc[positive_index[:experimental_test_num]]

# # crysdf = crysdf.drop(index=testdf1.index) # need to remove positive-test...

# # traindf1 = crysdf[crysdf.TARGET==1].sample(frac=1,random_state=123)
# # class_train_num = len(traindf1) #number of train-data from each class.

# # if small_data:
# #     traindf1 = traindf1.sample(frac = .05, random_state = 42)  #for quick computation and debugginng
# #     testdf1 = testdf1.sample(frac = .05, random_state = 43)  #for quick computation and debugginng
# # positivePredictionLength = len(testdf1)



# # trainLength = round(2*len(traindf1)*.8)-5 #The size of taining data is twice the number of the training data with either label.
# # valLength = round(2*len(traindf1)*.2)-5
# # innerTestLength = (2*len(traindf1))-(trainLength+valLength)   #Fatal error without internal test set.

# # %%
# cutoff = 5
# radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
# n_atom_basis = 30
# n_filters = 64
# dataDir = os.path.join(schnetDirectory,"schnetDatabases")
# # changed the database name below. Just uncomment and remove the lines below in the final version.
# # testDatapath =os.path.join(dataDir,experiment+'_test_dataset.db')
# # trainDataPath = os.path.join(dataDir,experiment+'_train_dataset.db')
# testDatapath =os.path.join(dataDir,experiment+test_strategy+'_test_dataset.db')
# trainDataPath = os.path.join(dataDir,experiment+test_strategy+'_train_dataset.db')
# bestModelPath = os.path.join(save_dir,'best_inference_model')
# pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms

# schnet = spk.representation.SchNet(
#     n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=3, radial_basis=radial_basis,    
#     cutoff_fn = spk.nn.CosineCutoff(cutoff),
# )

# pred_synth = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='synth')

# nnpot = spk.model.NeuralNetworkPotential(
#     representation=schnet,
#     input_modules=[pairwise_distance],
#     output_modules=[pred_synth],
# )

# output_synth = int2metric.ModelOutput4ACC(
#     name='synth',
#     loss_fn=torch.nn.BCEWithLogitsLoss(), 
#     loss_weight=1.,
#     metrics={
#         "Accur": torchmetrics.Accuracy("binary")   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
#     }
# )

# task = spk.task.AtomisticTask(
#     model=nnpot,
#     outputs=[output_synth],
#     optimizer_cls=torch.optim.AdamW,
#     optimizer_args={"lr": 1e-4}
# )

# converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
# # %%
# for it in range(start_iter, num_iter):      
        
#     print('we started iteration {}'.format(it))
#     splitFilestring = directory_setup(res_dir = res_dir, 
#                                       dataPath = trainDataPath, save_dir = save_dir,
#                                       bestModelPath= bestModelPath, iteration_num=it)
    
#     np.random.seed(it)
    
#     # traindf0 = crysdf[crysdf.TARGET==0].sample(n=class_train_num,random_state=it) #a different 'negative' train-set at each iteration.
#     # testdf0 = crysdf[crysdf.TARGET==0].drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
#     # if small_data:
#     #     traindf0 = traindf0.sample(frac = .05, random_state = 44)
#     #     testdf0 = testdf0.sample(frac = .05, random_state = 44)
        
#     # it_traindf = pd.concat([traindf0,traindf1])
#     # it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
# # !we need to read the train/test_id_path s below:    
#     train_id_path = os.path.join(split_id_path, f'train_id_{it}.txt')
#     test_id_path = os.path.join(split_id_path, f'test_id_{it}.txt')
    
#     with open(train_id_path, "r") as f:
#         id_val_train = [int(line.strip()) for line in f]
    
#     with open(test_id_path, "r") as f2:
#         id_test = [int(line.strip()) for line in f2]
    
#     it_traindf = crysdf.loc[id_val_train]
#     it_testdf = crysdf.loc[id_test]
        
#     # it_traindf = it_traindf.sample(frac=1,random_state=it+1, ignore_index=True)
#     # it_testdf = it_testdf.sample(frac=1,random_state=it+2, ignore_index=True)
#     # the above was changed; already shuffled in pu_data_selection.
#     need to change this validation length accordingly
#     valLength = round(len(it_traindf)*.2)-5
#     trainLength = round(len(it_traindf)*.8)-5 #The size of taining data is twice the number of the training data with either label.
#     innerTestLength = len(it_traindf)-(trainLength+valLength)   #Fatal error without internal test set.

#     positivePredictionLength = it_testdf.TARGET.sum()
#     unlabeledPredictionLength = len(it_testdf)-positivePredictionLength
#     testLength = len(it_testdf)

#     # del traindf0, testdf0 

#     it_traindf = it_traindf.sample(frac=1,random_state=it, ignore_index=True) #shuffling for each iteration.
#     # change here, reset index according to stackoverflow.
#     it_traindf.reset_index(drop=True, inplace=True)
#     it_testdf.reset_index(drop=True, inplace=True)
    
#     if it==1:
#         print('The #training data is {}, #validation data {} and #internal test data {}. '.format(trainLength, valLength, innerTestLength))
#         print(f"The total number of test-set (predictions) is {testLength}, out of which {unlabeledPredictionLength} are unlabeled\
#  and {positivePredictionLength} are labeled positive.")

#     class_dataset = ASEAtomsData.create(trainDataPath, 
#                                     distance_unit='Ang',
#                                     property_unit_dict={'synth':int(1)}     #The unit is int(1); aka unitless.                             
#                                      )
#     print('adding systems to dataset')
#     class_dataset.add_systems(np.array(it_traindf.targets), np.array(it_traindf.atoms))  
#     print('creating data module')
#     crysData = AtomsDataModule(datapath=trainDataPath,
#                    batch_size=batch_size,
#                     num_train=trainLength,
#                     num_val=valLength,
#                     transforms=[
#                         trn.ASENeighborList(cutoff=5.),
#                         trn.CastTo32(), 
#                                 ],
#                     property_units={'synth':int(1)},
#                     num_workers=4,    
#                     split_file = splitFilestring, 
#                     pin_memory=True, # set to false, when not using a GPU
#                     load_properties=['synth'], 
#     )
    
#     crysData.prepare_data()
#     crysData.setup()
    
#     splitFilestringTest = directory_setup(res_dir = res_dir, 
#                                           dataPath = testDatapath,save_dir = save_dir, 
#                                           bestModelPath= bestModelPath, iteration_num=it)

#     test_dataset = ASEAtomsData.create(testDatapath, 
#                                     distance_unit='Ang',
#                                     property_unit_dict={
#                                         'synth':int(1),
#                                         })
#     print('adding systems to the test dataset')
#     test_dataset.add_systems(np.array(it_testdf.targets), np.array(it_testdf.atoms))  

#     print('creating data module')

#     crysTest = DataModuleWithPred(
#                     datapath=testDatapath,
#                     batch_size=batch_size,
#                     num_train=0,
#                     num_val=0, 
#                     num_test=len(it_testdf),
#                     transforms=[
#                         trn.ASENeighborList(cutoff=5.),
#                         trn.CastTo32(), 
#                                 ],
#                     property_units={'synth':int(1)},
#                     num_workers=4,
#                     split_file = splitFilestringTest, 
#                     pin_memory=True, # set to false, when not using a GPU
#                     load_properties=['synth'], 
#                                 )

#     crysTest.prepare_data()
#     crysTest.setup("test")
    
#     means, stddevs = crysData.get_stats(
#     'synth', divide_by_atoms=True, remove_atomref=True)
    
#     meansTest, stddevsTest = crysTest.get_stats(
#     'synth', divide_by_atoms=True, remove_atomref=True,
#     mode = 'test')
    
#     print('Mean atomization energy / atom:', means.item())
#     print('Std. dev. atomization energy / atom:', stddevs.item())
#     # This doesn't work when no test data is given.    
#     logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
#     callbacks = [
#         # early_stopping,
#     spk.train.ModelCheckpoint(
#         inference_path=os.path.join(save_dir, "best_inference_model"),
#         save_top_k=1,
#         # monitor="val_loss"
#         monitor="val_synth_Accur"
#     )
#     ]
       

#     trainer = pl.Trainer(
#     accelerator='gpu',
#     gpus=1,
#     auto_select_gpus = True,
#     strategy=None, 
#     precision=16,
#     callbacks=callbacks,
#     logger=logger,
#     default_root_dir=save_dir,
#     max_epochs=epoch_num, 
#     )

#     trainer.fit(task, datamodule=crysData)
    
#     myaccuracy = trainer.callback_metrics["val_synth_Accur"]
#     print(myaccuracy)
    
#     predictions = trainer.predict(model=task, 
#                     dataloaders= crysTest.predict_dataloader(),
#                     return_predictions=True)

#     # %%
#     results = []
#     for batch in predictions:    
#         for datum in batch['synth']:
#             results = results+[predProb(datum.float())]
            
#     res_list = []
#     for i, datum in enumerate(crysTest.test_dataset):
#         groundTruth = int(datum['synth'].detach())
#         ind = int(datum['_idx'])
#         res_list.append([ind,groundTruth,results[i]])

#     resdf = pd.DataFrame(res_list, columns=['testIndex','GT','pred_'+str(it)])  #GT is a duplicate
#     resdf = resdf.set_index('testIndex').sort_index() 
        
#     it_testdf = it_testdf[['material_id']] 
#     it_testdf = it_testdf.merge(resdf['pred_'+str(it)],
#                                 left_index=True, right_index=True)
#     iteration_results = iteration_results.merge(it_testdf,
#                             left_on='material_id', right_on='material_id', how='outer')

#     print("===the {}th iteration is done.".format(it))
#     iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp'))   #overwriting results at each iteration
   
# # %%
# iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName))

# # %%
