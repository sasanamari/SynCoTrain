# %%
import os
from pathlib import Path 
import sys
import argparse
from experiment_setup import current_setup, str_to_bool

parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--experiment",
    default="schnet0",
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
parser.add_argument(
    "--startIt", 
    type=int, 
    default=0, 
    help="Starting iteration No.")
parser.add_argument(
    "--gpu_id", 
    type=int, 
    default=3, 
    help="GPU ID to use for training.")

args = parser.parse_args(sys.argv[1:])
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) # use before loading lightning.gpu
experiment = args.experiment 
ehull015 = args.ehull015
small_data = args.small_data
startIt = args.startIt
# schnettest  =args.schnettest
# ntest = args.ntest

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
from pu_schnet.pu_learn.Datamodule4PU import *
from pu_schnet.pu_learn.schnet_funcs  import directory_setup, predProb
import time
from pytorch_lightning.callbacks import EarlyStopping

cs = current_setup(small_data=small_data, experiment=experiment, ehull015 = ehull015)
                #    schnettest=schnettest)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]
print(f"Running the {data_prefix}{experiment} experiment for the {prop} property.")
start_time = time.time()
# %%
config_path = 'pu_schnet/schnet_configs/pu_config_schnetpack.json'
# if half_way_iteration:
#     config_path = 'pu_schnet/schnet_configs/pu_config_schnetpack_hw.json'
# if schnettest:
#     config_path = 'pu_schnet/schnet_configs/pu_config_schnetpackTest.json'

with open(config_path, "r") as read_file:
    print("Read Experiment configuration")
    config = json.load(read_file)
schnetDirectory = config["schnetDirectory"]
# %%
print(Path().resolve())  #current working directory
print(Path().absolute()) #file path in jupyter
print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# %%
epoch_num = config["epoch_num"]
if small_data:
    epoch_num = int(epoch_num*0.5)
config["start_iter"] = startIt #For consistency 
start_iter = config["start_iter"] 
num_iter = config["num_iter"]
batch_size = config["batch_size"]

res_df_fileName = f'{data_prefix}{experiment}_{str(start_iter)}_{str(num_iter)}ep{str(epoch_num)}'

# save_dir = os.path.join(schnetDirectory,f'PUOutTest_{ntest}_{data_prefix}{experiment}')
save_dir = os.path.join(schnetDirectory,f'PUOutput_{data_prefix}{experiment}')
if ehull015:
    save_dir = os.path.join(schnetDirectory,f'PUehull015_{experiment}')
data_dir = config["data_dir"]
res_dir = os.path.join(save_dir,'res_df')
# %%
np.random.seed(42)
crysdf = pd.read_pickle(propDFpath) 
# %%
split_id_dir = f"{data_prefix}{TARGET}_{prop}"
split_id_dir_path = os.path.join(data_dir, split_id_dir)        
#%% 
crysdf["targets"] = crysdf[TARGET].map(lambda target: np.array(target).flatten())
#we need the array to have the shape (1,), hence we use flatten()
crysdf["targets"] = crysdf.targets.map(lambda target: {prop: np.array(target)})  
#The above changes targets fromat from array to dict with array val
# if half_way_iteration:    
if startIt != 0:
    iteration_results = pd.read_pickle(os.path.join(res_dir,f'{data_prefix}{experiment}_0_{str(num_iter)}ep{str(epoch_num)}'+'tmp'))
else:
    iteration_results = crysdf[["material_id", prop, TARGET]]
    iteration_results = iteration_results.loc[:, ~iteration_results.columns.duplicated()] # drops duplicated props at round zero.
# %%
cutoff = 5
n_rbf = 30
n_atom_basis = 64 #from the figure in their paper
n_filters = 64
n_interactions = 3
lr = 1e-3
# dataDir = os.path.join(schnetDirectory,"schnetDatabases")
# %%
for it in range(start_iter, num_iter):      
# also add ntest         
    print('we started iteration {}'.format(it))
    np.random.seed(it) 
    # scheduler = {'scheduler_cls':None,'scheduler_args':None}
    scheduler = {'scheduler_cls':torch.optim.lr_scheduler.ReduceLROnPlateau,
                 'scheduler_args':{"mode": "max", #mode is min for loss, max for merit
                               "factor": 0.5,
                               "patience": 15,
                               "threshold": 0.01,
                               "min_lr": 1e-6
                               },
                #  'scheduler_monitor':f"val_{prop}_recalll" 
                 'scheduler_monitor':f"val_{prop}_Accuracy"
                #  'scheduler_monitor':"val_loss"
                }

    save_it_dir = os.path.join(save_dir, f'iter_{it}')
    dataDir = os.path.join(save_it_dir,"schnetDatabases")
    

    testDatapath =os.path.join(dataDir,f"{data_prefix}{experiment}_{prop}_test_dataset.db")
    trainDataPath = os.path.join(dataDir,f"{data_prefix}{experiment}_{prop}_train_dataset.db")
    # testDatapath =os.path.join(dataDir,f"{data_prefix}{experiment}_{prop}_{ntest}_test_dataset.db")
    # trainDataPath = os.path.join(dataDir,f"{data_prefix}{experiment}_{prop}_{ntest}_train_dataset.db")
    bestModelPath = os.path.join(save_it_dir,'best_inference_model')
    
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff) # for 0.1 Ang distance
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=n_interactions, radial_basis=radial_basis,    
        cutoff_fn = spk.nn.CosineCutoff(cutoff),
    )

    pred_prop = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=prop)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_prop],
    )

    output_prop = int2metric.ModelOutput4ACC(
        name=prop,
        loss_fn=torch.nn.BCEWithLogitsLoss(), 
        loss_weight=1.,
        metrics={
            "Accuracy": torchmetrics.Accuracy("binary"),   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
            "recalll": torchmetrics.Recall("binary")   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_prop],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": lr}, #based on their supplemet info for materials project
        scheduler_monitor=scheduler['scheduler_monitor'],
        scheduler_cls=scheduler['scheduler_cls'],
        scheduler_args=scheduler['scheduler_args'],
    )

    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
    
    splitFilestring = directory_setup(res_dir = res_dir, 
                                      dataPath = trainDataPath, save_dir = save_it_dir,
                                      bestModelPath= bestModelPath,)# iteration_num=it)
    
    train_id_path = os.path.join(split_id_dir_path, f'train_id_{it}.txt')
    test_id_path = os.path.join(split_id_dir_path, f'test_id_{it}.txt')
    experimentalDataSize_path = os.path.join(split_id_dir_path, 'experimentalDataSize.txt')
    with open(train_id_path, "r") as f:
        id_val_train = [int(line.strip()) for line in f]
    
    with open(test_id_path, "r") as f2:
        id_test = [int(line.strip()) for line in f2]
        
    with open(experimentalDataSize_path, "r") as f3:
        experimentalDataSize = [int(float(line.strip())) for line in f3][0]        
    
    it_traindf = crysdf.loc[id_val_train]
    it_testdf = crysdf.loc[id_test]
    total_data_length = len(it_testdf)+len(it_traindf)
    
    valLength = int(len(it_traindf)*.1)-5
    trainLength = int(len(it_traindf)*.9)-5 
    innerTestLength = len(it_traindf)-(trainLength+valLength)   #Fatal error without internal test set.
    
    positivePredictionLength = it_testdf[TARGET].sum()
    unlabeledPredictionLength = len(it_testdf)-positivePredictionLength
    testLength = len(it_testdf)

    it_traindf = it_traindf.sample(frac=1,random_state=it, ignore_index=True) #shuffling for each iteration.
    it_traindf.reset_index(drop=True, inplace=True)
    it_testdf.reset_index(drop=True, inplace=True)
    
    if it==1:
        print('The #training data is {}, #validation data {} and #internal test data {}. '.format(trainLength, valLength, innerTestLength))
        print(f"The total number of test-set (predictions) is {testLength}, out of which {unlabeledPredictionLength} are unlabeled\
 and {positivePredictionLength} are labeled positive.")

    class_dataset = ASEAtomsData.create(trainDataPath, 
                                    distance_unit='Ang',
                                    property_unit_dict={prop:int(1)}     #The unit is int(1); aka unitless.                             
                                     )
    print('adding systems to dataset')
    class_dataset.add_systems(np.array(it_traindf.targets), np.array(it_traindf.atoms))  
    print('creating data module')
    crysData = AtomsDataModule(datapath=trainDataPath,
                   batch_size=batch_size,
                    num_train=trainLength,
                    num_val=valLength,
                    transforms=[
                        trn.ASENeighborList(cutoff=float(cutoff)),
                        trn.CastTo32(), 
                                ],
                    property_units={prop:int(1)},
                    num_workers=4,    
                    split_file = splitFilestring, 
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=[prop], 
    )
    
    crysData.prepare_data()
    crysData.setup()
    
    splitFilestringTest = directory_setup(res_dir = res_dir, 
                                          dataPath = testDatapath,save_dir = save_it_dir, 
                                          bestModelPath= bestModelPath,)# iteration_num=it)

    test_dataset = ASEAtomsData.create(testDatapath, 
                                    distance_unit='Ang',
                                    property_unit_dict={
                                        prop:int(1),
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
                        trn.ASENeighborList(cutoff=float(cutoff)),
                        trn.CastTo32(), 
                                ],
                    property_units={prop:int(1)},
                    num_workers=4,
                    split_file = splitFilestringTest, 
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=[prop], 
                                )

    crysTest.prepare_data()
    crysTest.setup("test")
    
    means, stddevs = crysData.get_stats(
    prop, divide_by_atoms=True, remove_atomref=True)
    
    meansTest, stddevsTest = crysTest.get_stats(
    prop, divide_by_atoms=True, remove_atomref=True,
    mode = 'test')
    
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())
    # This doesn't work when no test data is given.    
    early_stopping = EarlyStopping(
    verbose=2,
    mode= 'max', #min for loss, max for merit.
    monitor=f"val_{prop}_Accuracy",  #if it works, also change in ModelCheckpoint?
    # monitor=f"val_{prop}_recalll",  #if it works, also change in ModelCheckpoint?
    # monitor="val_loss",  #if it works, also change in ModelCheckpoint?
    # rhe error says"Pass in or modify your `EarlyStopping` callback to use any of the following: `train_loss`, `val_loss`, `val_synth_Accur`, `train_synth_Accur`""
    min_delta=0.02,
    patience=30,
)    
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    callbacks = [
        early_stopping,
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(save_it_dir, "best_inference_model"),
        save_top_k=1,
        # monitor="nonesense2"
        # monitor="val_loss"
        monitor=f"val_{prop}_Accuracy"
        # monitor=f"val_{prop}_recalll"
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
    default_root_dir=save_it_dir, 
    max_epochs=epoch_num, 
    )

    trainer.fit(task, datamodule=crysData)
    
    # myaccuracy = trainer.callback_metrics[f"val_{prop}_Accur"]
    # myloss = trainer.callback_metrics["val_loss"]
    # print(myloss)
    
    predictions = trainer.predict(model=task, 
                    dataloaders= crysTest.predict_dataloader(),
                    return_predictions=True)

    results = []
    for batch in predictions:    
        for datum in batch[prop]:
            results = results+[predProb(datum.float())]
            
    res_list = []
    for i, datum in enumerate(crysTest.test_dataset):
        groundTruth = int(datum[prop].detach())
        ind = int(datum['_idx'])
        res_list.append([ind,groundTruth,results[i]])

    resdf = pd.DataFrame(res_list, columns=['testIndex','GT','pred_'+str(it)])  #GT is a duplicate
    resdf = resdf.set_index('testIndex').sort_index() 
        
    it_testdf = it_testdf[['material_id']] 
    it_testdf = it_testdf.merge(resdf['pred_'+str(it)],
                                left_index=True, right_index=True)
    iteration_results = iteration_results.merge(it_testdf,
                            left_on='material_id', right_on='material_id', how='outer')
    
    try:
        os.remove(testDatapath)
        os.remove(trainDataPath)
        print("File removed successfully!")
    except Exception as e:
        print("An error occurred:", str(e))
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("===the {}th iteration is done.".format(it))
    iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp'))   #overwriting results at each iteration
    elapsed_time = time.time() - start_time
    remaining_iterations = num_iter - it - 1
    time_per_iteration = elapsed_time / (it - start_iter + 1)
    estimated_remaining_time = remaining_iterations * time_per_iteration
    remaining_days = int(estimated_remaining_time // (24 * 3600))
    remaining_hours = int((estimated_remaining_time % (24 * 3600)) // 3600)
    
    time_log_path = os.path.join('time_logs',f'schnet_remaining_time_{data_prefix}{experiment}_{prop}.txt')
    with open(time_log_path, 'w') as file:
        file.write(f"Iterations completed: {it - start_iter}\n")
        file.write(f"Iterations remaining: {remaining_iterations}\n")
        file.write(f"Estimated remaining time: {remaining_days} days, {remaining_hours} hours\n")

    print(f"Iteration {it} completed. Remaining time: {remaining_days} days, {remaining_hours} hours")   
# %%
iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName))

# %%
# Final summary
elapsed_days = int(elapsed_time // (24 * 3600))
elapsed_hours = int((elapsed_time % (24 * 3600)) // 3600)

with open(time_log_path, 'w') as file:
    file.write(f"Iterations completed: {num_iter - start_iter}\n")
    file.write(f"Total time taken: {elapsed_days} days, {elapsed_hours} hours\n")

print(f"PU Learning completed. Total time taken: {elapsed_days} days, {elapsed_hours} hours")

# %%