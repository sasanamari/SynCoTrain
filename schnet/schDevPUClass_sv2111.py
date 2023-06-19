# %%
# need to upgrade a library to have all the needed versions of the code.
# pandas version in sch211 was 1.2.4
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0) # safer to use before loading lightning.gpu
from pathlib import Path  #recommended path library for python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import schnetpack as spk
from schnetpack.data import ASEAtomsData, BaseAtomsData, AtomsDataFormat, AtomsDataModule
import schnetpack.transform as trn
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
import json
import pandas as pd
import time
from datetime import timedelta
import int2metric
from Datamodule4PU import *
# from Datamodule4PUTrial import *
from pytorch_lightning.callbacks import EarlyStopping
import datetime
# from synth.data_scripts.cotraining_labeling import cotrain_labeling_schnet

# %%
# current_config = "25runTest_config.json"
# current_config = "debugsch_config.json"
# current_config = "longDebug_config.json"
# current_config = "100runs_config.json"
# current_config = "3runs_config.json"
# current_config = "cotrain_debug_config.json"
current_config = "coSchAl1_config.json"

saveDfs = False
startTime = time.time()

# remove saveDFs parts if the code works fine.
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
print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
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
small_data = config["small_data"]
res_df_fileName = experiment+"_"+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)

# prev_fileName = 'blahblah/res_df/poster_iter70ep15tmp'
save_dir = os.path.join(schnetDirectory,experiment+'_res_log')
fulldatapath = config["fulldatapath"]
res_dir = os.path.join(save_dir,'res_df')
# %%
def directory_setup(dataPath,save_dir, bestModelPath, iteration_num=None):
    if iteration_num == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Logging directory was created.')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            print('Result directory was created.')
        
    splitFile_path = Path('split.npz') #should this be split.lock?
    try:
        splitFile_path.unlink()
    except OSError as e:
        print(e)
        splitFile_path = Path(os.path.join(save_dir,str(splitFile_path)))
        try:
            splitFile_path.unlink()
        except OSError as e:
            print(e)
            
    datapathObj = Path(dataPath)
    try:
        datapathObj.unlink()
        print('unlinked')
    except OSError as e:
        print(e)        
        
    bestModelPath_obj = Path(bestModelPath)
    try:
        bestModelPath_obj.unlink()
    except OSError as e:
        print(e)       
         
    return str(splitFile_path)



# %%
print(torch.cuda.is_available())
np.random.seed(42)

# %%
crysdf = pd.read_pickle(fulldatapath)
crysdf["TARGET"] = crysdf.synth.copy()
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
########################
# ###clean this after this run! This is for consistency!
# I kept the same random selection as before. In the future,
# We should first pick the experimental part of the test-set. Then, 
# we'll choose the train-set based on current labels of the REMAINING data.
num_pos = crysdf.synth.sum()
num_theo = crysdf.shape[0]-num_pos
class_train_num = int(0.9*min(num_pos, num_theo))
traindf1 = crysdf[crysdf.synth==1].sample(n=class_train_num,random_state=123) #the positive data for training
testdf1 = crysdf[crysdf.synth==1].drop(index=traindf1.index)

del num_pos, num_theo, class_train_num, traindf1

crysdf = crysdf.drop(index=testdf1.index) # need to remove positive-test...

# num_pos = crysdf.TARGET.sum()
# num_theo = crysdf.shape[0]-num_pos
# class_train_num = int(0.9*min(num_pos, num_theo))
traindf1 = crysdf[crysdf.TARGET==1].sample(frac=1,random_state=123)
class_train_num = len(traindf1) #number of train-data from each class.
# when you're done, uncomment/modify the remaining code below!
# #####################


# num_pos = crysdf.TARGET.sum()
# num_theo = crysdf.shape[0]-num_pos
# class_train_num = int(0.9*min(num_pos, num_theo)) #positive class is the smaller class
# # We separate test before iterating begins.
# # We can do this in the loop, if we want predictions for all the crystals.
# # The random_state of sampling then should change with the iteration number.
# traindf1 = crysdf[crysdf.TARGET==1].sample(n=class_train_num,random_state=123) #the positive data for training
# testdf1 = crysdf[crysdf.TARGET==1].drop(index=traindf1.index)  #the positive data for testing
if small_data:
    traindf1 = traindf1.sample(frac = .05, random_state = 42)  #for quick computation and debugginng
    testdf1 = testdf1.sample(frac = .05, random_state = 43)  #for quick computation and debugginng
positivePredictionLength = len(testdf1)



trainLength = round(2*len(traindf1)*.8)-5 #The size of taining data is twice the number of the training data with either label.
valLength = round(2*len(traindf1)*.2)-5
innerTestLength = (2*len(traindf1))-(trainLength+valLength)   #Fatal error without internal test set.

# %%
cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64
dataDir = os.path.join(schnetDirectory,"schnetDatabases")
testDatapath =os.path.join(dataDir,experiment+'_test_dataset.db')
trainDataPath = os.path.join(dataDir,experiment+'_train_dataset.db')
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
def predProb(score): 
    """returns class label from network score"""
    prob = nn.Sigmoid()
    pred_prob = prob(score)     
    if 0<=pred_prob< 0.5:
        return 0
    else:
        return 1
# %%
if saveDfs:
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    crysdf.to_pickle(os.path.join(res_dir, "crysdf_bi"))
# %%
for it in range(start_iter, num_iter):
    # if it==start_iter and it!=0:
        # !crysdf = pd.read_pickle(prev_fileName)
        # we don't work with crysdf within the loop.
        # crysdf.target_pd = crysdf.target_pd.map(lambda x: int(x[1]))  #some it's string, changing back to int (it was because of csv)
        
        
    st = time.time()
    print('we started iteration {}'.format(it))
    splitFilestring = directory_setup(dataPath = trainDataPath,save_dir = save_dir, 
                                      bestModelPath= bestModelPath, iteration_num=it)
    
    np.random.seed(it)
    
    traindf0 = crysdf[crysdf.TARGET==0].sample(n=class_train_num,random_state=it) #a different 'negative' train-set at each iteration.
    testdf0 = crysdf[crysdf.TARGET==0].drop(index=traindf0.index) #The remaining unlabeled data to be labeled.   
    if small_data:
        traindf0 = traindf0.sample(frac = .05, random_state = 44)
        testdf0 = testdf0.sample(frac = .05, random_state = 44)
        
    it_traindf = pd.concat([traindf0,traindf1])
    it_testdf = pd.concat([testdf0,testdf1]) #positive test and unlabled prediction.
    
    it_traindf = it_traindf.sample(frac=1,random_state=it+1, ignore_index=True)
    it_testdf = it_testdf.sample(frac=1,random_state=it+2, ignore_index=True)

    unlabeledPredictionLength = len(testdf0)
    testLength = len(it_testdf)

    del traindf0, testdf0 #, df1, testdf1


    it_traindf = it_traindf.sample(frac=1,random_state=it, ignore_index=True) #shuffling for each iteration.
    if it==0:
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
    
    splitFilestringTest = directory_setup(dataPath = testDatapath,save_dir = save_dir, 
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
    # This doesn't work when no test data is given, and it has no docsAccurtring. Does it calculate the mean and and std of test data?
    
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
    strategy=None, # or 'ddp'?
    precision=16,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=save_dir,
    max_epochs=epoch_num, 
    )

    trainer.fit(task, datamodule=crysData)
    
    myaccuracy = trainer.callback_metrics["val_synth_Accur"]
    print(myaccuracy)
    
    t = trainer.predict(model=task, 
                    # datamodule=crysTest,
                    dataloaders= crysTest.predict_dataloader(),
                    return_predictions=True)

    # %%
    results = []
    for batch in t:    
        for datum in batch['synth']:
            results = results+[predProb(datum.float())]
            

    mid = []
    for i, datum in enumerate(crysTest.test_dataset):
        groundTruth = int(datum['synth'].detach())
        ind = int(datum['_idx'])
        mid.append([ind,groundTruth,results[i]])

    resdf = pd.DataFrame(mid, columns=['testIndex','GT','pred_'+str(it)])  #GT is a duplicate
    resdf = resdf.set_index('testIndex').sort_index() #new line
    if saveDfs:
        resdf.to_pickle(os.path.join(res_dir, 'resdf_'+str(it)))
        it_testdf.to_pickle(os.path.join(res_dir, 'it_testdf')+str(it))
        
    it_testdf = it_testdf[['material_id']] #new line
    it_testdf = it_testdf.merge(resdf['pred_'+str(it)],
                                left_index=True, right_index=True)
    iteration_results = iteration_results.merge(it_testdf,
                            left_on='material_id', right_on='material_id', how='outer')
    # it_testdf = it_testdf.merge(resdf, left_index=True, right_on='testIndex')
    # it_testdf = it_testdf[['material_id','pred_'+str(it)]]
    # crysdf = pd.merge(left = crysdf, right=it_testdf,
    #         how = 'outer',left_on='material_id', right_on='material_id')
    
    # crysdf = crysdf.merge(resdf[['pred_'+str(it), 'GT']], left_index=True, right_index=True, how = 'outer')

    et = time.time()
    itt = et-st
    print("===the {}th iteration took  minutes{} to run===".format(it, timedelta(seconds=itt)//60))
    # !choose smaller data for quicker debugging. just pick a small subset.
    # iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp_'+str(it)))   #saving results at each iteration
    iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp'))   #overwriting results at each iteration
    # crysdf.to_pickle(os.path.join(res_dir,res_df_fileName+'tmp'))   #saving results at each iteration
    # it_testdf.to_pickle(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration
    # I chanegd crysdf to it_test_df!!!
    try:  #just reporting progress in a small text file.
        deltaT = time.time() - startTime
        avgIterTime = deltaT/(it+1)
        estimateRemainSeconds = (num_iter - it) * avgIterTime
        estimateRemainTime = timedelta(seconds = estimateRemainSeconds)
        remain_days = estimateRemainTime.days
        remain_hours = estimateRemainTime.seconds//3600


        with open("iterReport"+experiment+".txt", "w") as text_file:
            print(f"So far we have finished iteration #{it}.", file=text_file)
            print(F"Estimated remaining time is {remain_days} days and {remain_hours} hours.", file=text_file)
    except:
        pass
    

# %%
# crysdf.target_pd = crysdf.target_pd.map(lambda x: int(x[0])) #changing array to target for easier handling
# !I saved only the test results. What is it I want?
# !we should change target_pd here!
iteration_results.to_pickle(os.path.join(res_dir,res_df_fileName))

# %%
endTime = time.time()
elapsed_time = endTime-startTime
print("This script took {} to run.".format(timedelta(seconds=elapsed_time)))
# !also fix the timing in BOTH scripts
# %%



