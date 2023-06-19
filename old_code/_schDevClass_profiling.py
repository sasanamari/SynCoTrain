# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2,3"
import sys
# os.chdir(os.path.dirname(sys.argv[0]))
from pathlib import Path  #recommended path library for python3
import numpy as np
import matplotlib.pyplot as plt
import schnetpack as spk
from schnetpack.data import ASEAtomsData, BaseAtomsData, AtomsDataFormat, AtomsDataModule
import schnetpack.transform as trn
import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler, AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping

import pickle
import pandas as pd
import time
from datetime import timedelta
import int2metric
from Datamodule4PU import *
# %%
startTime = time.time()

# %%
print(Path().resolve())  #current working directory
print(Path().absolute()) #file path in jupyter
print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script
# %%
# ###quick settings
# res_df_fileName = 'crysClassDfLogits.csv'
epoch_num = 15
start_iter = 0 #not sure about  directory setup for starting after 0.
num_iter = 75
res_df_fileName = 'profiling'+str(start_iter)+'_'+str(num_iter)+'ep'+str(epoch_num)

######################################## USE EARLY STOP!!! ########################################

# prev_fileName = 'qm9tut/res_df/poster_iter70ep15tmp'

theoretical_path = "data_for_dev_try/theoretical/"
experimental_path = "data_for_dev_try/experimental/"

cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64
testdatapath ='./class_profile_dataset_test4.db'
dataPath = './class_profile_dataset4.db'
save_dir = './profiling'
bestModelPath = save_dir+'/'+'best_inference_model'

# %%
def directory_setup(dataPath,save_dir, bestModelPath, iteration_num=None):
    if iteration_num == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Logging directory was created.')
        if not os.path.exists(save_dir+'/res_df'):
            os.makedirs(save_dir+'/res_df')
            print('Result directory was created.')
        
    splitFile_path = Path('split.npz')
    try:
        splitFile_path.unlink()
    except OSError as e:
        print(e)
        splitFile_path = Path(save_dir+'/'+str(splitFile_path))
        try:
            splitFile_path.unlink()
        except OSError as e:
            print(e)
            
    datapathObj = Path(dataPath)
    try:
        datapathObj.unlink()
        print('datapath was unlinked')
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
# torch.device('cuda')     #gpu was busy

# %%
currentDir = str(Path(__file__).parent.resolve())+'/'
np.random.seed(42)
pAtoms = np.load(currentDir+experimental_path+"PosAtomsUnd12arr.npy", allow_pickle=True)
tAtoms = np.load(currentDir+theoretical_path+"TheoAtomsUnd12arr.npy", allow_pickle=True)
data_types = list(tAtoms[0].keys())
# %%
pSynth = [np.array(1).flatten()]*len(pAtoms)    #we need the array to have the shape (1,), hence we use flatten()
tSynth = [np.array(0).flatten()]*len(tAtoms)

# %%
crysSynthData = np.concatenate([pAtoms, tAtoms])
targetData = [*pSynth, *tSynth]  
crysdf = pd.DataFrame()
for colName in data_types:
    crysdf[colName]  =[datum[colName] for datum in crysSynthData]

crysdf['target_pd'] = targetData
if targetData[0].shape !=(1,):
    print("Target data has the wrong shape of ", targetData[0].shape)
    # break
crysdf["targets"] = crysdf.target_pd.map(lambda crystalClass: dict(synth=np.array(crystalClass)) )     #changes targets fromat from array to dict with array val
crysdf = crysdf.reset_index(drop=False)
crysdf = crysdf.rename(columns={'index':'crystal_id'})
crysdf = crysdf.sample(frac=1, random_state=42).reset_index(drop=True)     #simply shuffles the rows of positive and Unlabeled(negative) data


# %%

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
    # postprocessors=[trn.CastTo64(), trn.AddOffsets('synth', 
                                                #    add_mean=True, 
                                                #    add_atomrefs=False)]  
)


output_synth = int2metric.ModelOutput4ACC(
    name='synth',
    loss_fn=torch.nn.BCEWithLogitsLoss(), 
    loss_weight=1.,
    metrics={
        "Accur": torchmetrics.Accuracy()   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
    }
)


task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_synth],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-5}
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
def pred_test(x, best_model,returnScore = False): 
    """Predicts class label for test data given a (best) model"""     
    inputs = converter(x)
    pred = best_model(inputs)
    score = pred['synth']
    classLabel = predProb(score)
    if returnScore:
        return pred
    return classLabel
    # pred_arr = next(iter(pred.values())).detach().numpy()    #dict_value to array



# %%
# ###if we sample without replacement, it'll be a wider search, but less robust for each datum.
# for it in range(start_iter, num_iter):
# it=0

for it in range(start_iter,num_iter):
    if it==start_iter and it!=0:
        print('first Iter, do you wanna load previous iters?')
        # crysdf = pd.read_pickle(prev_fileName)
        # crysdf.target_pd = crysdf.target_pd.map(lambda x: int(x[1]))  #some it's string, changing back to int (it was because of csv)
        
        
    st = time.time()
    print('we started iteration {}'.format(it))
    splitFilestring = directory_setup(dataPath = dataPath,save_dir = save_dir, bestModelPath= bestModelPath, iteration_num=it)

    np.random.seed(it)
    crysdf = crysdf.sample(frac=1, random_state=it).reset_index(drop=True)     #simply shuffles the rows of positive and Unlabeled(negative) data

    num_pos = crysdf.target_pd.sum()
    num_theo = crysdf.shape[0]-num_pos

    pos_train_num = int(0.9*num_pos)
    theo_train_num = int(0.85*num_theo)

    df1 = crysdf[crysdf.target_pd==1].sample(n=pos_train_num,random_state=it)
    df0 = crysdf[crysdf.target_pd==0].sample(n=theo_train_num,random_state=it+1)

    it_train_df = pd.concat([df0,df1]).reset_index(drop=True)
    it_train_df = it_train_df.sample(frac=1,random_state=it+2)

    theoTestDf = crysdf[crysdf.target_pd==0].drop(index=df0.index)    #how to mask for selecting theoretical data
    posTestDf = crysdf[crysdf.target_pd==1].drop(index=df1.index)    #how to mask for selecting positive data

    it_test_df = pd.concat([theoTestDf,posTestDf])
    it_test_df = it_test_df.sample(frac=1,random_state=it+2).reset_index(drop=True)

    trainLength = round(len(it_train_df)*.8)-3
    valLength = round(len(it_train_df)*.2)-3
    innerTestLength = len(it_train_df)-(trainLength+valLength)   #We do our own testing manually later on.



    if it==0:
        print('The #training data is {}, #validation data {} and #internal test data {}. '.format(trainLength, valLength, innerTestLength))
        print('The #actual test data is {}.'.format(len(it_test_df)))

    class_dataset = ASEAtomsData.create(dataPath, 
                                    distance_unit='Ang',
                                    property_unit_dict={
                                        'synth':int(1),
                                        # 'material_id':int(1)
                                        })
    print('adding systems to dataset')
    # 
    # #####IF THE SEPARATE DATALOADER WORKS< THIS NEED TO UPLOAD it_train_df #####
    # 
    class_dataset.add_systems(np.array(it_train_df.targets), np.array(it_train_df.atoms))  

    print('creating data module')
    # DataModuleWithPred
    # crysData = AtomsDataModule(datapath=dataPath,
    crysData = DataModuleWithPred(datapath=dataPath,
                    batch_size=512,
                    num_train=trainLength,
                    num_val=valLength,
                    transforms=[
                        trn.ASENeighborList(cutoff=5.),
                        trn.CastTo32(), 
                        # trn.RemoveOffsets('synth', remove_mean=True, 
                        #                   remove_atomrefs=True),
                                ],
                    property_units={'synth':int(1)},
                    # num_workers=6,    #for personal cpu
                    num_workers=10,    #for personal gpu
                    split_file = splitFilestring, 
                    # pin_memory=False, # set to false, when not using a GPU
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=['synth'], #only load U0 property
    )

    crysData.prepare_data()
    crysData.setup()

    # %%
    #   ###alternative formulation: a second test dataloader:
    # testdatapath ='./class_profile_dataset_test2.db'
    splitFilestring = directory_setup(dataPath = testdatapath,save_dir = save_dir, bestModelPath= bestModelPath, iteration_num=it)

    test_dataset = ASEAtomsData.create(testdatapath, 
                                    distance_unit='Ang',
                                    property_unit_dict={
                                        'synth':int(1),
                                        # 'material_id':int(1)
                                        })
    print('adding systems to the test dataset')
    test_dataset.add_systems(np.array(it_test_df.targets), np.array(it_test_df.atoms))  

    print('creating data module')
    # DataModuleWithPred

    crysTest = DataModuleWithPred(datapath=testdatapath,
                    batch_size=20,
                    num_train=len(it_test_df)+13,#accepts nonsensical values,
                    # just not zero.
                    num_val=-1,#accepts nonsensical values,
                    # just not zero.
                    num_test=len(it_test_df),
                    transforms=[
                        trn.ASENeighborList(cutoff=5.),
                        trn.CastTo32(), 
                        # trn.RemoveOffsets('synth', remove_mean=True,
                                        #   remove_atomrefs=True),
                                ],
                    property_units={'synth':int(1)},
                    num_workers=10,    #for personal gpu
                    split_file = splitFilestring, 
                    # pin_memory=False, # set to false, when not using a GPU
                    pin_memory=True, # set to false, when not using a GPU
                    load_properties=['synth'], #only load U0 property
    )

    crysTest.prepare_data()
    crysTest.setup()





    # %%

    means, stddevs = crysData.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True)

    meansTest, stddevsTest = crysTest.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True)

    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())
    # This doesn't work when no test data is given, and it has no docsAccurtring. Does it calculate the mean and and std of test data?

    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    early_stopping = EarlyStopping(
        monitor="val_synth_Accur",  #if it works, also change in ModelCheckpoint?
        # rhe error says"Pass in or modify your `EarlyStopping` callback to use any of the following: `train_loss`, `val_loss`, `val_synth_Accur`, `train_synth_Accur`""
        min_delta=0.01,
        patience=5,
    )
    callbacks = [
        early_stopping,
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(save_dir, "best_inference_model"),
        # model_path=os.path.join(save_dir, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
    ]
    # %%
    print(crysData)

    # %%
    # myProfiler = PyTorchProfiler(filename='profile_test') #pytorch profiler doesn't work. Error:
    # #AttributeError: Can't pickle local object 'schedule.<locals>.schedule_fn'
    myProfiler = AdvancedProfiler(filename='profile_test')


    trainer = pl.Trainer(#accelerator='gpu', devices=4,
    # accelerator='cpu',
    accelerator='gpu',
    gpus=[0,1,2],
    auto_select_gpus = True,
    strategy='ddp',
    # strategy='None',
    precision=16,
    callbacks=callbacks,
    logger=logger,
    # profiler=myProfiler,
    default_root_dir=save_dir,
    max_epochs=epoch_num, # for testing, we restrict the number of epochs
    )
    # %%
    trainer.fit(task, datamodule=crysData)
    best_model = torch.load(bestModelPath)
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

    it_test_df = it_test_df.merge(resdf, left_index=True, right_on='testIndex')
    #%%

    # it_test_df['pred_score'+str(it)] = it_test_df.atoms.apply(
        # lambda x: pred_test(x, best_model=best_model, returnScore=True))

    it_test_df = it_test_df[['crystal_id','pred_'+str(it)]]
    # it_test_df = it_test_df[['crystal_id','pred_'+str(it), 'pred_score'+str(it)]]

    crysdf = pd.merge(left = crysdf, right=it_test_df,how = 'outer',left_on='crystal_id', right_on='crystal_id')

    et = time.time()
    itt = et-st
    print("===the {}th iteration took {} to run===".format(it, timedelta(seconds=itt)))
            
    # crysdf.to_csv(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration
    # crysdf.to_parquet(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration
    crysdf.to_pickle(save_dir+'/res_df/'+res_df_fileName+'tmp')   #saving results at each iteration

# break


# %%
crysdf.target_pd = crysdf.target_pd.map(lambda x: int(x[0])) #changing array to target for easier handling
# t = it_test_df.pred_score0.map(lambda x: next(iter(x.values())).detach().numpy()[0])

# %%
# t.head(20)

# %%
# crysdf.to_csv(save_dir+'/res_df/'+res_df_fileName)
# crysdf.to_parquet(save_dir+'/res_df/'+res_df_fileName)
crysdf.to_pickle(save_dir+'/res_df/'+res_df_fileName)
# crysdf = pd.read_csv(save_dir+'/res_df/'+res_df_fileName, index_col=0)
# crysdf = pd.read_pickle(save_dir+'/res_df/'+res_df_fileName)

# %%
# tt = pd.read_csv(save_dir+'/res_df/crysdf.csv', index_col=0)

# %%
pred_columns = []
score_columns = []
if num_iter == 0:  #just patching it so it works for a single iteration
    pred_col_name = 'pred_'+str(it)
    pred_columns.append(pred_col_name)
    
    score_col_name = 'pred_score'+str(it)
    score_columns.append(score_col_name)

for it in range(0, num_iter):  #always start at 0 because we want to average Preds over all the iterations.
    pred_col_name = 'pred_'+str(it)
    pred_columns.append(pred_col_name)
    
    score_col_name = 'pred_score'+str(it)
    score_columns.append(score_col_name)


# Preds = crysdf.drop(columns=['crystal_id', 'atoms', 'target_pd', 'targets',])
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

crysdf['Preds'] = Preds.apply(scoreFunc, axis=1)

# %%
crysdf[['Preds', 'trial_num']] = crysdf.Preds.tolist()

# %%
res_df = crysdf[crysdf.Preds.notna()][['target_pd', 'Preds', 'trial_num']]  #selecting data with prediction values

# %%
experimental_df = res_df[res_df.target_pd==1]
theoretical_df = res_df[res_df.target_pd==0]

# %%
# true_positive_rate = sum(experimental_df[experimental_df.target_pd==1].Preds>=.5)/experimental_df.shape[0]  #extra 
true_positive_rate = sum(experimental_df.Preds>=.5)/experimental_df.shape[0]
unlabeled_synth_frac = sum(theoretical_df.Preds>=.5)/theoretical_df.shape[0]

# %%
print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))
print('and {:.1f}% ofour unlabeled data have been predicted to belong to the positive class.'.format(unlabeled_synth_frac*100))

# %%
endTime = time.time()
elapsed_time = endTime-startTime
print("This script took {} to run.".format(timedelta(seconds=elapsed_time)))

# %%

# true_positive_rate = posTest.pred.sum()/posTest.shape[0]
# print(f"our true-positive rate is {round(true_positive_rate, ndigits =3)}")


# %%
