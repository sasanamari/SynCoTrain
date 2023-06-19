# %% [markdown]
# First set up for using Schnet on our crystal data.

# %%
import os
from pathlib import Path  #recommended path library for python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import schnetpack as spk
from schnetpack.data import ASEAtomsData, BaseAtomsData, AtomsDataFormat, AtomsDataModule
import schnetpack.transform as trn
import torch
import torchmetrics
import pytorch_lightning as pl
import pickle
import pandas as pd
import time
from datetime import timedelta

# %%
startTime = time.time()

# %%
print(Path().resolve())  #current working directory
print(Path().absolute()) #file path in jupyter
# print(Path(__file__).parent.resolve()) #file path (dcript directory) path in a python script

# %%
def directory_setup(dataPath,save_dir, bestModelPath, iteration_num=None):
    if iteration_num == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Logging directory was created')
        
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
    except OSError as e:
        print(e)        
        
    bestModelPath_obj = Path(bestModelPath)
    try:
        bestModelPath_obj.unlink()
    except OSError as e:
        print(e)       
         
    return str(splitFile_path)

# %%
def pred_test(x, best_model):
    inputs = converter(x)
    pred = best_model(inputs)
    pred_arr = next(iter(pred.values())).detach().numpy()    #dict_value to array
    if (1-pred_arr<pred_arr):
        pred_arr = 1
    else:
        pred_arr = 0
    return pred_arr

# %% [markdown]
# #Important note!

# %% [markdown]
# I'm gonna use electron volt for the compulsory unit decleration to see if I can complete the classification workflow. But even if it works I'll need to fix it soon. 

# %% [markdown]
# First I'll try doing a regression and getting the round number (or closest number to label) as the class. Later I'll have to use classiification loss.

# %%
print(torch.cuda.is_available())
torch.device('cpu')     #gpu was busy
# print(torch.tensor([1., 2.]).device)      #checking the current device


# %%
np.random.seed(42)
pAtoms = np.load("data_for_dev_try/PosAtomsUnder12.npy", allow_pickle=True)
tAtoms = np.load("data_for_dev_try/TheoAtomsUnder12.npy", allow_pickle=True)

# %%
# pSynth = [np.array(1).flatten()]*len(pAtoms)    #we need the array to have the shape (1,), hence we use flatten()
# tSynth = [np.array(0).flatten()]*len(tAtoms)
pSynth = [np.array(1, dtype=np.int32).flatten()]*len(pAtoms)    #we need the array to have the shape (1,), hence we use flatten()
tSynth = [np.array(0, dtype=np.int32).flatten()]*len(tAtoms)

crysData = np.concatenate([pAtoms, tAtoms])
targetData = [*pSynth, *tSynth]   #again, we need distinct arrays. np.concatenate would merge all in one array.
crysdf = pd.DataFrame()
crysdf['myatoms'] = crysData
crysdf['target_pd'] = targetData
if targetData[0].shape !=(1,):
    print("Target data has the wrong shape of ", targetData[0].shape)
    # break
crysdf["targets"] = crysdf.target_pd.map(lambda crystalClass: dict(synth=np.array(crystalClass)) )     #changes targets fromat from array to dict with array val
crysdf = crysdf.reset_index(drop=False)
crysdf = crysdf.rename(columns={'index':'crystal_id'})
crysdf = crysdf.sample(frac=1, random_state=42).reset_index(drop=True)     #simply shuffles the rows of positive and Unlabeled(negative) data


# %%


# %%
cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64
dataPath = './class_dataset.db'
save_dir = './qm9tut'
bestModelPath = save_dir+'/'+'best_inference_model'
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
    # postprocessors=[trn.CastTo64(), trn.AddOffsets('synth', add_mean=True, add_atomrefs=False)]  
)

# output_synth = spk.task.ModelOutput(
#     name='synth',
#     loss_fn=torch.nn.MSELoss(), #this+metrics below later changes to BCELoss 
#     loss_weight=1.,
#     metrics={
#         "MAE": torchmetrics.MeanAbsoluteError()
#     }
# )
output_synth = spk.task.ModelOutput(
    name='synth',
    loss_fn=torch.nn.BCEWithLogitsLoss(), 
    # loss_fn=torch.nn.CrossEntropyLoss(), 
    # loss_fn=torch.nn.BCELoss(), 
    loss_weight=1.,
    metrics={
        "Accur": torchmetrics.Accuracy()   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
    }
)


task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_synth],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

epoch_num = 13

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)


# %%


# %%
num_iter = 2
# ###figure out time from Tensorboard
# ###if we sample without replacement, it'll be a wider search, but less robust for each datum.
for it in range(num_iter):
    st = time.time()
    print('we started iteration {}'.format(it))
    splitFilestring = directory_setup(dataPath = dataPath,save_dir = save_dir, bestModelPath= bestModelPath, iteration_num=it)
    
    np.random.seed(it)
    crysdf = crysdf.sample(frac=1, random_state=it).reset_index(drop=True)     #simply shuffles the rows of positive and Unlabeled(negative) data
    
    num_pos = crysdf.target_pd.sum()
    num_theo = crysdf.shape[0]-num_pos
    
    pos_train_num = int(0.9*num_pos)
    theo_train_num = int(0.85*num_theo)
    
    df0 = crysdf[crysdf.target_pd==0].sample(n=theo_train_num,random_state=it)
    df1 = crysdf[crysdf.target_pd==1].sample(n=pos_train_num,random_state=it+1)
    
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
        

    class_dataset = ASEAtomsData.create(dataPath, 
                                    distance_unit='Ang',
                                    property_unit_dict={'synth':int(1)}     #We need to do something about this unit.                             
                                     )
    print('adding systems to dataset')
    class_dataset.add_systems(np.array(crysdf.targets), np.array(crysdf.myatoms))  
    
    print('creating data module')
    crysData = AtomsDataModule(datapath=dataPath,
                   batch_size=20,
                    num_train=trainLength,
                    num_val=valLength,
                    transforms=[
                        trn.ASENeighborList(cutoff=5.),
                        # trn.CastTo32(), 
                        # trn.RemoveOffsets('fepa', remove_mean=True, remove_atomrefs=True),
                                ],
                    property_units={'synth':int(1)},
                    num_workers=10,    #we started with 1
                    split_file = splitFilestring, 
                    pin_memory=False, # set to false, when not using a GPU
                    # pin_memory=True, # set to false, when not using a GPU
                    load_properties=['synth'], #only load U0 property
    )
    
    crysData.prepare_data()
    crysData.setup()
    
    means, stddevs = crysData.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True)
    
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())
    # This doesn't work when no test data is given, and it has no docstring. Does it calculate the mean and and std of test data?
    
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    callbacks = [
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(save_dir, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
    ]
    
    trainer = pl.Trainer(#accelerator='gpu', devices=4,
    accelerator='cpu',
    # gpus=[0,1],
    # auto_select_gpus = True,
    strategy=None,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=save_dir,
    max_epochs=epoch_num, # for testing, we restrict the number of epochs
)
    trainer.fit(task, datamodule=crysData)
    best_model = torch.load(bestModelPath)
    best_model = best_model.to('cpu')
    
    it_test_df['pred_'+str(it)] = it_test_df.myatoms.apply(lambda x: pred_test(x, best_model=best_model))
    
    # t = it_test_df.sample(frac=1,random_state=it+2)#.set_index('crystal_id')
    it_test_df = it_test_df[['crystal_id','pred_'+str(it)]]
    
    crysdf = pd.merge(left = crysdf, right=it_test_df,how = 'outer',left_on='crystal_id', right_on='crystal_id')
    
    et = time.time()
    itt = et-st
    print("===the {}th iteration took {} to run===".format(it, timedelta(seconds=itt)))
    # break
    

# %%
crysdf.target_pd = crysdf.target_pd.map(lambda x: int(x[0])) #changing array to target for easier handling

# %%
crysdf.to_csv(save_dir+'/res_df/crysdf3.csv')

# %%
# tt = pd.read_csv(save_dir+'/res_df/crysdf.csv', index_col=0)

# %%
Preds = crysdf.drop(columns=['crystal_id', 'myatoms', 'target_pd', 'targets'])

# %%
def scoreFunc(x):
    iter_num = sum(x.notna())
    if iter_num == 0:
        return np.nan, iter_num
    res = x.sum()
    score = res/iter_num
    return score, iter_num

# %%
crysdf['Preds'] = Preds.apply(scoreFunc, axis=1)

# %%
crysdf[['Preds', 'iter_num']] = crysdf.Preds.tolist()

# %%
res_df = crysdf[crysdf.Preds.notna()][['target_pd', 'Preds', 'iter_num']]  #selecting data with prediction values

# %%
experimental_df = res_df[res_df.target_pd==1]
# theoretical_df = res_df[res_df.target_pd==0]

# %%
true_positive_rate = sum(experimental_df[experimental_df.target_pd==1].Preds>=.5)/experimental_df.shape[0]

# %%
print('Our true positive rate is {:.1f}% after {} iterations of {} epochs.'.format(true_positive_rate*100, num_iter, epoch_num))

# %%
endTime = time.time()
elapsed_time = endTime-startTime
print("This script took {} to run.".format(timedelta(seconds=elapsed_time)))

# %%



