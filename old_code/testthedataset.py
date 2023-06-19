# %% [markdown]
# First set up for using Schnet on our crystal data.

# %%
import os
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
# from pymatgen.io.ase import AseAtomsAdaptor as pase

# %% [markdown]
# #Important note!

# %% [markdown]
# I'm gonna use electron volt for the compulsory unit decleration to see if I can complete the classification workflow. But even if it works I'll need to fix it soon. 

# %% [markdown]
# First I'll try doing a regression and getting the round number (or closest number to label) as the class. Later I'll have to use classiification loss.

# %%
# try using the best_model to predict as the example. Just figure out the least amount of test data you can provide to the trainer.

# %%
torch.cuda.is_available()

# %%
torch.device('cpu')     #gpu was busy

# %%
torch.tensor([1., 2.]).device      #checking the current device

# %%
pAtoms = np.load("data_for_dev_try/PosAtomsUnder12.npy", allow_pickle=True)
tAtoms = np.load("data_for_dev_try/TheoAtomsUnder12.npy", allow_pickle=True)

# %%
print('The length of positive data is {} and the length of the theoretical data is {}'.format(len(pAtoms), len(tAtoms)))

# %%
np.random.seed(42)
np.random.shuffle(pAtoms)
np.random.shuffle(tAtoms)

# %%
mypAtoms = pAtoms[:100]
testpAtoms = pAtoms[-20:]
mytAtoms = tAtoms[:500]
testtAtoms = tAtoms[-100:]

# %%
# pSynth = np.repeat(1,len(mypAtoms))
# tSynth = np.repeat(0,len(mytAtoms))   #using np.repeat places everything in one array which is not the format shcnet format. we need distinct arrays for each datum.
pSynth = [np.array(1).flatten()]*len(mypAtoms)    #we need the array to have the shape (1,), hence we use flatten()
tSynth = [np.array(0).flatten()]*len(mytAtoms)

testpSynth = [np.array(1).flatten()]*len(testpAtoms)    #here we prepare test data in the same format as the  training data.
testtSynth = [np.array(0).flatten()]*len(testtAtoms)

# %%
crysData = np.concatenate([mypAtoms, mytAtoms])
testCrysData = np.concatenate([testpAtoms, testtAtoms])

targetData = [*pSynth, *tSynth]   #again, we need distinct arrays. np.concatenate would merge all in one array.
testTargetData = [*testpSynth, *testtSynth]   

# %%
crysdf = pd.DataFrame()
testCrysdf = pd.DataFrame()

# %%
crysdf['myatoms'] = crysData
testCrysdf['myatoms'] = testCrysData

# %%
crysdf['targets'] = targetData
testCrysdf['targets'] = testTargetData

# %%
print(targetData[0].shape, testTargetData[0].shape)      #each shape needs to be (1,)

# %%
crysdf.targets = crysdf.targets.map(lambda crystalClass: dict(synth=np.array(crystalClass)) )     #changes targets fromat from array to dict with array val
testCrysdf.targets = testCrysdf.targets.map(lambda crystalClass: dict(synth=np.array(crystalClass)) )

# %%
print(crysdf.loc[0].targets)
# this needs to return {'synth':array([1 or 0])}

# %%
# dummy_synth = pd.get_dummies(crysdf['targets'],prefix = 'synth')
# crysdf = pd.merge(crysdf,dummy_synth,
#     left_index=True,
#     right_index=True,
# )
# crysdf.head()

# %%
# we need to shuffle here to mix the positive and unlabaled data together.
crysdf = crysdf.sample(frac=1, random_state=1).reset_index(drop=True)     #simply shuffles the rows of the dataframe
testCrysdf = testCrysdf.sample(frac=1, random_state=1).reset_index(drop=True)     #simply shuffles the rows of the dataframe


# %%
# %rm split.npz
# %rm qm9tut/lightning_logs/split.npz

trainLength = round(len(crysdf)*.8)-2
# valLength = round(trainLength*.2)-2
valLength = round(len(crysdf)*.2)-2
testLength = len(crysdf)-(trainLength+valLength)

print('The #training data is {}, #validation data {} and #internal test data {}. '.format(trainLength, valLength, testLength))


# %%
# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     a1 = a[p]
#     b1 = b[p]
#     return a1,b1

# %%
# fepaTest = fepa_array[:testLength//2]
# atomsTest = newAtoms[:testLength//2]

# fepaTrain = fepa_array[testLength//2:]
# atomsTrain = newAtoms[testLength//2:]


# %%
datapth = './class_dataset.db'
# %rm {datapth}
pathname = os.path.abspath(os.path.join(datapth))
try:
    os.remove(pathname)
except:
    pass


qm9tut = './qm9tut/lightning_logs'
if not os.path.exists('qm9tut/lightning_logs'):
    os.makedirs(qm9tut)
    
dirname = ""
filename = "split.npz"
pathname = os.path.abspath(os.path.join(dirname, filename))
if pathname.startswith(dirname):
   try:
       os.remove(pathname)
   except:
       pass
    
# %rm qm9tut/lightning_logs/best_inference_model
# %rm synth/lightning_logs/best_inference_model

class_dataset = ASEAtomsData.create(datapth, 
                                  distance_unit='Ang',
                                  property_unit_dict={'synth':'eV'}                                  
                                #   property_unit_dict={'synth':'myUnit'}   #need to fix this
                        )


# %%
# class_dataset.add_systems(crysdf.targets, crysdf.myatoms)  #integer target
class_dataset.add_systems(np.array(crysdf.targets), np.array(crysdf.myatoms))  

# %%
# print(class_dataset[0])

# %%
class_dataset[0]['synth'].device

# %%
for p in class_dataset.available_properties:
    print('-', p)
print()

example = class_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)

# %%
qm9tut = './qm9tut/lightning_logs'
if not os.path.exists('qm9tut/lightning_logs'):
    os.makedirs(qm9tut)


# %%
crysData = AtomsDataModule(datapath=datapth,
                   batch_size=20,
                #    batch_size=100,
    num_train=trainLength,
    num_val=valLength,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        # trn.RemoveOffsets('fepa', remove_mean=True, remove_atomrefs=True),
        trn.CastTo32()
    ],
    property_units={'synth':'eV'},
    # property_units={'synth':'myUnit'},     #need to fix this
    num_workers=1,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=False, # set to false, when not using a GPU
    # pin_memory=True, # set to false, when not using a GPU
    load_properties=['synth'], #only load U0 property
    )

# %%


# %%
crysData.prepare_data()
crysData.setup()

# %%
print("The total length of the data set is", len(class_dataset))
print("The length of the training set is", trainLength)
print("The length of the validation set is", valLength)
print("The length of the test set is", len(class_dataset)-(trainLength+valLength))

# %%
# atomrefs = crysData.train_dataset.atomrefs
# atomrefs

# %%
means, stddevs = crysData.get_stats(
    'synth', divide_by_atoms=True, remove_atomref=True
)
print('Mean atomization energy / atom:', means.item())
print('Std. dev. atomization energy / atom:', stddevs.item())
# This doesn't work when no test data is given, and it has no docstring. Does it calculate the mean and and std of test data?

# %%
cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64

# %%
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms

# %%
from torch.nn import functional

# %%
# class SchNetInteraction(nn.Module):
#     r"""SchNet interaction block for modeling interactions of atomistic systems."""

#     def __init__(
#         self,
#         n_atom_basis: int,
#         n_rbf: int,
#         n_filters: int,
#         activation: Callable = shifted_softplus,
#     ):
#         """
#         Args:
#             n_atom_basis: number of features to describe atomic environments.
#             n_rbf (int): number of radial basis functions.
#             n_filters: number of filters used in continuous-filter convolution.
#             activation: if None, no activation function is used.
#         """
#         super(SchNetInteraction, self).__init__()
#         self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
#         self.f2out = nn.Sequential(
#             Dense(n_filters, n_atom_basis, activation=activation),
#             Dense(n_atom_basis, n_atom_basis, activation=None),
#         )
#         self.filter_network = nn.Sequential(
#             Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         f_ij: torch.Tensor,
#         idx_i: torch.Tensor,
#         idx_j: torch.Tensor,
#         rcut_ij: torch.Tensor,
#     ):
#         """Compute interaction output.

#         Args:
#             x: input values
#             Wij: filter
#             idx_i: index of center atom i
#             idx_j: index of neighbors j

#         Returns:
#             atom features after interaction
#         """
#         x = self.in2f(x)
#         Wij = self.filter_network(f_ij)
#         Wij = Wij * rcut_ij[:, None]

#         # continuous-filter convolution
#         x_j = x[idx_j]
#         x_ij = x_j * Wij
#         x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

#         x = self.f2out(x)
#         return x
    
#     # change the final activation
#     # this must be called in a loop

# %%
def mySigmoid(x: torch.Tensor):
    from torch.nn import functional
    
    """
    Inverse of the softplus function.

    Args:
        x (torch.Tensor): Input vector

    Returns:
        torch.Tensor: softplus inverse of input.
    """
    return #return mySigmoid

# %%
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=3, radial_basis=radial_basis,
    
    cutoff_fn = spk.nn.CosineCutoff(cutoff), activation=functional.sigmoid
)


# %%
# spk.representation.SchNet?

# %%
# pred_fepa = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='fepa')
pred_synth = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='synth')

# %%
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    # output_modules=[pred_fepa],
    output_modules=[pred_synth],
    # postprocessors=[trn.CastTo64(), trn.AddOffsets('fepa', add_mean=True, add_atomrefs=False)]  
    postprocessors=[trn.CastTo64(), trn.AddOffsets('synth', add_mean=True, add_atomrefs=False)]  
)

# %%
nnpot

# %%
output_synth = spk.task.ModelOutput(
    name='synth',
    loss_fn=torch.nn.MSELoss(), #this+metrics below later changes to BCELoss 
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

# %%

# output_synth = spk.task.ModelOutput(
#     name='synth',
#     loss_fn=torch.nn.BCELoss(), 
#     loss_weight=1.,
#     metrics={
#         "Accur": torchmetrics.Accuracy()   #potential alternatives: AUROC(increases the area under ROC curve), AveragePrecision (summarises the precision-recall curve)
#     }
# )

# %%
task = spk.task.AtomisticTask(
    model=nnpot,
    # outputs=[output_fepa],
    outputs=[output_synth],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# %%
logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
callbacks = [
    spk.train.ModelCheckpoint(
        inference_path=os.path.join(qm9tut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]


# %%

trainer = pl.Trainer(
    # devices=2,   #error says a strategy is selected which is not compatible with interactive mode; regardless of choosing a "compatible" strategy.
    # auto_select_gpus = False,
    # gpus=[0],  #this selects the number of gpus to use, not the exact one.
    # auto_select_gpus = True,
    # gpus=1,  #this selects the number of gpus to use, not the exact one.
    accelerator='cpu',
    strategy=None,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=3, # for testing, we restrict the number of epochs
)

# %% [markdown]
# We need sigmoid before cross_entropy loss
# 

# %%
# trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)#, datamodule=qm9data)
trainer.fit(task, datamodule=crysData)

# %%
best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'))

# %%


# %%
trainer.test(task, crysData)

# %% [markdown]
# When training on class labels: [{'test_loss': 2.7045037746429443, 'test_synth_MAE': 1.297343373298645}]
# It seems like the test_loss magnitude is changes more with the target?
# 

# %% [markdown]
# after 3 epochs: [{'test_loss': 11.951684951782227, 'test_fepa_MAE': 2.681666851043701}]

# %% [markdown]
# after 10 epochs: [{'test_loss': 2.2840123176574707, 'test_fepa_MAE': 1.183426022529602}]

# %%


# %%
# trainer.predict(task, crysData)

# %%
# %reload_ext tensorboard
# %tensorboard --logdir=./qm9tut/lightning_logs

# %%


# %%
# stop!

# %%
# crysData.setup()  #does not seem to make a difference

# %%
# for i,batch in enumerate(crysData.test_dataloader()):
#     print('starting {}'.format(i)) 
#     result = best_model(batch)
#     err = sum(abs(batch['fepa'].detach().numpy()-result['fepa'].detach().numpy()))
#     print('The absolute error is',err)

# %% [markdown]
# ##### Testing with data not already seen by the dataloader

# %%
converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)

# %%
best_model = best_model.to('cpu')

# %%
# testCrysdf.head()

# %%
predSynth = []
    
for ind, row in testCrysdf.iterrows():
    inputs = converter(row['myatoms'])
    pred = best_model(inputs)
    pred_arr = next(iter(pred.values())).detach().numpy()    #dict_value to array
    predSynth.append([ind,pred_arr])
    
for datum in predSynth:     #changing the reg output to class labels based on distance from label value
    if (1-datum[1]<datum[1]):
        datum[1] = 1
    else:
        datum[1] = 0

# %%
# how to check a data point or accuracy:
acc = []
for d in predSynth:
    acc.append(next(iter(testCrysdf.iloc[d[0]].targets.values()))==d[1])

# %%
sum(acc)/len(acc) #this doesn't mean much because we don't know about unlabele data.
# we want to group the positive label data points together anc check the rate of true positives there.

# %%


# %%


# # %%
# res = [datum['fepa'].detach().numpy() for datum in predFepa]
# fepaTestVals = [datum['fepa'] for datum in fepaTest]

# # %%
# plt.scatter(fepaTestVals, res)
# plt.xlabel("Target values");
# plt.ylabel("Predictions");

# # %%
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # %%
# # these are different test data

# # %%
# mean_absolute_error(fepaTestVals, res)

# # %%
# mean_absolute_error(fepaTestVals, res)

# # %%
# mean_squared_error(fepaTestVals, res)

# # %%
# mean_squared_error(fepaTestVals, res)

# # %%
# crysData.test_dataloader().pin_memory_device = 'cpu'

# %%


# %% [markdown]
# code we might use later:

# %%
# class MetricTracker(pl.callbacks.Callback):
#     def __init__(self):
#         self.val_error_batch   = []
#         self.val_error         = []
#         self.train_error_batch = []
#         self.train_error       = []

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.train_error_batch.append(outputs['loss'].item())

#     def on_train_epoch_end(self, *args, **kwargs):
#         self.train_error.append(np.mean(self.train_error_batch))
#         self.train_error_batch = []

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
#         self.val_error_batch.append(outputs['val_loss'].item())

#     def on_validation_epoch_end(self, trainer, pl_module):
#         self.val_error.append(np.mean(self.val_error_batch))
#         self.val_error_batch = []

# %%
# td = [batch for batch in crysData.test_dataloader()]

# %%


# %%
# # the log file is not saved, at least where it was before. Have a look around, perhaps clean the folder
# # or try in a different TEST folder and try again.
# import matplotlib.pyplot as plt
# from ase.units import kcal, mol

# results = np.loadtxt(os.path.join(qm9tut, 'log.csv'), skiprows=1, delimiter=',')

# time = results[:,0]-results[0,0]
# learning_rate = results[:,1]
# train_loss = results[:,2]
# val_loss = results[:,3]
# val_mae = results[:,4]

# print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
#       np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')

# plt.figure(figsize=(14,5))
# plt.subplot(1,2,1)
# plt.plot(time, val_loss, label='Validation')
# plt.plot(time, train_loss, label='Train')
# plt.yscale('log')
# plt.ylabel('Loss [eV]')
# plt.xlabel('Time [s]')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(time, val_mae)
# plt.ylabel('mean abs. error [eV]')
# plt.xlabel('Time [s]')
# # plt.show()
# # plt.savefig('tempfigDev.jpg')

# %%



