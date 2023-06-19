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
# from pymatgen.io.ase import AseAtomsAdaptor as pase

# %%
print(torch.cuda.is_available())

# %%
# with open("positive_data/dataUnder30Fepa", "wb") as fp:
#     pickle.dump(testfepa, fp)
with open("data_for_dev_try/pdataUnder30Fepa", "rb") as fp:
# with open("positive_data/dataUnder30Fepa", "rb") as fp:
    testfepa = pickle.load(fp)

# %%
# np.save("positive_data/newAtomsUnder30", newAtoms)
newAtoms = np.load("data_for_dev_try/pnewAtomsUnder30.npy", allow_pickle=True)
# newAtoms = np.load("positive_data/newAtomsUnder30.npy", allow_pickle=True)

# %%
newAtoms.shape, sys.getsizeof(newAtoms), len(testfepa)==len(newAtoms)

# %%
strucLenList = [len(datum) for datum in newAtoms]


# %%
fepa_list = [dict(fepa=crystalFepa) for crystalFepa in testfepa]
print(fepa_list[0])

# %%
datapth = './new_dataset.db'
# %rm {datapth}
# import os
pathname = os.path.abspath(os.path.join(datapth))
# print(os.getcwd())
# if pathname.startswith(dirname):
try:
    os.remove(pathname)
except:
    pass
# %%
new_dataset = ASEAtomsData.create(datapth, 
                                  distance_unit='Ang',
                                  property_unit_dict={'fepa':'eV',}                                  
                        # environment_provider= spk.environment.AseEnvironmentProvider(5)
                        )

new_dataset.add_systems(fepa_list, newAtoms)

# %%
for p in new_dataset.available_properties:
    print('-', p)
print()

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)

# %%
qm9tut = './qm9tut/lightning_logs'
if not os.path.exists('qm9tut/lightning_logs'):
    os.makedirs(qm9tut)


# %%
# %rm split.npz
# import os
dirname = ""
filename = "split.npz"
pathname = os.path.abspath(os.path.join(dirname, filename))
if pathname.startswith(dirname):
   try:
       os.remove(pathname)
   except:
       pass
   
# %%



trainLength = round(len(new_dataset)*.7)
valLength = round(trainLength*.2)
testLength = len(new_dataset)-(trainLength+valLength)

crysData = AtomsDataModule(datapath=datapth,
                   batch_size=100,
    num_train=trainLength,
    num_val=valLength,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        # trn.RemoveOffsets('fepa', remove_mean=True, remove_atomrefs=True),
        trn.CastTo32()
    ],
    property_units={'fepa': 'eV'},
    num_workers=1,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=['fepa'], #only load U0 property
    )

# %%
crysData.prepare_data()
crysData.setup()

# %%
print("The total length of the data set is", len(new_dataset))
print("The length of the training set is", trainLength)
print("The length of the validation set is", valLength)
print("The length of the test set is", len(new_dataset)-(trainLength+valLength))

# %%
# atomrefs = crysData.train_dataset.atomrefs
# atomrefs

# %%
means, stddevs = crysData.get_stats(
    'fepa', divide_by_atoms=True, remove_atomref=True
)
print('Mean atomization energy / atom:', means.item())
print('Std. dev. atomization energy / atom:', stddevs.item())

# %%
cutoff = 5
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
n_atom_basis = 30
n_filters = 64

# %%
pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms

# %%
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=3, radial_basis=radial_basis,
    
    cutoff_fn = spk.nn.CosineCutoff(cutoff),
)


# %%
pred_fepa = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='fepa')

# %%
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_fepa],
    postprocessors=[trn.CastTo64(), trn.AddOffsets('fepa', add_mean=True, add_atomrefs=False)]  
)

# %%
output_fepa = spk.task.ModelOutput(
    name='fepa',
    loss_fn=torch.nn.MSELoss(), #this+metrics below later changes to BCELoss 
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

# %%
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_fepa],
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

# trainer = pl.Trainer(
#     # devices=2,   #error says a strategy is selected which is not compatible with interactive mode; regardless of choosing a "compatible" strategy.
#     # auto_select_gpus = False,
#     # gpus=[0],  #this selects the number of gpus to use, not the exact one.
#     # auto_select_gpus = True,
#     # gpus=[1],  #this selects the number of gpus to use, not the exact one.
#     # strategy=None,
#     callbacks=callbacks,
#     logger=logger,
#     default_root_dir=qm9tut,
#     max_epochs=3, # for testing, we restrict the number of epochs
# )
trainer = pl.Trainer(
    auto_select_gpus = True,
    gpus=[1],  #this selects the number of gpus to use, not the exact one.
    strategy=None,
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=3, # for testing, we restrict the number of epochs
)

# %%
# trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)#, datamodule=qm9data)
trainer.fit(task, datamodule=crysData)

# %%


# %%
# stop!

# %%
best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'))

# %%
i = 0
for batch in crysData.test_dataloader():
# for batch in crysData.test_dataloader(pin_memory=False):
# for i,batch in enumerate(crysData.test_dataloader()):
    # print('starting {i}')
    print('starting {}'.format(i))
    result = best_model(batch)
    print("Result dictionary_{}:, {}".format(i, result))
    i+=1
    break

# %%


# %%
# # device = "cpu" # change to 'cpu' if gpu is not available
# device = "cuda:3" # change to 'cpu' if gpu is not available
# # original device was 'cuda'.
# n_epochs = 2 # 200 takes about 10 min on a notebook GPU. reduces for playing around (this comment was made for 200 epochs, not 20 as in here.)
# trainer.train(device=device, n_epochs=n_epochs)

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



