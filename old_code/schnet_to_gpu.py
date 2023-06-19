# ###This is an incomplete code sinppet to add to a schnet workflow.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(1) # safer to use before loading lightning.gpu
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
import pickle
import pandas as pd
import time
from datetime import timedelta
import int2metric
from Datamodule4PU import *
from pytorch_lightning.callbacks import EarlyStopping

testdatapath ='./testDataset.db'
# ### you do everything as before until you det up you training dataset,
# ### you create a separate test dataset. 
# ### I've assumed you keep your test data in a testdf dataframe.

test_dataset = ASEAtomsData.create(testdatapath, 
                                    distance_unit='Ang',
                                    property_unit_dict={
                                        'property_to_predict':'unit',
                                        })
print('adding systems to the test dataset')
test_dataset.add_systems(np.array(testdf.targets), np.array(testdf.atoms))  

crysTest = DataModuleWithPred(datapath=testdatapath,
                batch_size=20,
                num_train=5,#accepts nonsensical values,
                # just not zero.
                num_val=-1,#accepts nonsensical values, just not zero. 
                # Positive values result in "no mactch" error.
                num_test=len(testdf),
                transforms=[
                    trn.ASENeighborList(cutoff=5.),
                    trn.CastTo32(), 
                    # trn.RemoveOffsets('property_to_predict', remove_mean=True,
                                    #   remove_atomrefs=True),
                            ],
                property_units={'property_to_predict':'unit'},
                num_workers=10,   
                split_file = "splitFilestringTest", #provide new split_file for tests 
                pin_memory=True, # set to false, when not using a GPU
                load_properties=['property_to_predict'], #only load U0 property
)

crysTest.prepare_data()
crysTest.setup()

trainer = pl.Trainer()  #set up as usual

t = trainer.predict(model=task, 
                # datamodule=crysTest,
                dataloaders= crysTest.predict_dataloader(),
                return_predictions=True)


results = []
for batch in t:    
    for datum in batch['property_to_predict']:
        results = results+[datum.float()]
        
        
#I am assuming that results are ordered the way I imagine they are.
# meaning, test_dataset has the same order as the results which I have unpacked.
mid = []
for i, datum in enumerate(crysTest.test_dataset):
    groundTruth = datum['property_to_predict'].detach()
    ind = int(datum['_idx'])
    mid.append([ind,groundTruth,results[i]])

resdf = pd.DataFrame(mid, columns=['testIndex','GT','pred'])  #GT is a duplicate

testdf = testdf.merge(resdf, left_index=True, right_on='testIndex')
# testdf = testdf[['crystal_id','pred']]        