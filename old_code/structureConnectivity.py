### This script loads lightStructureEnv, calculates StructureConn, and saves SC instead of LSE under the same file names.

# This supresses warnings.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pickle

from pymatgen.util.coord import get_angle
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import ConnectivityFinder


# import multiprocessing
# from joblib import Parallel, delayed  #cant't install joblib. Needs to update anaconda, which I'm not athorized to do.
import logging
import os


# %%
data_chunk_list = [file for file in os.listdir('data') if file.startswith("coordata_")]

print("Files to be modified are", *data_chunk_list )
# %%
for chunk in data_chunk_list:
    lse_data = np.load('data/'+chunk,allow_pickle=True)
    cf= ConnectivityFinder()
    print('Calculating Structure Connectivity for ', chunk)
    for datum in lse_data:
        try:
            lse = datum['lightStructureEnv']
        except:
            continue
        sc=cf.get_structure_connectivity(light_structure_environments=lse)
        del datum['lightStructureEnv'] #this structure currently exists under the StructureConnectivity object, no need to keep 2 copies.
        datum['StructureConn'] = sc

    np.save('data/'+chunk, lse_data) 


