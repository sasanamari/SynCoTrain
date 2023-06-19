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

from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions

import logging
import os

from pymatgen.analysis.chemenv.connectivity import connectivity_finder  #weird syntax

coordPlayData = np.load("../chemheuristics/coordPlayData.npy", allow_pickle=True)

print(len(coordPlayData))


noLightStructure = []  #doesn't matter with the toy-data. Might matter with all the data.

for datum in coordPlayData:
    try: #not all data points have a "lightStructureEnv" object. Not sure why.        
        cf = connectivity_finder.ConnectivityFinder()
        res = cf.get_structure_connectivity(datum["lightStructureEnv"])
        datum['StructureConnectivity'] = res
    except:
        noLightStructure.append(datum['material_id'])   #better change all the previous problem lists to material_id instead of index
        continue

noLightStructure = np.array(noLightStructure)
np.save("noLightStructures", noLightStructure)

np.save("coordPlayData", coordPlayData) # rewriting it not to take too much space

print("an example for the StructureConnectivity object, shown as a dictionary: \n \n")
print(coordPlayData[0]['StructureConnectivity'].as_dict())
print('\n \n the StructureConnectivity object seems to contain all the previous (structure) data. \n do we need to keep all that data or just rewrite it with StructureConnectivity?')
