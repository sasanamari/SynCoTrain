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


# import multiprocessing
# from joblib import Parallel, delayed  #cant't install joblib. Needs to update anaconda, which I'm not athorized to do.
import logging
import os

# num_cores = multiprocessing.cpu_count()

goodata = np.load('data/goodata.npy',allow_pickle=True)

# goodata = goodata[:53]############delete meeeeeeeeeeeeeeeeeeeeeeeeeeee
        
def get_coord(datum, mystrategy = "simple"):
    if mystrategy == "simple":
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
    else:
        strategy = mystrategy
        
    struc = datum["structure"]
    bv = BVAnalyzer()  #This class implements a maximum a posteriori (MAP) estimation method to determine oxidation states in a structure.
    oxid_stats = bv.get_valences(struc)
    
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")  #to avoid printing to the console 
    lgf = LocalGeometryFinder() 
    # prints a long stroy every time it is initiated.
    sys.stdout = old_stdout # reset old stdout
    
    lgf.setup_structure(structure=struc)
    
    # Get the StructureEnvironments 
    se = lgf.compute_structure_environments(only_cations=True,valences=oxid_stats,
    additional_conditions=[AdditionalConditions.ONLY_ANION_CATION_BONDS])
    
    lse = LightStructureEnvironments.from_structure_environments(
    strategy=strategy, structure_environments=se)
    
    return lse        
        
        
        
        
def batch_coord(data, coord_problem=True):

    mycoord_problem = []
    tempList = [{} for _ in range(len(data))]
    tempArray = np.array(tempList)
    del tempList

    for j, datum in enumerate(data):

        try:
            currentMatID = datum["material_id"]
            tempArray[j]["material_id"] = currentMatID
            lse = get_coord(datum=datum, mystrategy="simple")
            tempArray[j]["lightStructureEnv"] = lse

        except:

            if coord_problem == True:

                mycoord_problem.append([j, datum["pretty_formula"]])

                print("Couldn't analyze valencce for ", j, "th datum ",
                      datum["pretty_formula"])

            else:
                print("Couldn't analyze valencce for ", j, "th datum ",
                      datum["pretty_formula"])
    return mycoord_problem, tempArray

def chunker(seq, size):
    """chunks the indices of an iterable object to pieces of a given size"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        
        

chunkSize = 1000

coord_problem = []
for i, chunk in enumerate(chunker(goodata, chunkSize)):
    print('calculating coordination for the range', i*chunkSize, 'to', min((i+1)*chunkSize, len(goodata)))
    # chunkCoord_problem, tempArray = Parallel(n_jobs=num_cores-1)(delayed(batch_coord)(i) for i in chunk)
    chunkCoord_problem, tempArray = batch_coord(chunk, coord_problem=True)
    coord_problem.append(chunkCoord_problem)  #unravel this list later
    np.save("data/coordata_" + str(i), tempArray)
    del tempArray
    
    
coord_problem = np.array(coord_problem)
np.save("data/coord_problems", coord_problem)

# import numpy as np
# test = np.load("coordata_1.npy", allow_pickle=True)
# print(test)
#should get the indecis out of coord_problem and throw them out!

#goodata = np.delete(goodata, [*coord_ind])
        
#np.save("data/coordata", goodata)   #saving the entries with the calculated coordination  
        
# neighbors_sites are accessed with the syntax datum["lightStructureEnv"].neighbors_sets[isite][0].neighb_sites_and_indices       
        
        
        
        

