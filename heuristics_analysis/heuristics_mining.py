# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from pymatgen.core import Structure, Element
import pandas as pd
from data_scripts.crystal_structure_conversion import ase_to_pymatgen
import time
import datetime
from atomic_feature_func import *
from typing import Tuple
import os
# %%
currentLoc='heuristics_analysis/'
os.chdir(currentLoc)
subdirectory = "data_chunks"
# os.makedirs(subdirectory, exist_ok=True)
# %%
df = pd.read_pickle("_coSchAl1.pkl")

# %%
df['pyst']=df.atoms.map(ase_to_pymatgen)

# %%
df['oxidation'] =np.NaN
df['sc'] = np.NaN

# %%
def analyze_env_light(struc : Structure, mystrategy : str = "simple") -> Tuple[List[int], StructureConnectivity]:
    '''Analyzes the coordination environments and returns the LightStructureEnvironment (LSE) object for the crystal and the list of oxidation states.
    First, BVAnalyzer() calculates the oxidation states. Then, the LocalGeometryFinder() computes the structure_environment object, 
    from which the LightStructureEnvironment (LSE) is derived.     
    This light function DOES NOT return StructureConnectivity (SE).
    Parameters:
    ----------------
    struc : Structure 
        crystal Structure object from pymatgen
    mystrategy : string
	    The simple or combined strategy for calculating the coordination environments.
'''
    if mystrategy == "simple":
        strategy = SimplestChemenvStrategy(distance_cutoff=1.5, angle_cutoff=0.1)
        # strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
    else:
        strategy = mystrategy
        
    # struc = datum["structure"]
    bv = BVAnalyzer()  #This class implements a maximum a posteriori (MAP) estimation method to determine oxidation states in a structure.
    oxid_states = bv.get_valences(struc)
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")  #to avoid printing to the console 
    lgf = LocalGeometryFinder() 
    # prints a long stroy every time it is initiated.
    sys.stdout = old_stdout # reset old stdout
    lgf.setup_structure(structure=struc)
    # Get the StructureEnvironments 
    # se = lgf.compute_structure_environments(only_cations=True,valences=oxid_states,
    se = lgf.compute_structure_environments(only_cations=False,valences=oxid_states,
    additional_conditions=[AdditionalConditions.ONLY_ANION_CATION_BONDS])
    # Get LightStructureEnvironments
    lse = LightStructureEnvironments.from_structure_environments(
    strategy=strategy, structure_environments=se)
    # Get StructreConnectuvuty object
    # cf= ConnectivityFinder()
    # sc=cf.get_structure_connectivity(light_structure_environments=lse)

    return oxid_states, lse

# %%
def batch_coord_light(pd_series , coord_problem : bool = True):
    '''Loops through a sequence of data analyzing the coordination environment. 
    Returns an array, each entity is a dictionary containing the features of one crystal structure.
    Parameters:
    ----------------
    data : Iterable
        The list/array of crystal data containing Structure and material_id
    coord_problem : bool
        Whether to report data points with a problem in analyzing their coordinates. 
    '''
    mycoord_problem = []
    tempList : list[dict] = [{} for _ in range(len(pd_series))]
    tempArray = np.array(tempList)
    del tempList

    # for j, datum in enumerate(data):
    for j,(material_id, struct )in enumerate(pd_series.iteritems()):
        try:
            oxid_states, lse = analyze_env_light(struc=struct, mystrategy="simple")
            tempArray[j]["oxidation"] = oxid_states
            tempArray[j]["lse"] = lse
            tempArray[j]["material_id"] = material_id
            tempArray[j]["struc"] = struct
        except:
            if coord_problem:
                mycoord_problem.append([material_id])
                print("Couldn't analyze valencce for ", material_id)
            else:
                print("Couldn't analyze valencce for ", material_id,)
    return mycoord_problem, tempArray


# %%
def lightNBfeats(lse_object, oxidation_list , struct : Structure) -> dict:
    structure_data : dict = {}
    for atomIndex, atom in enumerate(lse_object.neighbors_sets):
        angles = []    
        structure_data[atomIndex] = {}
        structure_data[atomIndex]['oxidation'] = oxidation_list[atomIndex]
       
        structure_data[atomIndex]['element'] = struct[atomIndex].species_string
        structure_data[atomIndex]['coords'] = struct[atomIndex].coords
        # structure_data[atomIndex]['localEnv'] = local_Envs_list[atomIndex]#[0] return a list including a single dict. Shoulbe be multiple dicts in the compound strategy.
        neighbors = atom[0].neighb_sites_and_indices
        structure_data[atomIndex]['NN_distances'] = []
        for k in range(len(neighbors)):
            nb=neighbors[k]
            nb_element = nb['site'].species_string
            nb_distance = nb['site'].distance_from_point(struct[atomIndex].coords)
            structure_data[atomIndex]['NN_distances'].append([nb_distance,nb_element])
            for k2 in range(k+1, len(neighbors)):
                nb2 = neighbors[k2]
                angle = get_angle(struct[atomIndex].coords-nb['site'].coords,
                                struct[atomIndex].coords-nb2['site'].coords)
                angles.append(angle)
                    
        structure_data[atomIndex]['nbs']=neighbors
        structure_data[atomIndex]['angles']=angles
    return structure_data


# %%
chunkSize = 1000

coord_problem = []
data_chunk_list = []
for i,chunk in enumerate(chunker(df.pyst,chunkSize)):
    print('beginning data chunk #', i)
    data_chunk_name = "coordata_" + str(i)
    data_chunk_list.append(data_chunk_name + '.npy') 
    if os.path.exists(os.path.join(currentLoc, subdirectory,data_chunk_name+'.npy')): ###COMMENT THIS OUT if you want to overwrite previous data!!!!!!!!
        continue
    print('calculating coordination for the range', i*chunkSize, 'to', min((i+1)*chunkSize, len(df)))
    chunkCoord_problem, tempArray = batch_coord_light(chunk, coord_problem=True)
    coord_problem.append(chunkCoord_problem)  #unravel this list later
    np.save(os.path.join(currentLoc,subdirectory,data_chunk_name)
            , tempArray)
    # if i == 1:
    #     break
    del tempArray

# %%
print("Files to be featurized are", *data_chunk_list )
# the separation is there to delete the tempArray and not to overflow the memory
crystal_features = np.empty(0)  #to initiate appending data below
for i,data_chunk_name in enumerate(data_chunk_list):
    print('Featurizing data chunk ', data_chunk_name)
    res= []
    lse_data = np.load(os.path.join(currentLoc,subdirectory,data_chunk_name ),
                      allow_pickle=True)
    print('Calculating features for ', data_chunk_name  )
    for j,crystal in enumerate(lse_data):
        if j%100==0:
            print('Data #', j)
        try:
            lse = crystal['lse']
            oxid_states = crystal["oxidation"]
            material_id = crystal['material_id']
            structure = crystal['struc']
            res.append({'feats' :lightNBfeats(lse,oxid_states, structure), 'material_id':material_id })
        except:
            continue
    np.save(os.path.join(currentLoc,subdirectory,'tmp_feats'),np.array(res))
    with open(os.path.join(currentLoc,subdirectory,data_chunk_name+'_featurized.log'), 'w') as file:
        file.write('')
    crystal_features = np.concatenate((crystal_features,np.array(res)))  #collects the features of different chuncks of data.
    # break

# %%
np.save(os.path.join(currentLoc,'crystal_features')
            , crystal_features)

# %%
