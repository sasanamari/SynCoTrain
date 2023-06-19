# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import logging
from pymatgen.ext.matproj import MPRester
import numpy as np

# %% [markdown]
# ### Structure analysis:

# %%
import logging
from pymatgen.util.coord import get_angle
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import ConnectivityFinder
import os


# %%



# %%
def crysFeaturizer(SC_object):
    struct = SC_object.light_structure_environments.structure 
    LSE = SC_object.light_structure_environments
    local_Envs_list = LSE.coordination_environments 
    structure_data = {}
    

    for i, atom in enumerate(LSE.neighbors_sets):
        if atom==None:
            # save coordniates here here
            atomIndex = i #just for naming consistency 
            structure_data[atomIndex] = {}
            structure_data[atomIndex]['index'] = atomIndex
            structure_data[atomIndex]['ion'] = 'anion'
            structure_data[atomIndex]['element'] = struct[atomIndex].species_string
            structure_data[atomIndex]['coords'] = struct[atomIndex].coords
            continue #not working with anions.
        atomIndex = atom[0].isite

        if i != atomIndex:
            print('The neighbor sets are not ordered according to the structure object!')
            break   #something for saving anionic coordinates here. Later do in cleanly, cations and anions in one go.

        structure_data[atomIndex] = {}
        structure_data[atomIndex]['index'] = atomIndex
        structure_data[atomIndex]['ion'] = 'cation'
        structure_data[atomIndex]['element'] = struct[atomIndex].species_string
        structure_data[atomIndex]['localEnv'] = local_Envs_list[atomIndex]
        structure_data[atomIndex]['coords'] = struct[atomIndex].coords
        neighbors = atom[0].neighb_sites_and_indices
        structure_data[atomIndex]['NN_distances'] = []
        for nb in neighbors:
            nb_element = nb['site'].species_string
            nb_distance = nb['site'].distance_from_point(struct[atomIndex].coords)
            structure_data[atomIndex]['NN_distances'].append([nb_distance,nb_element])

    nodes=SC_object.environment_subgraph().nodes()
    for node in nodes:
        distances=[]
        node_angleS = []

#         node_data = {}
#         node_data['element']=node.atom_symbol

        for edge in SC_object.environment_subgraph().edges(node, data=True):


            distance=struct[edge[2]["start"]].distance(struct[edge[2]["end"]], edge[2]["delta"])
            start_element = struct[edge[2]["start"]].species_string
            end_element = struct[edge[2]["end"]].species_string

            if node.atom_symbol!= end_element: #can't see an order on which side edge starts.
                neighbor_element = end_element
            else:
                neighbor_element = start_element  #this way if the 2 elements are different, the other name is saved.

            distance = [distance, neighbor_element]

            distances.append(distance)

            ligandS = edge[2]["ligands"]

            connectivity = {}
            if len(ligandS) == 0:
                connectivity['kind'] = "noConnection"
            if len(ligandS) == 1:
                connectivity['kind'] = "corner"
            elif len(ligandS) == 2:
                connectivity['kind'] = "edge"
            elif len(ligandS) >= 3:
                connectivity['kind'] = "face"
            else:
                print('There was a problem with the connectivity analysis at')
#                 add no connection as the connectivity type

            edge_angleS = []
            edge_angleS.append(connectivity)
            for ligand in ligandS:
                pos0=struct[ligand[1]["start"]].frac_coords
                pos1=struct[ligand[1]["end"]].frac_coords+ligand[1]["delta"]
                cart_pos0 = struct.lattice.get_cartesian_coords(pos0)
                cart_pos1 = struct.lattice.get_cartesian_coords(pos1)

                pos2=struct[ligand[2]["start"]].frac_coords
                pos3=struct[ligand[2]["end"]].frac_coords+ligand[2]["delta"]
                cart_pos2 = struct.lattice.get_cartesian_coords(pos2)
                cart_pos3 = struct.lattice.get_cartesian_coords(pos3)
                
                if edge[2]['start'] != node.isite:
                    poly_nb = structure_data[edge[2]['start']]['element']
                else:
                    poly_nb = structure_data[edge[2]['end']]['element']  

                angle = get_angle(cart_pos0-cart_pos1, cart_pos2-cart_pos3, units="degrees")
                edge_angleS.append([angle, poly_nb])

            node_angleS.append(edge_angleS)

        structure_data[node.isite]['poly_distances']=distances 

        structure_data[node.isite]['connectivity_angles']=node_angleS

        
    return structure_data


# %%



# %%
data_chunk_list = [file for file in os.listdir('data') if file.startswith("coordata_")]
# data_chunk_list = ['coordata_10.npy']

print("Files to be featurized are", *data_chunk_list )

crystal_features = np.empty(0)
for chunk in data_chunk_list:
    res= []
    sc_data = np.load('data/'+chunk,allow_pickle=True)
    print('Calculating features for ', chunk)
    for datum in sc_data:
        try:
            SC = datum['StructureConn']
            material_id = datum['material_id']
        except:
            continue
        res.append({'material_id':material_id, 'feats' :crysFeaturizer(SC)})
        # res.append(crysFeaturizer(SC))
    
    crystal_features = np.concatenate((crystal_features,np.array(res)))



# %%

    np.save('data/local_crystal_features', crystal_features) #this does not need to be indented?
    # np.save('data/crystal_features', crystal_features) 



