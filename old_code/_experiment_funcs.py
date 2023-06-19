# This supresses warnings.
from struct import Struct
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import logging
import os
import pandas as pd
from pymatgen.ext.matproj import MPRester
import requests
import json
import pickle
from typing import Iterable
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import ConnectivityFinder
from pymatgen.util.coord import get_angle
from pymatgen.core.structure import Structure
from pymatgen.analysis.chemenv.connectivity.structure_connectivity import StructureConnectivity

from typing import Generator, Iterable, Sequence, Any, Union



def exper_data_query(MPID : str, location : str=   "data/", experimental_data = True) -> str:
    '''Retrieves experimental (ICSD) crystallographic data through the Materials Project API.
    Currently queries for crystals which contain Oxide anions, are not theoretical, and have at least 2 elements.
    It stores their (pretty) formula, structure object and material_id.
    Parameters:
    ----------------
    MPID : string
        The user ID from Materials Project which allows data query.
    Location : string
	The directory in which the downloaded data will be stored.
'''
    # request below is just made to print the version of the database and pymatgen
    response = requests.get(
        "https://www.materialsproject.org/rest/v2/materials/mp-1234/vasp",
        {"API_KEY": MPID})
    response_data = json.loads(response.text)
    print(response_data.get('version'))
    # request above is just made to print the version of the database and pymatgen

    if experimental_data:
        criteria={
                "icsd_ids": {"$ne": []}, #allows data with existing "icsd_ids" tag
                "theoretical": {"$ne": experimental_data}, #allows data without the "theoretical" tag
                "elements": {"$all": ["O"]}, #allows for crystals with Oxygen present
                "oxide_type": {"$all": ["oxide"]}, #allows for oxides (e.g. not peroxide)
                "nelements": {"$gte": 2}, #allows crystals with at least 2 elements
                "nsites" : {"$lte": 12}
                    }
    else:
        criteria={
                # we don't need this limit for theoretical data "icsd_ids": {"$ne": []}, #allows data with existing "icsd_ids" tag
                "theoretical": {"$ne": experimental_data}, #allows data without the "theoretical" tag
                "elements": {"$all": ["O"]}, #allows for crystals with Oxygen present
                "oxide_type": {"$all": ["oxide"]}, #allows for oxides (e.g. not peroxide)
                "nelements": {"$gte": 2}, #allows crystals with at least 2 elements
                "nsites" : {"$lte": 12}
                    }
                
    

    with MPRester(api_key=MPID) as mpr:

        data = mpr.query(
        
            criteria,
            
            properties=[
                "exp.tags", "icsd_ids", "formula", "pretty_formula", "structure",
                "material_id", "theoretical", "formation_energy_per_atom", "e_above_hull"
                        ]   
            
                        )
            
    arrdata = np.array(data) #converts list to array, much faster to work with
    del data #free up memory

    destination = location+"arrdata"
    np.save(destination, arrdata)

    return destination+'.npy'



def oxide_check(initStruc : Structure) -> tuple[bool, bool, bool, Structure]:
    '''Checks the oxidation states of the atoms in primary cell. Retruns the primary structure, and
        booleans for whether the structure is bad, whether anions other than Oxygen are present and 
        whether the oxidatoin number of Oxygen is anything other tha -2.
        Parameters:
        ----------------
        initStruc : Structure
            The Structure object as it was queried from MP database.
    '''

    primStruc = initStruc  #just in case the conversion to primitive doesn't work, the type remains consistent when function returns    
    bad_structure = False
    other_anion = False
    other_oxidation = False
    
    try:
        primStruc = initStruc.get_primitive_structure() #returns the primite structure

    except ValueError:     
        bad_structure = True #True indicates there was an error in calculating the primitive structure.
        return other_anion, other_oxidation, bad_structure, primStruc  #function quits for bad structures

    
    bv = BVAnalyzer()  #This class implements a maximum a posteriori (MAP) estimation method to
    # determine oxidation states in a structure.
    oxid_stats = bv.get_valences(primStruc)

    for i, site in enumerate(primStruc): #checks for any element other than Oxygen with a negative oxidation state.
        if site.species_string != 'O':
            if oxid_stats[i] < 0:
                other_anion = True
                return other_anion, other_oxidation, bad_structure, primStruc
            
        else: #checks for any Oxygen with an oxidation state other than '-2'
            if oxid_stats[i] != -2:
                other_oxidation = True
                return other_anion, other_oxidation, bad_structure, primStruc
            
    return other_anion, other_oxidation, bad_structure, primStruc



# one thrid of the data is cleaned away. We have slightly less data points than before. Not sure why.
def exper_data_cleaning(queried_data_string : str, location : str = "data/", reportBadData : bool = False) -> str:
    '''Filters undesired data from the stored experimental data.
    Undesired data here include: 1- structures which cannot be converted to primitive cell. 
    2- data the oxidation states of which cannot be analyzed. 
    3- Include any anion element which is not Oxygen. 
    4- Include Oxygen with any oxidation number other than -2.
    Parameters:
    ----------------
    queried_data_string : string
        The address of the downloaded experimental data.
    Location : string
        The directory in which the cleaned data will be stored.
    reportBadData : bool
        Returns the four lists of undesired data which is removed during this cleaning. Useful for testing.
    '''

    arrdata = np.load(queried_data_string,allow_pickle=True)
    print("The initial data length is", len(arrdata))

    other_anion_IDs = []
    other_oxidation_IDs = []
    valence_problem_IDs = []
    bad_structure_IDs = []

    for j, datum in enumerate(arrdata):
        
        try:
            other_anion, other_oxidation, bad_structure, primStruc = oxide_check(initStruc=datum["structure"])
            
            if other_anion:
                # bad_datum = dict({'data_index':j, 'material_id':datum['material_id'], 'formula':datum["formula"]})
                other_anion_IDs.append([j,datum['material_id']])
            if other_oxidation:
                # bad_datum = dict({'data_index':j, 'material_id':datum['material_id'], 'formula':datum["formula"]})
                other_oxidation_IDs.append([j,datum['material_id']])
            
            if bad_structure:
                # bad_datum = dict({'data_index':j, 'material_id':datum['material_id'], 'formula':datum["formula"]})
                bad_structure_IDs.append([j,datum['material_id']])
            else: #this else only refers to the previous if
                datum["structure"] = primStruc #replaces the existing structure with the primitive structure

        except ValueError: 
            # bad_datum = dict({'data_index':j, 'material_id':datum['material_id'], 'formula':datum["formula"]})
            valence_problem_IDs.append([j,datum['material_id']])


    print("The number of entries with anions other than Oxygen were", len(other_anion_IDs))

    print("The number of entries with different oxidation types were", len(other_oxidation_IDs))

    print("The number of entries where valence/oxidation could not be analyzed were", len(valence_problem_IDs))

    print("The number of entries where the primitive structure could not be calculated were", len(bad_structure_IDs))

    anion_ind = [i[0] for i in other_anion_IDs]
    oxid_ind = [i[0] for i in other_oxidation_IDs]
    valence_ind = [i[0] for i in valence_problem_IDs]
    structure_ind = [i[0] for i in bad_structure_IDs]
                        
    goodata = np.delete(arrdata, [*anion_ind, *oxid_ind, *valence_ind, *structure_ind])
    del arrdata
    print("The length of data after removing undesired entries is ",len(goodata))
    
    goodata_location = location+"goodata"
    np.save(location+"goodata", goodata)    

    other_anion_IDs = [i[1] for i in other_anion_IDs]
    other_oxidation_IDs = [i[1] for i in other_oxidation_IDs]
    valence_problem_IDs = [i[1] for i in valence_problem_IDs]
    bad_structure_IDs = [i[1] for i in bad_structure_IDs]

    if reportBadData:
        return goodata_location+'.npy', other_anion_IDs, other_oxidation_IDs, valence_problem_IDs, bad_structure_IDs
    return goodata_location+'.npy'



def analyze_env(struc : Structure, mystrategy : str = "simple") -> tuple[list[int], StructureConnectivity]:
    '''Analyzes the coordination environments and returns the StructureConnectivity object for the crystal and the list of oxidation states.
    First, BVAnalyzer() calculates the oxidation states. Then, the LocalGeometryFinder() computes the structure_environment object, 
    from which the LightStructureEnvironment (LSE) is derived. Finally, The ConnectivityFinder() builds the StructureConnectivity (SE) based on LSE. 
    At the end only the SE is returned, as it includes the LSE object as an attribute.
    Parameters:
    ----------------
    struc : Structure 
        crystal Structure object from pymatgen
    mystrategy : string
	    The simple or combined strategy for calculating the coordination environments.
'''
    if mystrategy == "simple":
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
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
    se = lgf.compute_structure_environments(only_cations=True,valences=oxid_states,
    additional_conditions=[AdditionalConditions.ONLY_ANION_CATION_BONDS])
    
    # Get LightStructureEnvironments
    lse = LightStructureEnvironments.from_structure_environments(
    strategy=strategy, structure_environments=se)

    # Get StructreConnectuvuty object
    cf= ConnectivityFinder()
    sc=cf.get_structure_connectivity(light_structure_environments=lse)

    return oxid_states, sc



def chunker(seq : Sequence, size : int) -> Generator[Any, None, None]: #if this doesn't work, try Iterator or Iterator[Any]
    """chunks the indices of an iterable object to pieces of a given size.
    Parameters:
    ----------------
    seq : Iterable 
        The iterable object (list, tuple etc) to be chunked in smaller pieces.
    size : int
        the length of each chunk (not necessarily the last one).
    """
    ### not writing a test unless I need to change it. Currently, it's plain python syntax.
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def firstDegreeFeatures(SC_object : StructureConnectivity, oxidation_list : list[int], struct : Structure) -> dict:
    '''Calculates the desired primary features (related to the atom and nearest neighbors) based on SC object, 
        returns them as a dictionary. These features are stored for each atom, under their structure index.
        Features Include: Oxidation number, type of ion, element, coordination for all atoms.
        Cation specific features are the local(coordination) env and nearest neighbor elements & distances.
        Parameters:
        ----------------
        SC_object : StructureConnectivity
        oxidation_list : list[int]
            A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
        struct : Structure
            crystal Structure object from pymatgen
        '''
    LSE = SC_object.light_structure_environments #takes lightStructureEnvironment Obj from StructureConnecivity Obj
    local_Envs_list = LSE.coordination_environments #takes coordination/local environments from lightStructureEnvironment Obj
    structure_data : dict = {}
    for atomIndex, atom in enumerate(LSE.neighbors_sets):
        
        structure_data[atomIndex] = {}
        structure_data[atomIndex]['oxidation'] = oxidation_list[atomIndex]
        
        if atom==None:
            # save coordniates here 
            structure_data[atomIndex]['ion'] = 'anion'
            structure_data[atomIndex]['element'] = struct[atomIndex].species_string
            structure_data[atomIndex]['coords'] = struct[atomIndex].coords
            continue #This skips further featurization. We're not analyzing envs with anions.
        

        structure_data[atomIndex]['ion'] = 'cation'
        structure_data[atomIndex]['element'] = struct[atomIndex].species_string
        structure_data[atomIndex]['coords'] = struct[atomIndex].coords
        structure_data[atomIndex]['localEnv'] = local_Envs_list[atomIndex]#[0] return a list including a single dict. Shoulbe be multiple dicts in the compound strategy.
        neighbors = atom[0].neighb_sites_and_indices
        structure_data[atomIndex]['NN_distances'] = []
        for nb in neighbors:
            nb_element = nb['site'].species_string
            nb_distance = nb['site'].distance_from_point(struct[atomIndex].coords)
            structure_data[atomIndex]['NN_distances'].append([nb_distance,nb_element])
    
    return structure_data


def nnnFeatures(SC_object : StructureConnectivity, struct : Structure, structure_data : dict) -> dict:
    '''Calculates the desired NNN features based on SC object, and addes them to a dictionary (of primary features).
        These features are stored for each atom, under their structure index.
        NNN features Include: Polhedral neighbor elements, distances, connectivity angles & types. 
        Parameters:
        ----------------
        SC_object : StructureConnectivity
        oxidation_list : list[int]
            A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
        struct : Structure
            crystal Structure object from pymatgen
        structure_data : dict
            A dictionary containing primary features of the crystal. The NNN features will be added under the same atom index.
        '''
    nodeS=SC_object.environment_subgraph().nodes()
    for node in nodeS:
        distances=[]
        node_angleS = []

        for edge in SC_object.environment_subgraph().edges(node, data=True):

            ### NNN distance calculation
            distance=struct[edge[2]["start"]].distance(struct[edge[2]["end"]], edge[2]["delta"])
            start_element = struct[edge[2]["start"]].species_string
            end_element = struct[edge[2]["end"]].species_string

            if node.atom_symbol!= end_element: #can't see an order on which side edge starts.
                neighbor_element = end_element
            else:
                neighbor_element = start_element  #this way if the 2 elements are different, the other name is saved.

            distance = [distance, neighbor_element]
            distances.append(distance) #this will be recorded as distance for this edge (NNN) for this node (atom of interest)


            ### NNN angles calculation
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
                print('There was a problem with the connectivity.')

            edge_angleS : list[Union[list, dict]]= []
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
                
                angle = get_angle(cart_pos0-cart_pos1, cart_pos2-cart_pos3, units="degrees")

                if edge[2]['start'] != node.isite:
                    poly_nb = structure_data[edge[2]['start']]['element']
                else:
                    poly_nb = structure_data[edge[2]['end']]['element']  

                edge_angleS.append([angle, poly_nb])

            node_angleS.append(edge_angleS)

        structure_data[node.isite]['poly_distances']=distances 
        structure_data[node.isite]['connectivity_angles']=node_angleS
    
    return structure_data




def crysFeaturizer(SC_object : StructureConnectivity, oxidation_list : list[int]) -> dict:
    '''Calls firstDegreeFeatures() & nnnFeatures() functions to calculate the desired features 
        based on SC object, returns them as a dictionary. These features are stored for each atom,
        under their structure index.
        Features Include: Oxidation number, type of ion, element, coordination for all atoms.
        Cation specific features are the local(coordination) env, nearest neighbor elements & distances, 
        polhedral neighbor elements, distances, connectivity angles & types. 
        Parameters:
        ----------------
        SC_object : StructureConnectivity
        oxidation_list : list[int]
            A list of oxidation numbers of the atoms in the crystal with the same order as the atoms' index.
        '''
    struct = SC_object.light_structure_environments.structure #takes structure from StructureConnecivity Obj

    first_structure_data : dict = firstDegreeFeatures(SC_object=SC_object, oxidation_list = oxidation_list, struct=struct)
    # computing first degree features
    structure_data : dict = nnnFeatures(SC_object = SC_object, struct = struct, structure_data = first_structure_data)
    # adding NNN features
        
    return structure_data

