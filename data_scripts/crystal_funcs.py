# This supresses warnings.
from struct import Struct
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import logging
import os
import pandas as pd
# from pymatgen.ext.matproj import MPRester  #legacy version
# from mp_api.client import MPRester
import requests
import json
import pickle
from typing import Iterable
from pymatgen.analysis.structure_analyzer import OxideType
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

from typing import Generator, Iterable, Sequence, Any, Union, List, Optional, Tuple


# def exper_oxygen_query(MPID : str, 
#                      theoretical_data = False, 
#                      num_sites = (1,150),
#                      fields : Union[str, List[str]] = "default",) -> str:
#     '''Retrieves experimental (ICSD) crystallographic data through the Materials Project API.
#     Currently queries for crystals which contain Oxide anions, are not theoretical, and have at least 2 elements.
#     It stores their (pretty) formula, structure object and material_id.
#     Parameters:
#     ----------------
#     MPID : string
#         The user ID from Materials Project which allows data query.
#     Location : string
# 	The directory in which the downloaded data will be stored.
# '''

#     if fields == "default":
#         fields = ["material_id", 
#                 "formula_pretty", 
#                 'composition',
#                 "structure",
#                 "e_total",
#                 'elements',
#                 'energy_above_hull',
#                 'energy_per_atom',
#                 'formation_energy_per_atom',
#                 'is_stable',
#                 'theoretical',
#                 'deprecated',
#                 ]
    
#     with MPRester(MPID) as mpr:
#         db_version = mpr.get_database_version()
#     # If you get error 404, try upgrading mp-api from the terminal.
#         results = mpr.summary.search(
#             elements=["O"],
#             num_elements = (2,100),
#             theoretical=theoretical_data,
#             num_sites = num_sites,  #change this to (1,150) when pipeline is ready.
#         #  all_fields=True,
#             fields=fields
#                                     )
        
#     print("Database version is ", db_version)    
#     print("The number of entries retrieved is ", len(results))
#     results = [d.dict() for d in results]  #so it can be pickled/saved on disk.
#     _ = [d.pop('fields_not_requested') for d in results]
#     del _
    
        
#     arrdata = np.array(results) #converts list to array, much faster to work with
#     del results #free up memory
    
#     # if return_data:
#         # return arrdata
#     return arrdata, db_version



def oxide_check(initStruc : Structure) -> Tuple[bool, bool, bool, Structure]:
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
        bad_structure = True #True indicates there was an error in calculating the primitive structure/valances.
        return other_anion, other_oxidation, bad_structure, primStruc  #function quits for bad structures
    
    bv = BVAnalyzer()  #This class implements a maximum a posteriori (MAP) estimation method to
    # determines oxidation states in a structure.
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



def clean_oxide(experimental : bool, pymatgenArray : np.ndarray, 
                reportBadData : bool = False, read_oxide_type : bool = True) -> np.ndarray:
    '''Filters undesired data points from the pymatgen data.
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
        Returns  cleaned data, 
        and optionally the four lists of undesired data which is removed during this cleaning. Useful for testing.
    '''
    print("The initial data length is", len(pymatgenArray))

    other_anion_IDs = []
    other_oxidation_IDs = []
    valence_problem_IDs = []
    bad_structure_IDs = []
    ustable_experimental_IDs = []

    for j, material in enumerate(pymatgenArray):       
        if read_oxide_type:
            if OxideType(material["structure"]).oxide_type != "oxide":
                other_oxidation_IDs.append([j,material['material_id']])
                continue
# read_oxide_type is not necessary, but it speeds up the filtering when available.            
        if experimental:
            if material['energy_above_hull']>1:
                ustable_experimental_IDs.append([j,material['material_id']])
                continue
        try:
            other_anion, other_oxidation, bad_structure, primStruc = oxide_check(initStruc=material["structure"])
            if other_anion:
                other_anion_IDs.append([j,material['material_id']])
            elif other_oxidation:
                other_oxidation_IDs.append([j,material['material_id']])
            elif bad_structure:
                bad_structure_IDs.append([j,material['material_id']])
            else: #this else only refers to the previous if
                material["structure"] = primStruc #replaces the existing structure with the primitive structure

        except ValueError: 
            valence_problem_IDs.append([j,material['material_id']])

    print("The number of entries with anions other than Oxygen were", len(other_anion_IDs))
    print("The number of entries with different oxidation types were", len(other_oxidation_IDs))
    print("The number of entries where valence/oxidation could not be analyzed were", len(valence_problem_IDs))
    print("The number of entries where the primitive structure could not be calculated were", len(bad_structure_IDs))

    anion_ind = [i[0] for i in other_anion_IDs]
    unstable_ind = [i[0] for i in ustable_experimental_IDs]
    oxid_ind = [i[0] for i in other_oxidation_IDs]
    valence_ind = [i[0] for i in valence_problem_IDs]
    structure_ind = [i[0] for i in bad_structure_IDs]
                        
    goodata = np.delete(pymatgenArray, [*anion_ind, *unstable_ind, *oxid_ind, *valence_ind, *structure_ind])
    # del pymatgenArray
    print("The length of data after removing undesired entries is ",len(goodata))   
    
    if not reportBadData:
        return goodata      

    other_anion_IDs = [i[1] for i in other_anion_IDs]
    other_oxidation_IDs = [i[1] for i in other_oxidation_IDs]
    valence_problem_IDs = [i[1] for i in valence_problem_IDs]
    bad_structure_IDs = [i[1] for i in bad_structure_IDs]

    return goodata, other_anion_IDs, ustable_experimental_IDs, other_oxidation_IDs, valence_problem_IDs, bad_structure_IDs

