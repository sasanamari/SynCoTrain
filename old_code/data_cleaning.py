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

arrdata = np.load('arrdata.npy',allow_pickle=True)
print("The initial data length is", len(arrdata))


def oxide_check(j, datum):
    
    initStruc = datum["structure"]    
    bad_structure = False
    
    try:
        primStruc = initStruc.get_primitive_structure() #returns the primite structure

    except ValueError:     
        bad_structure = True #True indicates there was an error in calculating the primitive structure.
    
    bv = BVAnalyzer()  #This class implements a maximum a posteriori (MAP) estimation method to
    # determine oxidation states in a structure.
    oxid_stats = bv.get_valences(primStruc)

    for i, site in enumerate(primStruc): #checks for any element other than Oxygen with a negative oxidation state.
        if site.species_string != 'O':
            if oxid_stats[i] < 0:
                other_anion_list.append([j, datum["pretty_formula"]])
                return bad_structure, primStruc
            
        else: #checks for any Oxygen with an oxidation state other than '-2'
            if oxid_stats[i] != -2:
                other_oxidation_list.append([j, datum["pretty_formula"]])
                return bad_structure, primStruc
            
    return bad_structure, primStruc



other_anion_list = []
other_oxidation_list = []
valence_problem_list = []
bad_structure_list = []

for j, datum in enumerate(arrdata):
    
    try:
        bad_structure, primStruc = oxide_check(j=j, datum=datum)
        
        datum["structure"] = primStruc #replaces the existing structure with the primitive structure

    except ValueError: 
        valence_problem_list.append([j, datum["pretty_formula"]]) #lists entries for which the oxidation states (valences) couldn't be analyzed.
        
    if bad_structure:
        bad_structure_list.append([j, datum["pretty_formula"]]) #lists any entry with error while calculating primitive structure.
        
        

anion_ind = [i[0] for i in other_anion_list]
print("The number of entries with anions other than Oxygen were", len(anion_ind))

oxid_ind = [i[0] for i in other_oxidation_list]
print("The number of entries with different oxidation types were", len(oxid_ind))

valence_ind = [i[0] for i in valence_problem_list]
print("The number of entries where valence/oxidation could not be analyzed were", len(valence_ind))

structure_ind = [i[0] for i in bad_structure_list]
print("The number of entries where the primitive structure could not be calculated were", len(structure_ind))

        
        
goodata = np.delete(arrdata, [*anion_ind, *oxid_ind, *valence_ind, *structure_ind])
del arrdata
np.save("goodata", goodata)
print("The length of data after removing undesired entries is ",len(goodata))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
