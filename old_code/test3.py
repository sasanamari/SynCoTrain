
# %%
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('experiment_funcs.py')))
#%%
# path = os.path.join(os.path.dirname(__file__), '..', 'experiment_funcs.py')
# sys.path.append(path)
print('-----')
print(os.getcwd())
print('-----')
# for p in sys.path:
#     print(p)


# %%
from experiment_funcs import oxide_check, exper_data_cleaning, analyze_env
    
    # %%
import numpy as np
import logging
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


# %%
import pytest


# %%

f = []
d = []
for (i,(dirpath, dirnames, filenames)) in enumerate(os.walk(os.getcwd())):
    f.extend(filenames)
    d.extend(dirnames)
    # if i > 3:
    break
# print(f)
print(d)

testData = np.load('tests/testDataDir/testData.npy', allow_pickle=True)


# %%
def ababa(myval):
    myval = 'Mr.'
    return myval

# def get_gender_heading(myval):
#     if myval in ['Phil', 'Jay', 'Luke', 'Manny']:
#         return 'Mr.'
#     return 'Ms.'


@pytest.fixture
def male_names():
    # return ['Phil', 'Jay', 'Luke', 'Manny']
    return {'Phil':'Mr.', 'Jay':'Mr.', 'Luke':'Mr.', 'Manny':'Mr.'}

def test_ababa(male_names):
    for dat in testData:
        dat2 = dat.copy()
        del dat2
        assert 'Mr.' == male_names['Jay']
    # for name in male_names:
    #     assert 'Mr.' == male_names[name)    


@pytest.fixture
def actual_oxide_check():
    with open('testDataDir/oxide_check_dict.pkl', 'rb') as f:
        oxide_check_dict = pickle.load(f)
    return oxide_check_dict    

# %%
def test_oxide_check(actual_oxide_check):
    mytestData = np.load('testDataDir/testData.npy', allow_pickle=True)
    for Tdatum in mytestData:
    # for Tdatum in mytestData:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
        print(type(Tdatum['material_id']))
        print(Tdatum['material_id'])
        assert other_anion == actual_oxide_check[Tdatum['material_id']]['other_anion']
        assert other_oxidation == actual_oxide_check[Tdatum['material_id']]['other_oxidation']
        assert bad_structure == actual_oxide_check[Tdatum['material_id']]['bad_structure']
        assert np.isclose(primStruc.volume , actual_oxide_check[Tdatum['material_id']]['primStruc'].volume)  #there could be permutations or machine percision difference
        i = Tdatum['material_id']
        print(f'Oxide_check function tests were passed for material id {i}')


# %%
test_oxide_check(actual_oxide_check)


# %%
