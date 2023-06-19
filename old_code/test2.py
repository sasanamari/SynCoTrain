# %%
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

# %%
sys.path.append(os.path.dirname(os.path.abspath('../experiment_funcs.py')))
# os.path.dirname(os.path.abspath('experiment_funcs.py'))

# %%
import pytest

# %%
from experiment_funcs import oxide_check, exper_data_cleaning, analyze_env

# %%
testData = np.load('testDataDir/testData.npy', allow_pickle=True)

# %%
def test_exper_data_cleaning():
    goodtest_location,other_anion_IDs, other_oxidation_IDs, valence_problem_IDs, bad_structure_IDs = exper_data_cleaning('testDataDir/testData.npy', location='testDataDir/',reportBadData=True)
    assert all(item in other_anion_IDs for item in ('mp-5634', 'mp-788'))
    assert len(other_anion_IDs) == 2
    assert 'mp-5634' not in other_oxidation_IDs
    assert any(item['material_id'] not in valence_problem_IDs for item in testData)
    assert any(item['material_id'] not in bad_structure_IDs for item in testData)
    print('Tests ran successfully for data cleaning.')
    return goodtest_location

# %%
# # This is used to calculate and save the actual results of the tests. Not to be included in the test script.
# oxide_check_dict = dict()
# for Tdatum in testData:
#     other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
#     temp_dict = dict((('other_anion',other_anion), ('other_oxidation',other_oxidation),
#      ('bad_structure',bad_structure), ('primStruc',primStruc)))
#     oxide_check_dict[Tdatum['material_id']] = temp_dict

# with open('testDataDir/oxide_check_dict.pkl', 'wb') as f:
#     pickle.dump(oxide_check_dict,f) 

# %%


# %%
@pytest.fixture()
def actual_oxide_check():
    with open('testDataDir/oxide_check_dict.pkl', 'rb') as f:
        oxide_check_dict = pickle.load(f)
    return oxide_check_dict


@pytest.fixture#()
def tff():
    return actual_oxide_check()

# @pytest.fixture#()
def tff2():
    return testData

@pytest.fixture()
def argument_printer():
    def _foo(*args, **kwargs):
        return (args, kwargs)

    return _foo





def ababa(myval):
    myval = 'Mr.'
    return myval

def get_gender_heading(myval):
    if myval in ['Phil', 'Jay', 'Luke', 'Manny']:
        return 'Mr.'
    return 'Ms.'



@pytest.fixture
def male_names():
    # return ['Phil', 'Jay', 'Luke', 'Manny']
    return {'Phil':'Mr.', 'Jay':'Mr.', 'Luke':'Mr.', 'Manny':'Mr.'}
def test_ababa(male_names):
    for dat in testData:
        assert 'Mr.' == male_names['Jay']
    # for name in male_names:
    #     assert 'Mr.' == male_names[name)    
# %%
# oxide_check_vals()[testData[0]['material_id']]

# %%
# testData[0]['structure']

# %%
def test_oxide_check(actual_oxide_check):
    # actual_oxide_check = a 
    # actual_oxide_check = actual_oxide_check  #useless
    # # actual_oxide_check = actual_oxide_check()  #used to check f
    # mytestData = b(tff)
    # with open('testDataDir/oxide_check_dict.pkl', 'rb') as f:
    #     actual_oxide_check = pickle.load(f)
    mytestData = np.load('testDataDir/testData.npy', allow_pickle=True)
    # mytestData = iter(np.load('testDataDir/testData.npy', allow_pickle=True))

    for Tdatum in mytestData:
    # for Tdatum in mytestData:
        other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
        assert other_anion == actual_oxide_check[Tdatum['material_id']]['other_anion']
        assert other_oxidation == actual_oxide_check[Tdatum['material_id']]['other_oxidation']
        assert bad_structure == actual_oxide_check[Tdatum['material_id']]['bad_structure']
        assert np.isclose(primStruc.volume , actual_oxide_check[Tdatum['material_id']]['primStruc'].volume)  #there could be permutations or machine percision difference
        i = Tdatum['material_id']
        print(f'Oxide_check function tests were passed for material id {i}')


# %%
# test_oxide_check(actual_oxide_check)

# %%
# # This is the test for the oxide_check function:
# with open('testDataDir/oxide_check_dict.pkl', 'rb') as f:
#     oxide_check_dict = pickle.load(f)

# def test_oxide_check(Tdatum, oxide_check_dict):
#     other_anion, other_oxidation, bad_structure, primStruc = oxide_check(Tdatum['structure'])
#     assert other_anion == oxide_check_dict[Tdatum['material_id']]['other_anion']
#     assert other_oxidation == oxide_check_dict[Tdatum['material_id']]['other_oxidation']
#     assert bad_structure == oxide_check_dict[Tdatum['material_id']]['bad_structure']
#     assert np.isclose(primStruc.volume , oxide_check_dict[Tdatum['material_id']]['primStruc'].volume)  #there could be permutations or machine percision difference
#     i = Tdatum['material_id']
#     print(f'Oxide_check function tests were passed for material id {i}')


# for Tdatum in testData:
#     test_oxide_check(Tdatum, oxide_check_dict)



# %%
goodtest_location = test_exper_data_cleaning()

# %%


# %%
# This is used to calculate and save the actual results of the tests. Not to be included in the test script.
# analyze_env_dict = dict()
# for Tdatum in testData:
#     oxid_states, sc = analyze_env(Tdatum['structure'], mystrategy='simple')
#     temp_dict = dict([('oxid_states',oxid_states), ('sc',sc)])
#     analyze_env_dict[Tdatum['material_id']] = temp_dict

# with open('testDataDir/analyze_env_dict.pkl', 'wb') as f:
#     pickle.dump(analyze_env_dict,f) 

# # %%
# # This is the test for the analyze_env function:
# with open('testDataDir/analyze_env_dict.pkl', 'rb') as f:
#     analyze_env_dict = pickle.load(f)

# def test_analyze_env(Tdatum, analyze_env_dict):
#     oxid_states, sc = analyze_env(Tdatum['structure'], mystrategy='simple')
#     assert oxid_states == analyze_env_dict[Tdatum['material_id']]['oxid_states']
#     assert sc.as_dict()['connectivity_graph'] == analyze_env_dict[Tdatum['material_id']]['sc'].as_dict()['connectivity_graph']
#       #check the part which is invariable and human-readable
#     i = Tdatum['material_id']
#     print(f'analyze_env function tests were passed for material id {i}')


# for Tdatum in testData:
#     test_analyze_env(Tdatum, analyze_env_dict)

# %%


# %%


# %%


# %%


# %%



