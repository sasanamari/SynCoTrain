# %%
import os
from crystal_structure_conversion import jarvis_to_pymatgen,pymatgen_to_ase, jarvisP_to_ase
# import pickle
from pymatgen.core import Structure, Element
# from ase import Atoms as AseAtoms
# from pymatgen.io.ase import AseAtomsAdaptor as pase
from jarvis.core.atoms import Atoms as JarvisAtoms
# from pymatgen.analysis.structure_analyzer import OxideType
from crystal_funcs import clean_oxide
import numpy as np
import pandas as pd
# %%
print('Starting to clean.')
raw_location = "data/raw"
clean_location = "data/clean_data/"
experimental_path =  os.path.join(raw_location,"experimental_raw_oxygen.npy")
theoretical_path =  os.path.join(raw_location,"theoretical_raw_oxygen.npy")
# %%
experimental_oxygens = np.load(experimental_path, allow_pickle=True)
theoretical_oxygens = np.load(theoretical_path, allow_pickle=True)

# %%
print("Initially we have ", len(experimental_oxygens), "experimental data and ")
print(len(theoretical_oxygens), "theoretical data points. ")
# if data_path == theoretical_path:
# convert to pymatgen structure if format is different 
for material in theoretical_oxygens: #we prepare {'material_id', 'structure'} for data cleaning
    material["atoms"] = JarvisAtoms.from_dict(material["atoms"]) #build object from dict
    material["structure"] = jarvis_to_pymatgen(material["atoms"]) #convert to pymatgen
    material["material_id"] = material.pop('_oqmd_entry_id')
for material in experimental_oxygens:
    material["structure"] = Structure.from_dict(material["structure"]) 
# %%
good_experimental_data = clean_oxide(experimental_oxygens, reportBadData=False)
# %%
good_theoretical_data = clean_oxide(theoretical_oxygens, reportBadData=False)
# %%
# now we need to fix the fields.
keys_to_keep = ["material_id", "atoms", 
                "energy_above_hull", "formation_energy_per_atom", ]
exp_keys = set(good_experimental_data[0].keys())
key_exclusion_exp = exp_keys.difference(set(keys_to_keep))
theo_keys = set(good_theoretical_data[0].keys())
key_exclusion_theo = theo_keys.difference(set(keys_to_keep))
target = ["material_id","synth"]
# %%
for material in good_experimental_data:
    material["atoms"] = pymatgen_to_ase(material["structure"])
    for key in key_exclusion_exp:
        material.pop(key, None)
    material["synth"] = 1
# %%
for material in good_theoretical_data:
    material["atoms"] = jarvisP_to_ase(material["atoms"])
    material['formation_energy_per_atom'] = material.pop('_oqmd_delta_e')/material["atoms"].get_global_number_of_atoms()
    material['energy_above_hull'] = material.pop('_oqmd_stability')
    for key in key_exclusion_theo:
        material.pop(key, None)
    material["synth"] = 0    
# %%
full_data = np.concatenate((good_experimental_data,
                            good_theoretical_data))
# %%
full_data = pd.DataFrame.from_records(full_data)
# %%
full_data["material_id"] = full_data.material_id.map(str)  #string format for all material_id
# %%
full_data = full_data.loc[full_data.astype(str).drop_duplicates(keep = False).index]
# we use string because atome object is unhashable; so drop_duplicates won't work directly.
full_data = full_data.drop_duplicates('material_id')
full_data = full_data.reset_index(drop = True)
# %%
full_data["schnet0"] = np.nan
full_data["alignn0"] = np.nan
full_data["coSchAl1"] = np.nan #cotraining SchNet on Alignn labels 1st time
full_data["coAlSch1"] = np.nan #cotraining Alignn on SchNet labels 1st time
full_data["coSchAl2"] = np.nan #cotraining SchNet on Alignn labels 2nd time
full_data["coAlSch2"] = np.nan #cotraining Alignn on SchNet labels 2nd time
full_data["coSchAl3"] = np.nan #cotraining SchNet on Alignn labels 3rd time
full_data["coAlSch3"] = np.nan #cotraining Alignn on SchNet labels 3rd time
# %%
if not os.path.exists(clean_location):
    os.mkdir(clean_location)
full_data.to_pickle(os.path.join(clean_location, "synthDF"))
# %%
# #checking out the data
# testdf = pd.read_pickle(os.path.join(clean_location, "synthDF"))
# # %%
# testdf["schnet0"] = np.nan
# testdf["alignn0"] = np.nan
# testdf["coSchAl1"] = np.nan #cotraining SchNet on Alignn labels 1st time
# testdf["coAlSch1"] = np.nan #cotraining Alignn on SchNet labels 1st time
# testdf["coSchAl2"] = np.nan #cotraining SchNet on Alignn labels 2nd time
# testdf["coAlSch2"] = np.nan #cotraining Alignn on SchNet labels 2nd time
# # # %%
# ## testdf = testdf.loc[testdf.astype(str).drop_duplicates(keep = False).index]
# ## # we use string because atome object is unhashable.
# ## testdf = testdf.reset_index(drop = True)
# # %%
# ## testdf.to_pickle(os.path.join(clean_location, "synthDF"))

# # %%
# tardf = testdf[["material_id", "synth"]].copy()
# # %%
# tardf["schlab0"] = np.nan
# tardf["aliglab0"] = np.nan
# # %%
# testdf= testdf.drop(columns = "synth")
# # %%
# # newdf = pd.merge(testdf, tardf, on="material_id")
# # %%
# newdf = pd.merge(testdf.drop_duplicates(subset='material_id'), 
#                  tardf, on='material_id')
# %%
