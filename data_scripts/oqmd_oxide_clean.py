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
from ase.io import write
# %%
print('Starting to clean.')
raw_location = "data/raw"
# clean_location = "data/clean_data/"
clean_location = 'schnet_pred/data'
# experimental_path =  os.path.join(raw_location,"experimental_raw_oxygen.npy")
theoretical_path =  os.path.join(raw_location,"oqmd_raw_oxygen.npy")
# %%
# experimental_oxygens = np.load(experimental_path, allow_pickle=True)
theoretical_oxygens = np.load(theoretical_path, allow_pickle=True)

# %%
# print("Initially we have ", len(experimental_oxygens), "experimental data and ")
print(f"Initially we have {len(theoretical_oxygens)} theoretical data points. ")
# if data_path == theoretical_path:
# convert to pymatgen structure if format is different 
for material in theoretical_oxygens: #we prepare {'material_id', 'structure'} for data cleaning
    material["atoms"] = JarvisAtoms.from_dict(material["atoms"]) #build object from dict
    material["structure"] = jarvis_to_pymatgen(material["atoms"]) #convert to pymatgen
    material["material_id"] = material.pop('_oqmd_entry_id')
# for material in experimental_oxygens:
#     material["structure"] = Structure.from_dict(material["structure"]) 
# %%
# good_experimental_data = clean_oxide(experimental=True, pymatgenArray = experimental_oxygens, reportBadData=False)
# %%
good_theoretical_data = clean_oxide(experimental=False, pymatgenArray = theoretical_oxygens, reportBadData=False)
# %%
# now we need to fix the fields.
keys_to_keep = ["material_id", "atoms", 
                "energy_above_hull", "formation_energy_per_atom", ]
# exp_keys = set(good_experimental_data[0].keys())
# key_exclusion_exp = exp_keys.difference(set(keys_to_keep))
theo_keys = set(good_theoretical_data[0].keys())
key_exclusion_theo = theo_keys.difference(set(keys_to_keep))
target = ["material_id","synth"]
# %%
# for material in good_experimental_data:
#     material["atoms"] = pymatgen_to_ase(material["structure"])
#     for key in key_exclusion_exp:
#         material.pop(key, None)
#     material["synth"] = 1
# %%
for material in good_theoretical_data:
    material["atoms"] = jarvisP_to_ase(material["atoms"])
    material['formation_energy_per_atom'] = material.pop('_oqmd_delta_e') #https://static.oqmd.org/static/docs/analysis.html
    material['energy_above_hull'] = material.pop('_oqmd_stability')
    for key in key_exclusion_theo:
        material.pop(key, None)
    material["synth"] = 0    
# %%
# full_data = np.concatenate((good_experimental_data,
#                             good_theoretical_data))
# %%
# full_data = pd.DataFrame.from_records(full_data)
oqmd_df = pd.DataFrame.from_records(good_theoretical_data)
# %%
oqmd_df["material_id"] = oqmd_df.material_id.map(str)  #string format for all material_id
# %%
oqmd_df = oqmd_df.loc[oqmd_df.astype(str).drop_duplicates(keep = False).index]
# we use string because atome object is unhashable; so drop_duplicates won't work directly.
oqmd_df = oqmd_df.drop_duplicates('material_id')
oqmd_df = oqmd_df.reset_index(drop = True)
# %%
# oqmd_df["schnet0"] = np.nan
# oqmd_df["alignn0"] = np.nan
# oqmd_df["coSchnet1"] = np.nan #cotraining SchNet on Alignn labels 1st time
# oqmd_df["coAlignn1"] = np.nan #cotraining Alignn on SchNet labels 1st time
# oqmd_df["coSchnet2"] = np.nan #cotraining SchNet on Alignn labels 2nd time
# oqmd_df["coAlignn2"] = np.nan #cotraining Alignn on SchNet labels 2nd time
# oqmd_df["coSchnet3"] = np.nan #cotraining SchNet on Alignn labels 3rd time
# oqmd_df["coAlignn3"] = np.nan #cotraining Alignn on SchNet labels 3rd time
# %%
if not os.path.exists(clean_location):
    os.mkdir(clean_location)
oqmd_df.to_pickle(os.path.join(clean_location, "oqmd_df.pkl"))
# %%
# %%
save_dir = "predict_target/label_alignn_format/poscars_for_synth_prediction/oqmd_oxygens"
# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

for i, row in oqmd_df.iterrows():
    atoms = row['atoms']
    filename = f"POSCAR-{row['material_id']}.vasp"
    filepath = os.path.join(save_dir, filename)
    write(filepath, atoms, format="vasp")
# %%
