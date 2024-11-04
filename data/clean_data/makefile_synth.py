# %%con
import warnings
import sys
import os
print(os.getcwd())
warnings.filterwarnings('ignore')
import numpy as np
print(sys.path)
from data_scripts.crystal_funcs import exper_oxygen_query
# %%
from mp_api.client import MPRester
import argparse
from pymatgen.analysis.structure_analyzer import OxideType
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Structure
from crystal_funcs import clean_oxide
import pandas as pd
from crystal_structure_conversion import pymatgen_to_ase

# %%
parser = argparse.ArgumentParser(
    description="Downloading experimental data."
)
parser.add_argument(
    "--MPID",
    default="",
    help="This is your Materials Project ID.",
)
args = parser.parse_args(sys.argv[1:])
MPID = args.MPID 
# %%
location = "data/raw"
clean_location = "data/clean_data/"
if not os.path.exists(location):
    os.mkdir(location)
destination_file = os.path.join(location, "experimental_raw_oxygen")
database_text = os.path.join(location, "MP_db_version.txt")

num_sites = (2,3)
dataFrame_name = 'miniTestSynthdf.pkl'
# #Uncomment the lines below if you wish you to query the data from scratch.
# num_sites = (1,150)
# dataFrame_name = 'synthDF'
# %%
pymatgenExpArray , db_version = exper_oxygen_query(MPID=MPID,
                # location = location,
                theoretical_data = False,
                num_sites = num_sites,
                fields = "default",
                )
pymatgenTheoArray , db_version = exper_oxygen_query(MPID=MPID,
                # location = location,
                theoretical_data = True,
                num_sites = num_sites,
                fields = "default",
                )
# %%
# np.save(destination_file, pymatgenExpArray)
# np.save(destination_file, pymatgenTheoArray)
# %%
print(db_version)
# with open(database_text, "w") as text_file:
#     text_file.write(db_version)
# %%
print(f"We retrieved {len(pymatgenExpArray)} experimental crystals from the Materials Project database.")
print(f"We retrieved {len(pymatgenTheoArray)} theoretical crystals from the Materials Project database.")
# %%
for material in pymatgenExpArray:
    material["structure"] = Structure.from_dict(material["structure"]) 
for material in pymatgenTheoArray:
    material["structure"] = Structure.from_dict(material["structure"])     
# %%
good_experimental_data = clean_oxide(experimental=True, pymatgenArray = pymatgenExpArray, 
                                    reportBadData=False) #also removes "experimental" crystals with e_above_hull > 1 eV
print(f"We have {len(good_experimental_data)} experimental oxides after cleaning.")
good_theoretical_data = clean_oxide(experimental=False, pymatgenArray = pymatgenTheoArray, 
                                    reportBadData=False)
print(f"We have {len(good_theoretical_data)} theoretical oxides after cleaning.")
# %%
keys_to_keep = ["material_id", "atoms", 
                "energy_above_hull", "formation_energy_per_atom", ]
current_keys = set(good_experimental_data[0].keys())
key_exclusion_exp = current_keys.difference(set(keys_to_keep))
# %%
for material in good_experimental_data:
    material["atoms"] = pymatgen_to_ase(material["structure"])
    for key in key_exclusion_exp:
        material.pop(key, None)
    material["synth"] = 1
    
for material in good_theoretical_data:
    material["atoms"] = pymatgen_to_ase(material["structure"])
    for key in key_exclusion_exp:
        material.pop(key, None)
    material["synth"] = 0    
# %%
experimental_df = pd.DataFrame.from_records(good_experimental_data)
theoretical_df = pd.DataFrame.from_records(good_theoretical_data)
# %%
synthDF = pd.concat([experimental_df,theoretical_df])
synthDF = synthDF.sample(frac=1, ignore_index=True) #shuffling experimental and theoretical data
synthDF["material_id"] = synthDF.material_id.map(str)  #string format for all material_id
experiments = ['schnet0', 'alignn0', 'coSchnet1', 'coAlignn1', 'coSchnet2',
               'coAlignn2', 'coSchnet3', 'coAlignn3']
synthDF[experiments] = np.nan
# %%
synthDF.to_pickle(os.path.join(clean_location, dataFrame_name))
print(f'The dataframe was saved in {os.path.join(clean_location, dataFrame_name)}.')
# %%
