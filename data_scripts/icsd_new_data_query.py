# %%
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
import os
from ase.io import write
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
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
location = "data/raw"
clean_location = "data/clean_data/"
if not os.path.exists(location):
    os.mkdir(location)
new_destination_file = os.path.join(location, "new_experimental_raw_oxygen")
new_database_text = os.path.join(location, "new_MP_db_version.txt")

# %%
pymatgenArray , db_version = exper_oxygen_query(MPID=MPID,
                # location = location,
                theoretical_data = False,
                num_sites = (1,150),
                fields = "default",
                # return_data=True
                )
# %%
# np.save(new_destination_file, pymatgenArray)
# # %%
# with open(new_database_text, "w") as text_file:
#     text_file.write(db_version)
# %%
for material in pymatgenArray:
    material["structure"] = Structure.from_dict(material["structure"]) 
# %%
all_new_IDs = [i['material_id'] for i in pymatgenArray]
synthDF = pd.read_pickle("data/clean_data/synthDF")
new_IDs = [id for id in all_new_IDs if id not in synthDF.material_id.values]
# we removed the old data from the data in the new version of the database.
new_pymatgenArray = np.array([entry for entry in pymatgenArray if entry['material_id'] in new_IDs])
# %%
good_experimental_data = clean_oxide(experimental=True, pymatgenArray = new_pymatgenArray, reportBadData=False)

# %%
keys_to_keep = ["material_id", "atoms", 
                "energy_above_hull", "formation_energy_per_atom", ]
exp_keys = set(good_experimental_data[0].keys())
key_exclusion_exp = exp_keys.difference(set(keys_to_keep))
# %%
for material in good_experimental_data:
    material["atoms"] = pymatgen_to_ase(material["structure"])
    for key in key_exclusion_exp:
        material.pop(key, None)
    material["synth"] = 1
# %%
new_experimental_df = pd.DataFrame.from_records(good_experimental_data)
# %%
new_experimental_df["material_id"] = new_experimental_df.material_id.map(str)  #string format for all material_id

# %%
new_experimental_df.to_pickle(os.path.join(clean_location, "new_ICSD_df.pkl"))
# %%
save_dir = "predict_target/label_alignn_format/poscars_for_synth_prediction/new_icsd_oxides"
# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

for i, row in new_experimental_df.iterrows():
    atoms = row['atoms']
    filename = f"POSCAR-{row['material_id']}.vasp"
    filepath = os.path.join(save_dir, filename)
    write(filepath, atoms, format="vasp")