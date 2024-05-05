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
    description="Downloading theoretical data."
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
new_destination_file = os.path.join(location, "new_icsd_theoretical_raw_oxygen")
new_database_text = os.path.join(location, "theoretical_MP_db_version.txt")

# %%
pymatgenArray , db_version = exper_oxygen_query(MPID=MPID,
                theoretical_data = True,
                num_sites = (1,150),
                fields = "default",
                )
# %%
print(f"We retrieved {len(pymatgenArray)} theoretical crystals from the Materials Project database.")
# np.save(new_destination_file, pymatgenArray)
# # %%
# with open(new_database_text, "w") as text_file:
#     text_file.write(db_version)
# %%
for material in pymatgenArray:
    material["structure"] = Structure.from_dict(material["structure"]) 
# %%
# all_new_IDs = [i['material_id'] for i in pymatgenArray]
# synthDF = pd.read_pickle("data/clean_data/synthDF")
# new_IDs = [id for id in all_new_IDs if id not in synthDF.material_id.values]
# # we removed the old data from the data in the new version of the database.
# new_pymatgenArray = np.array([entry for entry in pymatgenArray if entry['material_id'] in new_IDs])
# %%
good_theoretical_data = clean_oxide(experimental=False, pymatgenArray = pymatgenArray, 
                                    reportBadData=False)
print(f"We have {len(good_theoretical_data)} theoretical oxides after cleaning.")
# %%
keys_to_keep = ["material_id", "atoms", 
                "energy_above_hull", "formation_energy_per_atom", ]
theoretical_keys = set(good_theoretical_data[0].keys())
key_exclusion_exp = theoretical_keys.difference(set(keys_to_keep))
# %%
for material in good_theoretical_data:
    material["atoms"] = pymatgen_to_ase(material["structure"])
    for key in key_exclusion_exp:
        material.pop(key, None)
    material["synth"] = 0
# %%
new_theoretical_df = pd.DataFrame.from_records(good_theoretical_data)
# %%
new_theoretical_df["material_id"] = new_theoretical_df.material_id.map(str)  #string format for all material_id

# %%
new_theoretical_df.to_pickle(os.path.join(clean_location, "theoretical_ICSD_df.pkl"))
# %%
save_dir = "predict_target/label_alignn_format/poscars_for_synth_prediction/theoretical_icsd_oxides"
# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

for i, row in new_theoretical_df.iterrows():
    atoms = row['atoms']
    filename = f"POSCAR-{row['material_id']}.vasp"
    filepath = os.path.join(save_dir, filename)
    write(filepath, atoms, format="vasp")