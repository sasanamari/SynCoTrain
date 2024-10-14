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
location = "data/raw"
if not os.path.exists(location):
    os.mkdir(location)
destination_file = os.path.join(location, "experimental_raw_oxygen")
database_text = os.path.join(location, "MP_db_version.txt")

# %%
pymatgenArray , db_version = exper_oxygen_query(MPID=MPID,
                # location = location,
                theoretical_data = False,
                num_sites = (1,150),
                fields = "default",
                # return_data=True
                )


# %%
np.save(destination_file, pymatgenArray)
# %%
with open(database_text, "w") as text_file:
    text_file.write(db_version)
# %%
