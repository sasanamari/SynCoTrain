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
import pickle
from pymatgen.analysis.structure_analyzer import OxideType
from pymatgen.analysis.bond_valence import BVAnalyzer

# %%
MPID="your materials project ID here."
location = "data/raw"
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
# not_oxide = []
# for i,material in enumerate(pymatgenArray):
#     s = material["structure"]
#     if OxideType(s).oxide_type != "oxide":
#         not_oxide.append(i)
        
# pymatgenArray = np.delete(pymatgenArray, not_oxide)

# %%
np.save(destination_file, pymatgenArray)
# %%
with open(database_text, "w") as text_file:
    text_file.write(db_version)
# %%
