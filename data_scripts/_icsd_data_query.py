# %%con
import warnings
import sys
import os
print(os.getcwd())
warnings.filterwarnings('ignore')
import numpy as np
print(sys.path)
from experiment_funcs import exper_data_query, exper_data_cleaning
# %%
from mp_api.client import MPRester
import pickle
from pymatgen.analysis.structure_analyzer import OxideType
from pymatgen.analysis.bond_valence import BVAnalyzer

# %%
MPID="xOsYnlHNKBgOvQFhIthUJ68KAMxqrCBL"
location = "/home/samariam/projects/chemheuristics/data/experimental"
# %%
pymatgenArray = exper_data_query(MPID=MPID,
                location = location,
                theoretical_data = False,
                num_sites = (1,150),
                fields = "default",
                return_data=True)
# %%
not_oxide = []
for i,material in enumerate(pymatgenArray):
    s = material["structure"]
    if OxideType(s).oxide_type != "oxide":
        not_oxide.append(i)
        
pymatgenArray = np.delete(pymatgenArray, not_oxide)
# %%
good_location = exper_data_cleaning(location=location, arrdata=pymatgenArray)

# %%
