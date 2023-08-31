# %%
from jarvis.db.figshare import data
import pickle
import os
import numpy as np
# %matplotlib inline
# %%
location = "data/raw"
destination_file = os.path.join(location, "theoretical_raw_oxygen")
database_text = os.path.join(location, "database_citation.txt")

if not os.path.exists(location):
    os.makedirs(location)
# %%
oqmd_list = data('oqmd_3d_no_cfid')
# other_dbs = ["dft_3d", "cfid_3d"]

# %%
oxygens = []
print("initial number of oqmd data is",len(oqmd_list))
for d in oqmd_list:
    elems = d['atoms']["elements"]
    if 0<len(elems)<150: #filter few structures with more than 150 atoms.
        if "O " in elems:
            oxygens.append(d)        
print("And the number of structures with oxygen is",len(oxygens))
del oqmd_list
oxygens = np.array(oxygens)
# file_name = "oxygens_"+str(len(oxygens))+"_datapoints"
# file_path = os.path.join(location, file_name)
# with open(file_path, "wb") as fp:
#     pickle.dump(oxygenscheck this, fp)

# %%
np.save(destination_file, oxygens)
# %%
database_citation = "https://doi.org/10.6084/m9.figshare.14206169.v1"
with open(database_text, "w") as text_file:
    text_file.write(database_citation)
# %%
