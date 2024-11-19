import numpy as np

from jarvis.db.figshare import data

# citation:
# https://doi.org/10.6084/m9.figshare.14206169.v1

# output filenames
destination_file = "oqmd_raw_oxygen.npy"

# database name
oqmd_list = data('oqmd_3d_no_cfid')

# get oxides
oxygens = []
print("initial number of oqmd data is",len(oqmd_list))
for d in oqmd_list:
    elems = d['atoms']["elements"]
    # filter few structures with more than 150 atoms
    if 0 < len(elems) < 150:
        if "O " in elems:
            oxygens.append(d)        

print("The number of structures with oxygen is", len(oxygens))

# convert list to numpy arraw
oxygens = np.array(oxygens)

# save result
np.save(destination_file, oxygens)
