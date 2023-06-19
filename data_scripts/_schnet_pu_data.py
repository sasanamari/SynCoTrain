# %%
import numpy as np
from crystal_structure_conversion import *
import os
from pymatgen.core import Structure, Element

# %%
# theoretical_dev_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/theoretical/"
experimental_dev_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/experimental/"
# np.random.seed(42)
pAtoms = np.load(experimental_dev_path+"PosAtomsUnd12arr.npy", allow_pickle=True)
# tAtoms = np.load(theoretical_dev_path+"TheoAtomsUnd12arr.npy", allow_pickle=True)
# %%
sample = pAtoms[0]
sample['energy_above_hull'] = sample.pop('e_above_hull') #key consistency

# %%
# tAtoms[0]
# %%
experimental_path = "/home/samariam/projects/chemheuristics/data/experimental"
experimentData = os.path.join(experimental_path, "goodata.npy")
theoretical_path = "/home/samariam/projects/chemheuristics/data/theoretical"
theoData = os.path.join(theoretical_path, "goodata.npy")
# %%
pData = np.load(experimentData, allow_pickle=True)
# %%
exclusion = pData[0].keys()-sample.keys()
# %%
for d in pData:
    d["atoms"] = pymatgen_to_ase(d["structure"])
    # d['e_above_hull'] = d.pop('energy_above_hull')
    for key in exclusion:
        d.pop(key, None)
    
# %% 
final_experimental_path = os.path.join(experimental_path,"schnet_experimental_data")
np.save(final_experimental_path,pData)
# %%
del pData #heavy on memory
# %%
tData = np.load(theoData, allow_pickle=True)
# %%
exclusion = tData[0].keys()-sample.keys()
for d in tData:
    d["atoms"] = jarvisP_to_ase(d["atoms"])
    d["theoretical"] = True
    d['formation_energy_per_atom'] = d.pop('_oqmd_delta_e')/d["atoms"].get_global_number_of_atoms()
    d['energy_above_hull'] = d.pop('_oqmd_stability')
    for key in exclusion:
        d.pop(key, None)
# %%
final_theoretical_path = os.path.join(theoretical_path,"schnet_theoretical_data")
np.save(final_theoretical_path,tData)
# %%
