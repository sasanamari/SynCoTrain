# %%
from crystal_structure_conversion import jarvis_to_pymatgen
import pickle
from pymatgen.core import Structure, Element
from ase import Atoms as AseAtoms
from pymatgen.io.ase import AseAtomsAdaptor as pase
from jarvis.core.atoms import Atoms as JarvisAtoms
from pymatgen.analysis.structure_analyzer import OxideType
from experiment_funcs import exper_data_cleaning
import numpy as np
# %%
location = "/home/samariam/projects/chemheuristics/data/theoretical"
# %%
with open("/home/samariam/projects/chemheuristics/data/theoretical/oxygens_114409_datapoints", "rb") as fp:
    oxygens = pickle.load(fp)

# %%
for d in oxygens:
    d["atoms"] = JarvisAtoms.from_dict(d["atoms"])
    d["structure"] = jarvis_to_pymatgen(d["atoms"])
    d["material_id"] = d.pop('_oqmd_entry_id')
    
# %%
print("Initially we have ", len(oxygens), "data points.")
not_oxide = []
for i,material in enumerate(oxygens):
    s = material["structure"]
    if OxideType(s).oxide_type != "oxide":
        # not_oxide.append(i)
        del oxygens[i]

print("Next we are left with ", len(oxygens), "oxides.")
        
# arrdata = np.delete(arrdata, not_oxide)

# %%
good_location = exper_data_cleaning(location=location, arrdata=oxygens,)
# %%
