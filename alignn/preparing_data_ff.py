# %%
import pandas as pd
import numpy as np
import os
import json
from ase import Atoms as AseAtoms
from jarvis.core.atoms import Atoms
import random
# %%
def ase_to_atoms(ase_atoms="", cartesian=True):
    """Convert ase structure to Atoms."""
    return Atoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        cartesian=cartesian,
    )
# %%
df = pd.read_json('/home/samariam/projects/alignn/alignn/examples/sample_data_ff/id_prop.json')
# %%
# df.head()
# %%
theoretical_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/theoretical/"
experimental_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/experimental/"
mydatapath = "/home/samariam/projects/chemheuristics/data_for_dev_try/testData4alignn.json"
# %%
pAtoms = np.load(experimental_path+"PosAtomsUnd12arr.npy", allow_pickle=True)
tAtoms = np.load(theoretical_path+"TheoAtomsUnd12arr.npy", allow_pickle=True)
# #array of dicst, has ase atom objects.
data_types = list(tAtoms[0].keys())
# %%
# pSynth = [np.array(1).flatten()]*len(pAtoms)    #we need the array to have the shape (1,), hence we use flatten()
# tSynth = [np.array(0).flatten()]*len(tAtoms)

pSynth = [np.array(1, dtype=np.int16).flatten()]*len(pAtoms)    
tSynth = [np.array(0, dtype=np.int16).flatten()]*len(tAtoms)

# pSynth = [int(1)]*len(pAtoms)
# tSynth = [int(0)]*len(tAtoms)
# %%
# t = df.atoms.iloc[0]
# %%
def loadjson(filename=""):
    """Provide helper function to load a json file."""
    f = open(filename, "r")
    d = json.load(f)
    f.close()
    return d
root_dir = '/home/samariam/projects/alignn/alignn/examples/sample_data_ff'
dat = loadjson(os.path.join(root_dir, "id_prop.json"))

# %%
config = loadjson(os.path.join(root_dir, "config_example_atomwise.json"))
# %%
crysData = np.concatenate([pAtoms, tAtoms])
targetData = [*pSynth, *tSynth]   #again, we need distinct arrays. np.concatenate would merge all in one array.

for i in range(len(targetData)):
    crysData[i]['synth'] = targetData[i]
    
# %%
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def dumpjson(data=[], filename="", indent = None):
    """Provide helper function to write a json file."""
    f = open(filename, "w")
    f.write(json.dumps(data, cls=NumpyArrayEncoder, indent=indent))
    f.close()

# %%
for datum in crysData:
    datum['atoms'] = ase_to_atoms(datum['atoms']).to_dict()



# %%
np.random.seed(42)
np.random.shuffle(crysData)  #shuffles positive and negative class
# %%
# c3 = json.dumps(crysData, cls=NumpyArrayEncoder, indent=2)
dumpjson(crysData, filename=mydatapath, indent=2)
# %%
# t3 = json.loads(c3)
# %%
mydata = loadjson(mydatapath)
# %%
