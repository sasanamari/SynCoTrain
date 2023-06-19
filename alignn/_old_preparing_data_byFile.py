# %%
import pandas as pd  #more conveneient for ehull setup
import sys
sys.path.append('/home/samariam/projects/chemheuristics')
import numpy as np
import os
from ase import Atoms as AseAtoms
from jarvis.core.atoms import Atoms
import json
from data_scripts.cotraining_labeling import cotrain_labeling_schnet
# %%
cotraining = True
reverse_label = False
ehull_test = False
normal_experiment = False
if sum([cotraining, reverse_label, ehull_test, normal_experiment])!= 1:
    print("Contradiction in experiment setup!!!")
    exit()

# %%
def ase_to_atoms(ase_atoms="", cartesian=True):
    """Convert ase structure to Atoms."""
    return Atoms(
        lattice_mat=ase_atoms.get_cell(),
        elements=ase_atoms.get_chemical_symbols(),
        coords=ase_atoms.get_positions(),
        cartesian=cartesian,
    )
os.chdir('/home/samariam/projects/chemheuristics')
# theoretical_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/theoretical/"
# experimental_path = "/home/samariam/projects/chemheuristics/data_for_dev_try/experimental/"
# mydatapath = "/home/samariam/projects/chemheuristics/data_for_dev_try/testData4alignn.json"
# %%
# pAtoms = np.load(experimental_path+"PosAtomsUnd12arr.npy", allow_pickle=True)
# tAtoms = np.load(theoretical_path+"TheoAtomsUnd12arr.npy", allow_pickle=True)
full_experimental_data_path = "/home/samariam/projects/chemheuristics/data/experimental/schnet_experimental_data.npy"
full_theoretical_data_path = "/home/samariam/projects/chemheuristics/data/theoretical/schnet_theoretical_data.npy"
new_labels_path = '/home/samariam/projects/chemheuristics/data/schnet_0.pkl'

pAtoms = np.load(full_experimental_data_path, allow_pickle=True)
tAtoms = np.load(full_theoretical_data_path, allow_pickle=True)
if cotraining:
    pAtoms, tAtoms = cotrain_labeling_schnet(pAtoms, tAtoms, new_labels_path)
    

# # #array of dicst, has ase atom objects.
# # data_types = list(tAtoms[0].keys())
# # moved this to alignn_setup.py
# data_size = {"positive_data_size":len(pAtoms),
#              "unlabeled_data_size":len(tAtoms)}

# with open("alignn/dataSize.json", "w") as f:
#     f.write(json.dumps(data_size))    

# %%
# pSynth = [np.array(1).flatten()]*len(pAtoms)    #we need the array to have the shape (1,), hence we use flatten()
# tSynth = [np.array(0).flatten()]*len(tAtoms)
if reverse_label: #reverse labels on pretrained model
    pSynth = [np.array(0, dtype=np.int16).flatten()]*len(pAtoms)    
    tSynth = [np.array(1, dtype=np.int16).flatten()]*len(tAtoms)
elif ehull_test:
    phull = [d["energy_above_hull"] for d in pAtoms]
    thull = [d["energy_above_hull"] for d in tAtoms]
    # separating into positive and negative class with threshold 0.1:
    phull = [1 if d<0.1 else 0 for d in phull]
    thull = [1 if d<0.1 else 0 for d in thull]
else:
    pSynth = [np.array(1, dtype=np.int16).flatten()]*len(pAtoms)    
    tSynth = [np.array(0, dtype=np.int16).flatten()]*len(tAtoms)

# pSynth = [int(1)]*len(pAtoms)
# tSynth = [int(0)]*len(tAtoms)
# %%
crysData = np.concatenate([pAtoms, tAtoms])
if ehull_test:
    targetData = [*phull, *thull]   #again, we need distinct arrays. np.concatenate would merge all in one array.
    # targetData.sort(reverse=True)  #orders positive class (of ehull) first
    # BIG mistake: sorted targets before connecting to atoms.
    for i in range(len(targetData)):
        crysData[i]['ehull_class'] = targetData[i]
    crysdf = pd.DataFrame(list(crysData), columns=crysData[0].keys())
    crysdf["pu_ehull"] = crysdf.ehull_class.copy()
    unlabeling_mask = crysdf[crysdf.ehull_class==1].sample(
        frac = 0.2, random_state = 3).index
    #selecting 20% of data with class 1 label
    crysdf['pu_ehull'].loc[unlabeling_mask] = 0
    crysdf = crysdf.sort_values('pu_ehull',
                                ascending= False, ignore_index=True)
    
else:
    targetData = [*pSynth, *tSynth]   #again, we need distinct arrays. np.concatenate would merge all in one array.
    for i in range(len(targetData)):
        crysData[i]['synth'] = targetData[i]
    
# %%
if ehull_test:
    crysdf["atoms"]=crysdf.atoms.map(ase_to_atoms)
else:
    for datum in crysData:
        datum['atoms'] = ase_to_atoms(datum['atoms'])#.to_dict() 
    # if we're saving as json, no need to convert to dictionary.
# %%
# np.random.seed(42)
# np.random.shuffle(crysData)  #shuffles positive and negative class
# %%
print('CWD is ',os.getcwd())
if ehull_test:
    prop = "pu_ehull"
else:
    prop = "synth"
max_samples = 10
data_dest = "/home/samariam/projects/chemheuristics/data/alignn_full_data"
if reverse_label: #reverse labels on pretrained model
    # f = open("alignn/sample_synth/synth_id_prop_rev.csv", "w")
    f = open(os.path.join(data_dest, "synth_id_prop_rev.csv"), "w")
elif cotraining:
    f = open(os.path.join(data_dest, "synth_id_prop_cotraining.csv"), "w")
elif ehull_test:
    crysdf[['material_id','ehull_class', 'pu_ehull']].to_pickle(
        os.path.join(data_dest, "ehull_test_ref")
    )
    f = open(os.path.join(data_dest, "ehull_test.csv"), "w")
else:
    # f = open("alignn/sample_synth/synth_id_prop.csv", "w")
    f = open(os.path.join(data_dest,"synth_id_prop.csv"), "w")
count = 0
# %%
if ehull_test:
    for _,i in crysdf.iterrows():
        # atoms = Atoms.from_dict(i["atoms"])
        jid = i["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target = i[prop]
        if target != "na":
            i["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
            f.write("%s,%6f\n" % (poscar_name, target))
            count += 1
    f.close()
else:
    for i in crysData:
        # atoms = Atoms.from_dict(i["atoms"])
        jid = i["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target = i[prop]
        if target != "na":
            i["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
            f.write("%s,%6f\n" % (poscar_name, target))
            count += 1
            # if count == max_samples:
            #     break
    f.close()
# %%
# os.getcwd()
# %%
# ###read output?
# import json
# def loadjson(filename=""):
#     """Provide helper function to load a json file."""
#     f = open(filename, "r")
#     d = json.load(f)
#     f.close()
#     return d
# # %%
# def loadjsonS(filename=""):
#     """Provide helper function to load a json file."""
#     f = open(filename, "r")
#     d = json.loads(f)
#     f.close()
#     return d
# # %%
# tvt_ids = loadjson('alignn/output/ids_train_val_test.json')
# %%
# import pandas as pd
# res = pd.read_csv('output/prediction_results_test_set.csv')
# %%
# %%
