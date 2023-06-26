# %%
import pandas as pd  #more conveneient for ehull setup
import sys
import numpy as np
import os
from ase import Atoms as AseAtoms
from jarvis.core.atoms import Atoms
import json
from synth.data_scripts.crystal_structure_conversion import ase_to_jarvis
import argparse
# %%
parser = argparse.ArgumentParser(
    description="Data preparation for ALIGNN"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment
cotraining = False
if experiment!='alignn0':
    cotraining = True
    
experiment_train_match = { #output_dir: training_label_column
            'alignn0':'synth',
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
    }
reference_col = experiment_train_match[experiment]    
    

if cotraining:
    print(f'You have selected {reference_col} as your reference column!!!!!!!!!')
reverse_label = False
ehull_test = False
normal_experiment = False
if sum([cotraining, reverse_label, ehull_test, normal_experiment])!= 1:
    print("Contradiction in experiment setup!!!")
    exit()
old_crysdf = 'dummy_var until I update the rest.'
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
crysdf = pd.read_pickle("data/clean_data/synthDF")
crysdf["synth"] = crysdf.synth.astype('int16')
# the order should be first postive, then unlabeld class.
# %%
# if cotraining:
#     pAtoms, tAtoms = cotrain_labeling_schnet(pAtoms, tAtoms, new_labels_path)
    
# %%
print('CWD is ',os.getcwd())
if ehull_test:
    prop = "pu_ehull"
elif cotraining:
    prop = reference_col
else:
    prop = "synth"
max_samples = 100
data_dest = "/home/samariam/projects/synth/data/clean_data/alignn_format"
if reverse_label: #reverse labels on pretrained model
    # f = open("alignn/sample_synth/synth_id_prop_rev.csv", "w")
    f = open(os.path.join(data_dest, "synth_id_prop_rev.csv"), "w")
elif cotraining:
    f = open(os.path.join(data_dest, 
                "synth_id_from_"+reference_col+".csv"), "w")
elif ehull_test:
    old_crysdf[['material_id','ehull_class', 'pu_ehull']].to_pickle(
        os.path.join(data_dest, "ehull_test_ref")
    )
    f = open(os.path.join(data_dest, "ehull_test.csv"), "w")
else:
    # f = open("alignn/sample_synth/synth_id_prop.csv", "w")
    f = open(os.path.join(data_dest,"synth_id_prop.csv"), "w")
count = 0
# %%
if ehull_test:
    for _,row in old_crysdf.iterrows():
        # atoms = Atoms.from_dict(i["atoms"])
        jid = row["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target = row[prop]
        if target != "na":
            row["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
            f.write("%s,%6f\n" % (poscar_name, target))
            count += 1
    f.close()
else:
    for _,row in crysdf.iterrows():
        # atoms = Atoms.from_dict(i["atoms"])
        jid = row["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target = row[prop]
        if target != "na":
            jarvisAtom = ase_to_jarvis(row["atoms"])
            jarvisAtom.write_poscar(os.path.join(data_dest, poscar_name))
            # i["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
            f.write("%s,%6f\n" % (poscar_name, target))
            count += 1
            # if count == max_samples:
            #     break
    f.close()
# %%
# t = pd.read_csv(os.path.join(data_dest, 
#                 "synth_id_from_"+reference_col+".csv"))
# %%
