# This is a modified version of a script with the same name in the ALIGNN repo.
# %%
import pandas as pd  #more conveneient for ehull setup
import os
from synth.data_scripts.crystal_structure_conversion import ase_to_jarvis
from experiment_setup import current_setup
# %%

# %%
def prepare_alignn_data(experiment, ehull_test, small_data):
    cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment)
    propDFpath = cs["propDFpath"]
    # result_dir = cs["result_dir"]
    prop = cs["prop"]
    TARGET = cs["TARGET"]
    data_prefix = cs["dataPrefix"]
    print(f'You have selected {TARGET} as your training target!')

    SynthCoTrain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    propDFpath = os.path.join(SynthCoTrain_dir, propDFpath)
    # %%
    crysdf = pd.read_pickle(propDFpath)
    crysdf[prop] = crysdf[prop].astype('int16')
    # the order should be first postive, then unlabeld class.    
    # max_samples = 100
    data_dest = "data/clean_data/alignn_format"
    data_dest = os.path.join(SynthCoTrain_dir, data_dest)
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    f = open(os.path.join(data_dest, 
                    f"{data_prefix}{prop}_id_from_{TARGET}.csv"), "w")

    count = 0

    # %%
    data_files_dir = os.path.join(data_dest,f"{data_prefix}atomistic_{prop}_{experiment}")
    if not os.path.exists(data_files_dir):
        os.makedirs(data_files_dir)
    for _,row in crysdf.iterrows():
        jid = row["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target_value = row[TARGET]
    #The line below keeps NaN values in the dataframe. Used for running small_data experiments.
        if pd.notna(target_value):
            formatted_target = "%6f" % target_value
        else:
            formatted_target = "NaN"
        jarvisAtom = ase_to_jarvis(row["atoms"])
        jarvisAtom.write_poscar(os.path.join(data_files_dir, poscar_name))
        f.write("%s,%s\n" % (poscar_name, formatted_target))
        count += 1
            # if count == max_samples:
            #     break
    f.close()
    
    return f'Data was prepared in {data_files_dir} directory.'
# %%
# I move ahead to pu_data_selection, to add this script as a function.\
#     Then I'll have to come back to the rest below.
# need to figure out data_dest for normal, small and ehull data.
# need to f'PUOutput_{experiment}' with f'PUehull_{experiment}' in PU_alignn for ehull test.
# also modify alignn analysis to work well with ehull and small data.
# at the end, make this file into a function and run it inside pu_data_selection.
# # data/clean_data/alignn_format/synth_id_from_coSchAl2.csv
# # I'll need to adapt: target, and adapt csv name.
# alignn config should also be modified to accomodate stability; fix it from scratch.
# Also, the analysis files need to adapt to target.
# Alignn analysis is done, haven't touched schnet yet.
# # For starters, I can just make a classification reverse ehull column > done!
# %%
'''import pandas as pd  #more conveneient for ehull setup
import sys
import numpy as np
import os
from ase import Atoms as AseAtoms
from jarvis.core.atoms import Atoms
import json
from synth.data_scripts.crystal_structure_conversion import ase_to_jarvis
import argparse
from experiment_setup import current_setup
# %%

parser = argparse.ArgumentParser(
    description="Data preparation for ALIGNN"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)
parser.add_argument(
    "--ehull",
    default=False, #change this! manually defined cos lazy.
    help="Predicting stability to evaluate PU Learning's efficacy.",
)
parser.add_argument(
    "--small_data",
    default=False, #change this! manually defined cos lazy.
    help="Run the synthesizability experiment with smaller data to check the pipeline.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment
ehull_test = args.ehull
small_data = args.small_data
cs = current_setup(ehull_test=ehull_test, small_data=small_data)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
data_prefix = "small_" if small_data else ""

cotraining = False
experiment_target_match = { #output_dir: training_label_column
            'alignn0':prop, 
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
            'coAlSch3':'coSchAl2',
            'coAlSch4':'coSchAl3',
            'coAlSch5':'coSchAl4',
            'final_class':'change_to_desired_label'
    }
if not experiment!='alignn0': #Works for synth and for ehull.
    cotraining = True
    TARGET = experiment_target_match[experiment]    
    print(f'You have selected {TARGET} as your reference column!!!!!!!!!')

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
# if ehull_test:
crysdf = pd.read_pickle(propDFpath)
crysdf[prop] = crysdf[prop].astype('int16')
# the order should be first postive, then unlabeld class.    
# %%
print('CWD is ',os.getcwd())

# max_samples = 100
data_dest = "data/clean_data/alignn_format"


if cotraining:
    f = open(os.path.join(data_dest, 
                f"{data_prefix}{prop}_id_from_{TARGET}.csv"), "w")
    target = TARGET
else:
    f = open(os.path.join(data_dest, 
                f"{data_prefix}{prop}_ids.csv"), "w")
    target = prop
count = 0

# %%
data_files_path = os.path.join(data_dest,f"{data_prefix}atomistic_{prop}_{experiment}")
if not os.path.exists(data_files_path):
    os.makedirs(data_files_path)
for _,row in crysdf.iterrows():
    jid = row["material_id"]
    poscar_name = "POSCAR-" + str(jid) + ".vasp"
    target_value = row[target]
#The line below keeps NaN values in the dataframe. Used for running small_data experiments.
    if pd.notna(target_value):
        formatted_target = "%6f" % target_value
    else:
        formatted_target = "NaN"
    jarvisAtom = ase_to_jarvis(row["atoms"])
    jarvisAtom.write_poscar(os.path.join(data_files_path, poscar_name))
    f.write("%s,%s\n" % (poscar_name, formatted_target))
    count += 1
        # if count == max_samples:
        #     break
f.close()'''
# %%
# %%
# t = pd.read_csv(os.path.join(data_dest, 
#                 "synth_id_from_"+reference_col+".csv"))
# %%



# # %%
# import pandas as pd  #more conveneient for ehull setup
# import sys
# import numpy as np
# import os
# from ase import Atoms as AseAtoms
# from jarvis.core.atoms import Atoms
# import json
# from synth.data_scripts.crystal_structure_conversion import ase_to_jarvis
# import argparse
# # %%
# parser = argparse.ArgumentParser(
#     description="Data preparation for ALIGNN"
# )
# parser.add_argument(
#     "--experiment",
#     default="alignn0",
#     help="name of the experiment and corresponding config files.",
# )
# args = parser.parse_args(sys.argv[1:])
# experiment = args.experiment
# cotraining = False
# if experiment!='alignn0':
#     cotraining = True
    
# experiment_train_match = { #output_dir: training_label_column
#             'alignn0':'synth',
#             'coAlSch1':'schnet0',
#             'coAlSch2':'coSchAl1',
#     }
# reference_col = experiment_train_match[experiment]    
    

# if cotraining:
#     print(f'You have selected {reference_col} as your reference column!!!!!!!!!')
# reverse_label = False
# ehull_test = False
# normal_experiment = False
# if sum([cotraining, reverse_label, ehull_test, normal_experiment])!= 1:
#     print("Contradiction in experiment setup!!!")
#     exit()
# old_crysdf = 'dummy_var until I update the rest.'
# # %%
# def ase_to_atoms(ase_atoms="", cartesian=True):
#     """Convert ase structure to Atoms."""
#     return Atoms(
#         lattice_mat=ase_atoms.get_cell(),
#         elements=ase_atoms.get_chemical_symbols(),
#         coords=ase_atoms.get_positions(),
#         cartesian=cartesian,
#     )
# # %%
# crysdf = pd.read_pickle("data/clean_data/synthDF")
# crysdf["synth"] = crysdf.synth.astype('int16')
# # the order should be first postive, then unlabeld class.
# # %%
# # if cotraining:
# #     pAtoms, tAtoms = cotrain_labeling_schnet(pAtoms, tAtoms, new_labels_path)
    
# # %%
# print('CWD is ',os.getcwd())
# if ehull_test:
#     prop = "pu_ehull"
# elif cotraining:
#     prop = reference_col
# else:
#     prop = "synth"
# max_samples = 100
# data_dest = "/home/samariam/projects/synth/data/clean_data/alignn_format"
# if reverse_label: #reverse labels on pretrained model
#     # f = open("alignn/sample_synth/synth_id_prop_rev.csv", "w")
#     f = open(os.path.join(data_dest, "synth_id_prop_rev.csv"), "w")
# elif cotraining:
#     f = open(os.path.join(data_dest, 
#                 "synth_id_from_"+reference_col+".csv"), "w")
# elif ehull_test:
#     old_crysdf[['material_id','ehull_class', 'pu_ehull']].to_pickle(
#         os.path.join(data_dest, "ehull_test_ref")
#     )
#     f = open(os.path.join(data_dest, "ehull_test.csv"), "w")
# else:
#     # f = open("alignn/sample_synth/synth_id_prop.csv", "w")
#     f = open(os.path.join(data_dest,"synth_id_prop.csv"), "w")
# count = 0
# # %%
# if ehull_test:
#     for _,row in old_crysdf.iterrows():
#         # atoms = Atoms.from_dict(i["atoms"])
#         jid = row["material_id"]
#         poscar_name = "POSCAR-" + str(jid) + ".vasp"
#         target = row[prop]
#         if target != "na":
#             row["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
#             f.write("%s,%6f\n" % (poscar_name, target))
#             count += 1
#     f.close()
# else:
#     for _,row in crysdf.iterrows():
#         # atoms = Atoms.from_dict(i["atoms"])
#         jid = row["material_id"]
#         poscar_name = "POSCAR-" + str(jid) + ".vasp"
#         target = row[prop]
#         if target != "na":
#             jarvisAtom = ase_to_jarvis(row["atoms"])
#             jarvisAtom.write_poscar(os.path.join(data_dest, poscar_name))
#             # i["atoms"].write_poscar(os.path.join(data_dest, poscar_name))
#             f.write("%s,%6f\n" % (poscar_name, target))
#             count += 1
#             # if count == max_samples:
#             #     break
#     f.close()
# # %%
# # t = pd.read_csv(os.path.join(data_dest, 
# #                 "synth_id_from_"+reference_col+".csv"))
# # %%
