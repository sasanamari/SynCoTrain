# This is a modified version of a script with the same name in the ALIGNN repo.
# %%
import pandas as pd  #more conveneient for ehull setup
import os
from data_scripts.crystal_structure_conversion import ase_to_jarvis
# from experiment_setup import current_setup
import sys
import argparse
# %%
parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--prop",
    default="synth",
    help="The property to predict; synth or stability.",
)
# %%
def prepare_alignn_labels(prop='synth'):
    if prop == 'synth':
        labelPath = "data/results/synth/synth_labels"
    if prop == 'stability':
        labelPath = "data/results/stability/stability_labels"

    SynthCoTrain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labelPath = os.path.join(SynthCoTrain_dir, labelPath)
    # %%
    labeldf = pd.read_pickle(labelPath)
    labeldf[prop] = labeldf[prop].astype('int16')
    # the order should be first postive, then unlabeld class.    
    # max_samples = 100
    data_dest = "predict_target/label_alignn_format"
    data_dest = os.path.join(SynthCoTrain_dir, data_dest)
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    f = open(os.path.join(data_dest, 
                    f"{prop}_id_from_cotrain.csv"), "w")

    count = 0

    # %%
    data_files_dir = os.path.join(data_dest,f"atomistic_{prop}_labels")
    if not os.path.exists(data_files_dir):
        os.makedirs(data_files_dir)
    for _,row in labeldf.iterrows():
        jid = row["material_id"]
        poscar_name = "POSCAR-" + str(jid) + ".vasp"
        target_value = row[f"{prop}_labels"]
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
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    prepare_alignn_labels()