# This is a modified version of a script with the same name in the ALIGNN repo.
# %%
import pandas as pd  #more conveneient for ehull setup
import os
from data_scripts.crystal_structure_conversion import ase_to_jarvis
from experiment_setup import current_setup
# %%

# %%
def prepare_alignn_data(experiment, small_data, ehull015):
    cs = current_setup(small_data=small_data, experiment=experiment, ehull015 = ehull015)
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