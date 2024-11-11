import pandas as pd
import os

from syncotrainmp.utility.crystal_structure_conversion import ase_to_jarvis


def prepare_alignn_data(experiment, cs):
    """
    Prepares data for training by converting atomistic structures into a compatible format.

    Parameters:
    - experiment (str): The name of the experiment.
    - cs (dict): Current setup.

    Returns:
    - str: The path to the converted data.
    """
    # Load configuration and dataset paths
    propDFpath = cs["propDFpath"]
    prop = cs["prop"]
    TARGET = cs["TARGET"]
    data_prefix = cs["dataPrefix"]

    # Read dataset
    crysdf = pd.read_pickle(propDFpath)
    crysdf[prop] = crysdf[prop].astype('int16')

    # Set up directories for data storage
    data_dest = "data/clean_data/alignn_format"
    os.makedirs(data_dest, exist_ok=True)

    # Initialize output files and directories
    id_file_path = os.path.join(data_dest, f"{data_prefix}{prop}_id_from_{TARGET}.csv")
    data_files_dir = os.path.join(data_dest, f"{data_prefix}atomistic_{prop}_{experiment}")
    os.makedirs(data_files_dir, exist_ok=True)

    # Write dataset in desired format
    with open(id_file_path, "w") as id_file:
        for _, row in crysdf.iterrows():
            poscar_name = f"POSCAR-{row['material_id']}.vasp"
            target_value = row[TARGET]
            formatted_target = f"{target_value:.6f}" if pd.notna(target_value) else "NaN"

            # Save atomic structure to POSCAR file
            jarvis_atom = ase_to_jarvis(row["atoms"])
            jarvis_atom.write_poscar(os.path.join(data_files_dir, poscar_name))

            # Write the mapping of POSCAR files to target values
            id_file.write(f"{poscar_name},{formatted_target}\n")

    return data_files_dir
