import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import argparse

from pymatgen.core import Structure
from syncotrainmp.utility.crystal_funcs import clean_oxide
from syncotrainmp.utility.crystal_funcs import exper_oxygen_query
from syncotrainmp.utility.crystal_structure_conversion import pymatgen_to_ase


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Downloading experimental data."
    )
    parser.add_argument(
        "--MPID",
        default="",
        help="This is your Materials Project ID.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="If specified, creates a smaller sample for testing purposes."
    )
    args = parser.parse_args(sys.argv[1:])

    return args


def query_data(MPID, num_sites):
    """Query the Materials Project database for experimental and theoretical data."""
    pymatgen_exp_array, db_version = exper_oxygen_query(
        MPID = MPID,
        theoretical_data = False,
        num_sites = num_sites,
        fields = "default",
    )
    pymatgen_theo_array, db_version = exper_oxygen_query(
        MPID=MPID,
        theoretical_data = True,
        num_sites = num_sites,
        fields = "default",
    )

    print(db_version)
    print(f"We retrieved {len(pymatgen_exp_array)} experimental crystals from the Materials Project database.")
    print(f"Retrieved {len(pymatgen_theo_array)} theoretical crystals from the Materials Project database.")

    return pymatgen_exp_array, pymatgen_theo_array


def convert_structures(pymatgen_array):
    """Convert structure dictionaries to Structure objects."""
    for material in pymatgen_array:
        material["structure"] = Structure.from_dict(material["structure"])
    return pymatgen_array


def clean_data(pymatgen_exp_array, pymatgen_theo_array):
    """Clean the experimental and theoretical data arrays."""

    # Also removes "experimental" crystals with e_above_hull > 1 eV
    good_experimental_data = clean_oxide(
        experimental = True,
        pymatgenArray = pymatgen_exp_array,
        reportBadData = False
    )
    good_theoretical_data = clean_oxide(
        experimental = False,
        pymatgenArray = pymatgen_theo_array,
        reportBadData = False
    )
    print(f"We have {len(good_experimental_data)} experimental oxides after cleaning.")
    print(f"We have {len(good_theoretical_data)} theoretical oxides after cleaning.")

    return good_experimental_data, good_theoretical_data


def filter_keys(data_array, keys_to_keep):
    """Keep only the specified keys in each entry of the data array."""
    current_keys = set(data_array[0].keys())
    keys_to_remove = current_keys.difference(set(keys_to_keep))

    for material in data_array:
        material["atoms"] = pymatgen_to_ase(material["structure"])
        for key in keys_to_remove:
            material.pop(key, None)
    return data_array


def label_data(data_array, label):
    """Label the data as experimental or theoretical."""
    for material in data_array:
        material["synth"] = label
    return data_array


def create_dataframe(exp_data, theo_data):
    """Create a combined DataFrame from experimental and theoretical data arrays."""
    exp_df = pd.DataFrame.from_records(exp_data)
    theo_df = pd.DataFrame.from_records(theo_data)

    synth_df = pd.concat([exp_df, theo_df])
    synth_df = synth_df.sample(frac=1, ignore_index=True)  # Shuffle data
    synth_df["material_id"] = synth_df["material_id"].astype(str)  # Ensure all IDs are strings

    # Add columns for experiments with NaN values
    experiment_columns = ['schnet0', 'alignn0', 'coSchnet1', 'coAlignn1', 'coSchnet2', 'coAlignn2', 'coSchnet3', 'coAlignn3']
    synth_df[experiment_columns] = np.nan

    return synth_df


def save_dataframe(synth_df, filename):
    """Save the DataFrame to a pickle file."""
    synth_df.to_pickle(filename)
    print(f"Dataframe saved to {filename}.")


def main():
    # Parse arguments and set up directories
    args = parse_arguments()

    # Query data
    if args.small:
        pymatgen_exp_array, pymatgen_theo_array = query_data(args.MPID, (1, 150))
    else:
        pymatgen_exp_array, pymatgen_theo_array = query_data(args.MPID, (2, 3))

    # Convert structures from dictionaries to Structure objects
    pymatgen_exp_array = convert_structures(pymatgen_exp_array)
    pymatgen_theo_array = convert_structures(pymatgen_theo_array)

    # Clean the data to remove unstable entries
    good_exp_data, good_theo_data = clean_data(pymatgen_exp_array, pymatgen_theo_array)

    # Filter keys and label data
    keys_to_keep = ["material_id", "atoms", "energy_above_hull", "formation_energy_per_atom"]
    good_exp_data = filter_keys(good_exp_data, keys_to_keep)
    good_theo_data = filter_keys(good_theo_data, keys_to_keep)
    good_exp_data = label_data(good_exp_data, label=1)  # Label as experimental
    good_theo_data = label_data(good_theo_data, label=0)  # Label as theoretical

    # Create and save the DataFrame
    synth_df = create_dataframe(good_exp_data, good_theo_data)

    if args.small:
        save_dataframe(synth_df, 'miniTestSynthdf.pkl')
    else:
        save_dataframe(synth_df, 'synthDF')


if __name__ == "__main__":
    main()
