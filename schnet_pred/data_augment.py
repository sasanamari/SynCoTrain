import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.mixture import GaussianMixture
import pickle as pkl
from pathlib import Path
from separate_test import split_and_save_test, split_and_save_symmetrical_test

# Set random seed for reproducibility
np.random.seed(394)
noise_frac = 0.05  # 0.10
balanced = False
symmetrical_test = True
source = "synth_labels_2_threshold_75.pkl"
test_filename = "test_df_aug_75_symmetrical.pkl"
dest_filename = "augmented_data_75_symmetrical.pkl"

print(
    f"Source: {source}, Test filename: {test_filename}, Destination filename: {dest_filename}"
)


# Function to convert ASE Atoms to Pymatgen Structure
def ase_to_pymatgen(ase_atoms):
    return AseAtomsAdaptor.get_structure(ase_atoms)


# Function to convert Pymatgen Structure to ASE Atoms
def pymatgen_to_ase(pmg_structure):
    return AseAtomsAdaptor.get_atoms(pmg_structure)


# Function to fit the Gaussian Mixture Model
def get_gmm():
    with open(Path("schnet_pred") / "dists.pkl", "rb") as file:
        dist = pkl.load(file)
    return GaussianMixture(n_components=20, random_state=19).fit(
        np.array(dist).reshape(-1, 1)
    )


# Function to perturb the structure
def perturb(struct, data):
    def get_rand_vec(dist):
        vector = np.random.randn(3)
        vnorm = np.linalg.norm(vector)
        return vector / vnorm * dist if vnorm != 0 else get_rand_vec(dist)

    struct_per = struct.copy()
    for i in range(len(struct_per)):
        dist = np.random.choice(data.ravel())
        struct_per.translate_sites([i], get_rand_vec(dist), frac_coords=False)
    return struct_per


# Function to add balanced noise to the target values
def format_and_add_balanced_noise(df, noise_frac=0.05):
    df["targets"] = df["synth_labels"].copy()

    # Separate the data by class
    class_0 = df[df["synth_labels"] == 0].copy()
    class_1 = df[df["synth_labels"] == 1].copy()

    # Calculate the number of swaps for each class
    n_swap = int(min(len(class_0), len(class_1)) * noise_frac)

    # Add noise to class 0
    swap_indices_0 = np.random.choice(class_0.index, n_swap, replace=False)
    df.loc[swap_indices_0, "targets"] = 1

    # Add noise to class 1
    swap_indices_1 = np.random.choice(class_1.index, n_swap, replace=False)
    df.loc[swap_indices_1, "targets"] = 0

    # Format the targets
    df["targets"] = df["targets"].map(lambda target: np.array(target).flatten())
    df["targets"] = df["targets"].map(lambda target: {"synth": np.array(target)})

    return df


# Main execution
if __name__ == "__main__":
    # Directory setup
    base_dir = Path(__file__).parent.resolve()
    source_dir = Path("data/results/synth")

    data_dir = base_dir / "data"
    model_dir = base_dir / "models"
    logs_dir = base_dir / "logs"

    # Load your data
    propDFpath = source_dir / source
    # df = pd.read_pickle(propDFpath)
    print("Reading data...")
    if symmetrical_test:
        train_val_df = split_and_save_symmetrical_test(
            propDFpath=propDFpath, test_save_path=data_dir / test_filename
        )
    else:
        train_val_df = split_and_save_test(
            propDFpath=propDFpath, test_save_path=data_dir / test_filename
        )
    # Add the 'aug' column to mark original data
    train_val_df["aug"] = 0

    # Fit the Gaussian Mixture Model
    print("Fitting the Gaussian Mixture Model...")
    gm = get_gmm()
    data = gm.sample(10**8)[0]

    # Convert ASE Atoms to Pymatgen Structures
    print("Converting ASE Atoms to Pymatgen Structures...")
    train_val_df["pmg_structure"] = train_val_df["atoms"].apply(ase_to_pymatgen)

    # Add balanced noise to the DataFrame
    print("Adding balanced noise to the data...")
    df_noisy = format_and_add_balanced_noise(train_val_df, noise_frac=noise_frac)

    # Perturb the structures and store the augmented data
    print("Generating augmented data...")
    augmented_data = []
    for k, (idx, row) in enumerate(df_noisy.iterrows()):
        original_structure = row["pmg_structure"]
        perturbed_structure = perturb(original_structure, data)
        perturbed_atoms = pymatgen_to_ase(perturbed_structure)

        # Create a new entry for the perturbed structure
        new_entry = row.copy()
        new_entry["atoms"] = perturbed_atoms
        new_entry["aug"] = 1  # Mark as augmented
        new_entry["material_id"] = f"p_{row['material_id']}"  # Modify material_id

        augmented_data.append(new_entry)
        if k % 2000 == 0:
            print(f"Processed {k} crystals from {len(df_noisy)}.")
    # Convert the list of augmented data into a DataFrame
    augmented_df = pd.DataFrame(augmented_data)

    # Combine the original and augmented data
    final_df = pd.concat([df_noisy, augmented_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop the intermediate pmg_structure column
    final_df = final_df.drop(columns=["pmg_structure"])

    # Save the combined DataFrame if needed
    final_df.to_pickle(Path(data_dir) / dest_filename)
