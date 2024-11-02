
import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(42)

# Load the main dataset containing synthesizability data
synthDF = pd.read_pickle('synthDF')

# Create a copy of the dataset for stability experiments
stabilityDF = synthDF.copy()

# Separate the dataset into experimental (synthesizable) and theoretical (unsynthesizable) entries
experimental_df = synthDF[synthDF.synth == 1]
theoretical_df = synthDF[synthDF.synth == 0]

# Initialize a new column "stability" in stabilityDF and set it to NaN initially
stabilityDF.loc[:, "stability"] = np.nan

# Define stability based on "energy_above_hull": stable if ≤ 0.015, otherwise unstable
stabilityDF["stability"] = np.where(stabilityDF.energy_above_hull <= 0.015, 1, 0)

# Copy the stability column to a ground-truth stability column "stability_GT"
stabilityDF["stability_GT"] = stabilityDF["stability"].copy()

# Calculate the number of positive stability samples to unlabel
n_unlabel = stabilityDF["stability_GT"].sum() - len(experimental_df)

# Randomly select rows from the stable materials to unlabel to match the synthesizability class count
materials_to_unlabel = stabilityDF[stabilityDF["stability_GT"] == 1].sample(n=int(n_unlabel)).index
stabilityDF.loc[materials_to_unlabel, "stability"] = 0  # Unlabel by setting stability to 0

# Ensure "stability" and "stability_GT" are integers for clarity
stabilityDF["stability"] = stabilityDF["stability"].astype(int)
stabilityDF["stability_GT"] = stabilityDF["stability_GT"].astype(int)

# Shuffle the dataset to mix synthesizability values (just in case)
stabilityDF = stabilityDF.sample(frac=1).reset_index(drop=True)

# Sort by stability, with stable entries at the top, and reset the index
stabilityDF = stabilityDF.sort_values(by="stability", ascending=False).reset_index(drop=True)

# Drop the original "synth" column since it is no longer needed
stabilityDF = stabilityDF.drop(columns="synth")

# Save the dataset
stabilityDF.to_pickle('stabilityDF015')
