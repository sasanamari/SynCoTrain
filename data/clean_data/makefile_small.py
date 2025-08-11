import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(42)

# Fraction of data to sample for the smaller dataset
small_data_frac = 0.05

# Load the full dataset of crystal synthesizability information
synthDF = pd.read_pickle("synthDF")

# Separate experimental and theoretical entries
experimental_df = synthDF[synthDF.synth == 1]
theoretical_df = synthDF[synthDF.synth == 0]

# Sample a small fraction of each category for quick experiments
small_experimental_df = experimental_df.sample(
    frac=small_data_frac, random_state=42, ignore_index=True
)
small_theoretical_df = theoretical_df.sample(
    frac=small_data_frac, random_state=42, ignore_index=True
)

# Combine sampled experimental and theoretical data into a single DataFrame
small_df = pd.concat([small_experimental_df, small_theoretical_df], ignore_index=True)

# Save the smaller dataset for quick experimentation
small_df.to_pickle("small_synthDF")
