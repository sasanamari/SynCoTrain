import pandas as pd
import numpy as np

def format_and_add_noise(df, noise_frac=0.05):
    """
    Adds noise to the target values and formats them for model training.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    noise_frac (float): The fraction of the targets to apply noise to.
    
    Returns:
    pd.DataFrame: The dataframe with noise added (if specified) and targets formatted.
    """
    df["targets"] = df["synth_labels"].copy()
    # Add noise if noise_frac > 0
    if noise_frac > 0:
        targets = df["synth_labels"].copy()
        n_swap = int(len(targets) * noise_frac)
        np.random.seed(42)
        swap_indices = np.random.choice(len(targets), n_swap, replace=False)
        for idx in swap_indices:
            targets.iloc[idx] = 1 - targets.iloc[idx]
        df["targets"] = targets

    # Format the targets
    df["targets"] = df["targets"].map(lambda target: np.array(target).flatten())
    df["targets"] = df["targets"].map(lambda target: {"synth": np.array(target)})
    
    return df


