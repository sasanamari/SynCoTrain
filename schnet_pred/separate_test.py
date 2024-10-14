import pandas as pd
import numpy as np
from pathlib import Path

def split_and_save_test(propDFpath: str, test_save_path: Path, quick_debug: bool = False) -> pd.DataFrame:
    """
    Load data from propDFpath, split into training/validation and test sets,
    save the test set to the specified path and return the training/validation DataFrame.

    Parameters:
    - propDFpath (str): File path to the dataframe to load.
    - test_save_path (Path): File path to save the test dataframe.
    - quick_debug (bool): If true, use a smaller fraction of the data for quick debugging.

    Returns:
    - train_val_df (pd.DataFrame): The training and validation dataframe.
    """

    # Load the data
    crysdf = pd.read_pickle(propDFpath)
    crysdf = crysdf[['material_id', 'synth', 'synth_labels', 'atoms']]

    # Debugging option
    if quick_debug:
        crysdf = crysdf.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Split data into training/validation and test sets
    print('Separating test from training/validation data...')
    train_val_frac = 0.9
    train_length = int(len(crysdf) * train_val_frac)
    train_val_df = crysdf.sample(frac=train_val_frac, random_state=42)
    test_df = crysdf.drop(train_val_df.index).reset_index(drop=True)
    train_val_df = train_val_df.reset_index(drop=True)

    # Save the test set before applying noise or augmentation
    test_df.to_pickle(test_save_path)

    return train_val_df



def split_and_save_symmetrical_test(propDFpath: str, test_save_path: Path, quick_debug: bool = False) -> pd.DataFrame:
    """
    Load data from propDFpath, split into training/validation and test sets,
    save the test set to the specified path and return the training/validation DataFrame.

    Parameters:
    - propDFpath (str): File path to the dataframe to load.
    - test_save_path (Path): File path to save the test dataframe.
    - quick_debug (bool): If true, use a smaller fraction of the data for quick debugging.

    Returns:
    - train_val_df (pd.DataFrame): The training and validation dataframe.
    """

    # Load the data
    crysdf = pd.read_pickle(propDFpath)
    crysdf = crysdf[['material_id', 'synth', 'synth_labels', 'atoms']]

    # Debugging option
    if quick_debug:
        crysdf = crysdf.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Separate the test set with an equal number of data points from each class
    print('Separating test from training/validation data...')
    test_frac = 0.1
    n_test_per_class = int(len(crysdf) * test_frac / 2)
    
    # Split data into test and train/validation sets
    test_df_list = []
    train_val_df_list = []

    for label in crysdf['synth_labels'].unique():
        class_subset = crysdf[crysdf['synth_labels'] == label]
        test_class_samples = class_subset.sample(n=n_test_per_class, random_state=42)
        train_val_class_samples = class_subset.drop(test_class_samples.index)

        test_df_list.append(test_class_samples)
        train_val_df_list.append(train_val_class_samples)

    # Concatenate test set and train/validation set
    test_df = pd.concat(test_df_list)
    train_val_df = pd.concat(train_val_df_list)
    
    test_df = test_df.sample(frac=1, random_state=43).reset_index(drop=True)
    train_val_df = train_val_df.sample(frac=1, random_state=44).reset_index(drop=True)

    # Save the test set before applying noise or augmentation
    test_df.to_pickle(test_save_path)

    return train_val_df
