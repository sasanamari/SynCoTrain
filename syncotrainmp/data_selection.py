#! /usr/bin/env python
"""
This module prepares data for semi-supervised machine learning in synthesizability prediction.
It sets up data paths, processes and filters input data, and generates
train/test splits for a series of experiments.

Functions:
- parse_arguments: Parses command-line arguments for experiment configuration.
- setup_experiment: Retrieves setup configurations for the specified experiment.
- load_and_prepare_data: Loads and prepares the data by removing duplicates and handling NaN values.
- setup_output_directory: Creates an output directory for saving experiment results.
- prepare_experiment_data: Prepares data for specific experiments like "alignn" if required.
- leaveout_test_split: Creates a "leaveout" test set and updates the positive data accordingly.
- save_ids: Saves a DataFrame's index as a text file.
- train_test_split: Generates train/test splits with balanced classes for semi-supervised learning.
- save_splits: Saves train/test splits to specified output files.

Usage:
    Run this script from the command line with experiment-specific arguments.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from syncotrainmp.experiment_setup import current_setup, str_to_bool

# Constants [TODO: convert to options]
DATA_DIR = 'data/clean_data/'
TEST_PORTION = 0.1
LEAVEOUT_TEST_PORTION = TEST_PORTION * 0.5

def parse_arguments():
    """
    Parses command-line arguments for configuring experiment parameters.

    Returns:
        argparse.Namespace: Contains parsed arguments for experiment, ehull015 cutoff, and small data usage.
    """
    parser = argparse.ArgumentParser(description="Semi-Supervised ML for Synthesizability Prediction")
    parser.add_argument("--experiment", default="alignn0", help="Experiment name and corresponding config files.")
    parser.add_argument("--ehull015", type=str_to_bool, default=False, help="Evaluate PU Learning efficacy with 0.015 eV cutoff.")
    parser.add_argument("--small_data", type=str_to_bool, default=False, help="Use a small dataset for quick workflow checks.")
    return parser.parse_args(sys.argv[1:])

def setup_experiment(args):
    """
    Sets up experiment configurations by loading paths and properties based on the specified arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: Contains property DataFrame path, result directory, property name, target column, and data prefix.
    """
    cs = current_setup(small_data=args.small_data, experiment=args.experiment, ehull015=args.ehull015)
    return cs["propDFpath"], cs["result_dir"], cs["prop"], cs["TARGET"], cs["dataPrefix"]

def load_and_prepare_data(data_path, prop, TARGET):
    """
    Loads data, selects required columns, drops duplicates, and handles NaN values in target column.

    Args:
        data_path (str): Path to the property DataFrame file.
        prop (str): Name of the property column to retain.
        TARGET (str): Name of the target column.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for train/test splitting.
    """
    print(f"Reading data from {data_path}.")
    df = pd.read_pickle(data_path)
    df = df[['material_id', prop, TARGET]]
    df = df.loc[:, ~df.columns.duplicated()] # Drops duplicated props at round zero
    df = df[~df[TARGET].isna()]              # Remving NaN values. for small_data
    return df

def setup_output_directory(data_path, data_prefix, TARGET, prop):
    """
    Creates an output directory for storing train/test split files based on experiment configuration.

    Args:
        data_path (str): Path to the property DataFrame file.
        data_prefix (str): Prefix for data files.
        TARGET (str): Target column name.
        prop (str): Property column name.

    Returns:
        str: Path to the created output directory.
    """
    split_id_dir = f"{data_prefix}{TARGET}_{prop}"
    split_id_dir_path = os.path.join(os.path.dirname(data_path), split_id_dir)
    os.makedirs(split_id_dir_path, exist_ok=True)
    return split_id_dir_path

def prepare_experiment_data(experiment, small_data, ehull015):
    """
    Prepares data specifically for "alignn" or "coAl" experiments.

    Args:
        experiment (str): Name of the experiment.
        small_data (bool): Whether to use a small subset of the data.
        ehull015 (bool): Whether to use the 0.015 eV cutoff for hull calculations.
    """
    if experiment in {"alignn0", "coAl"}:
        from syncotrainmp.pu_alignn.preparing_data_byFile import prepare_alignn_data
        alignn_data_log = prepare_alignn_data(small_data=small_data, experiment=experiment, ehull015=ehull015)
        print(alignn_data_log)

def leaveout_test_split(df, prop, TARGET):
    """
    Creates a leave-out test set from experimental data and updates the positive data pool.

    Args:
        df (pd.DataFrame): Input DataFrame containing material data.
        prop (str): Column name for the property.
        TARGET (str): Column name for the target variable.

    Returns:
        tuple: Contains experimental, positive, and leaveout DataFrames.
    """
    experimental_df = df[df[prop] == 1]
    positive_df = df[df[TARGET] == 1]
    leaveout_df = experimental_df.sample(frac=LEAVEOUT_TEST_PORTION, random_state=4242)
    positive_df = positive_df.drop(index=leaveout_df.index)
    return experimental_df, positive_df, leaveout_df

def save_ids(data, filename):
    """
    Saves DataFrame indices to a specified file, typically for storing split IDs.

    Args:
        data (pd.DataFrame or pd.Series): Data whose indices need to be saved.
        filename (str): Output file path for saved IDs.
    """
    data.index.to_series().to_csv(filename, index=False, header=False)

def train_test_split(df, positive_df, leaveout_df, prop, TARGET, num_iter, test_ratio):
    """
    Generates train/test splits for a specified number of iterations with balanced classes.

    Args:
        df (pd.DataFrame): Main DataFrame with all data.
        positive_df (pd.DataFrame): Positive class subset of df.
        leaveout_df (pd.DataFrame): Leave-out test set.
        prop (str): Property column name.
        TARGET (str): Target column name.
        num_iter (int): Number of train/test iterations to generate.
        test_ratio (float): Ratio of test data.

    Returns:
        list: Contains tuples of training and testing DataFrames for each iteration.
    """
    splits = []
    for it in range(num_iter):
        testdf1 = positive_df.sample(frac=test_ratio, random_state=it)
        testdf1 = pd.concat([leaveout_df, testdf1])
        df_wo_test = df.drop(index=testdf1.index)
        traindf1 = df_wo_test[df_wo_test[TARGET] == 1].sample(frac=1, random_state=it+1)
        class_train_num = len(traindf1)
        unlabeled_df = df_wo_test[df_wo_test[TARGET] == 0]

        unlabeled_shortage = class_train_num - len(unlabeled_df)
        if unlabeled_shortage > 0:
            testdf0 = unlabeled_df.sample(n=int(test_ratio * max(len(unlabeled_df), len(positive_df))), random_state=it+4)
            unlabeled_df = unlabeled_df.drop(index=testdf0.index)
            traindf0 = pd.concat([unlabeled_df.sample(frac=1, random_state=it+2),
                                  unlabeled_df.sample(n=unlabeled_shortage, replace=True, random_state=it+3)])
        else:
            traindf0 = unlabeled_df.sample(n=class_train_num, random_state=it+2)
            testdf0 = unlabeled_df.drop(index=traindf0.index)

        splits.append((pd.concat([traindf0, traindf1]).sample(frac=1, random_state=it+3),
                       pd.concat([testdf0, testdf1]).sample(frac=1, random_state=it+4)))
    return splits

def save_splits(splits, output_dir):
    """
    Saves generated train/test splits for each iteration to the output directory.

    Args:
        splits (list): List of tuples containing train/test DataFrames for each iteration.
        output_dir (str): Directory to save the split files.
    """
    for it, (train_df, test_df) in enumerate(splits):
        save_ids(train_df, os.path.join(output_dir, f"train_id_{it}.txt"))
        save_ids(test_df, os.path.join(output_dir, f"test_id_{it}.txt"))

def main():
    """
    Main execution function to set up, process, and save experiment data splits.
    """
    args = parse_arguments()
    propDFpath, result_dir, prop, TARGET, data_prefix = setup_experiment(args)
    df = load_and_prepare_data(propDFpath, prop, TARGET)
    output_dir = setup_output_directory(propDFpath, data_prefix, TARGET, prop)

    prepare_experiment_data(args.experiment, args.small_data, args.ehull015)

    experimental_df, positive_df, leaveout_df = leaveout_test_split(df, prop, TARGET)
    save_ids(leaveout_df, os.path.join(output_dir, "leaveout_test_id.txt"))
    # Save the experimental data size as a scalar value
    with open(os.path.join(output_dir, "experimentalDataSize.txt"), "w") as f:
        f.write(str(experimental_df[prop].sum()))

    splits = train_test_split(df, positive_df, leaveout_df, prop, TARGET, num_iter=100, test_ratio=TEST_PORTION)
    save_splits(splits, output_dir)

    print(f"Train/Test splits for {args.experiment} experiment saved in {output_dir}.")

if __name__ == "__main__":
    main()
