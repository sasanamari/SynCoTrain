# %%
import os
import sys
import time
import argparse
from importlib.resources import files
from jarvis.db.jsonutils import loadjson, dumpjson

from syncotrainmp.experiment_setup import current_setup
from syncotrainmp.pu_alignn.alignn_setup import train_for_folder
from syncotrainmp.pu_alignn.alignn_configs.alignn_pu_config import (
    alignn_pu_config_generator,
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction -- ALIGNN PU Step"
    )
    parser.add_argument(
        "--experiment",
        default="alignn0",
        help="Name of the experiment and corresponding config files.",
    )
    parser.add_argument(
        "--small_data",
        action="store_true",
        default=False,
        help="Select a small subset of data for quicker workflow checks.",
    )
    parser.add_argument(
        "--ehull015",
        action="store_true",
        default=False,
        help="Predict stability to evaluate PU Learning's efficacy with 0.015 eV cutoff.",
    )
    parser.add_argument(
        "--startIt", type=int, default=0, help="Starting iteration number."
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID to use for training."
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Name of the experiment and corresponding config files.",
    )

    # Parse the arguments
    return parser.parse_args(sys.argv[1:])


def config_generator(
    newConfigName,
    data_prefix,
    iterNum,
    epochNum,
    output_dir,
    ehull015=False,
    experiment=None,
):
    """
    Generates a configuration file for the training process based on specified parameters.

    Args:
        newConfigName (str): The name of the new configuration file to be created.
        iterNum (int): The current iteration number.
        epochNum (int): The number of training epochs.
        alignn_dir (str): Directory for ALIGNN-related files.
        ehull015 (bool): Flag indicating if the 0.015 eV cutoff is used.
        experiment (str): Name of the current experiment.
    """
    config_path = files("syncotrainmp.pu_alignn.alignn_configs").joinpath(
        "default_class_config.json"
    )

    _config = loadjson(config_path)
    _config["random_seed"] = iterNum
    _config["epochs"] = epochNum

    # Set output directory based on the cutoff flag
    output_prefix = "PUehull015" if ehull015 else "PUOutput"
    output_dir = os.path.join(
        output_dir, "pu_alignn", f"{output_prefix}_{data_prefix}{experiment}"
    )
    _config["output_dir"] = os.path.join(output_dir, f"{iterNum}iter/")

    dumpjson(_config, filename=newConfigName)
    print(f"Config file for iteration {iterNum} was generated.")


def run_training_iterations(pu_setup, args, cs, split_id_path, start_time, output_dir):
    """
    Runs the training iterations for the specified configuration.

    Args:
        pu_setup (dict): Configuration settings for the PU training.
        args (argparse.Namespace): Parsed command line arguments.
        split_id_path (str): Path to the split ID files for training and testing.
        start_time (float): The start time of the training process for logging.
    """
    prop = cs["prop"]
    data_prefix = cs["dataPrefix"]

    for iterNum in range(
        pu_setup["start_of_iterations"], pu_setup["max_num_of_iterations"]
    ):
        config_generator(
            pu_setup["class_config_name"],
            cs["dataPrefix"],
            iterNum,
            pu_setup["epochs"],
            output_dir,
            experiment=args.experiment,
        )

        train_for_folder(
            gpu_id=args.gpu_id,
            root_dir=pu_setup["root_dir"],
            config_name=pu_setup["class_config_name"],
            keep_data_order=pu_setup["keep_data_order"],
            classification_threshold=pu_setup["classification_threshold"],
            output_dir=pu_setup["output_dir"],
            batch_size=None,  # Read separately for each iteration from the generated config file
            epochs=pu_setup["epochs"],
            file_format=pu_setup["file_format"],
            ehull015=args.ehull015,
            small_data=args.small_data,
            train_id_path=os.path.join(split_id_path, f"train_id_{iterNum}.txt"),
            test_id_path=os.path.join(split_id_path, f"test_id_{iterNum}.txt"),
            experiment=args.experiment,
        )

        # Calculate and log remaining time estimates
        elapsed_time = time.time() - start_time
        remaining_iterations = pu_setup["max_num_of_iterations"] - iterNum - 1
        time_per_iteration = elapsed_time / (
            iterNum - pu_setup["start_of_iterations"] + 1
        )
        estimated_remaining_time = remaining_iterations * time_per_iteration
        remaining_days = int(estimated_remaining_time // (24 * 3600))
        remaining_hours = int((estimated_remaining_time % (24 * 3600)) // 3600)

        if not os.path.exists(os.path.join(output_dir, "time_logs")):
            os.makedirs(os.path.join(output_dir, "time_logs"))

        time_log_path = os.path.join(
            output_dir,
            "time_logs",
            f"alignn_remaining_time_{data_prefix}{args.experiment}_{prop}.txt",
        )
        with open(time_log_path, "w") as file:
            file.write(
                f"Iterations completed: {iterNum - pu_setup['start_of_iterations']}\n"
            )
            file.write(f"Iterations remaining: {remaining_iterations}\n")
            file.write(
                f"Estimated remaining time: {remaining_days} days, {remaining_hours} hours\n"
            )

        print(
            f"Iteration {iterNum} completed. Remaining time: {remaining_days} days, {remaining_hours} hours"
        )

    # Final summary of elapsed time
    elapsed_days = int(elapsed_time // (24 * 3600))
    elapsed_hours = int((elapsed_time % (24 * 3600)) // 3600)

    with open(time_log_path, "w") as file:
        file.write(
            f"Iterations completed: {pu_setup['max_num_of_iterations'] - pu_setup['start_of_iterations']}\n"
        )
        file.write(f"Total time taken: {elapsed_days} days, {elapsed_hours} hours\n")

    print(
        f"PU Learning completed. Total time taken: {elapsed_days} days, {elapsed_hours} hours"
    )


def main():
    """
    Main function to execute the semi-supervised machine learning process for
    synthesizability prediction.
    """
    # Parse the arguments
    args = parse_arguments()

    # Set up the current experiment configuration
    cs = current_setup(
        small_data=args.small_data, experiment=args.experiment, ehull015=args.ehull015
    )
    propDFpath = cs["propDFpath"]
    prop = cs["prop"]
    TARGET = cs["TARGET"]
    data_prefix = cs["dataPrefix"]

    # Define paths for data management
    data_dir = os.path.dirname(propDFpath)
    split_id_dir = f"{data_prefix}{TARGET}_{prop}"
    split_id_path = os.path.join(data_dir, split_id_dir)

    # Load PU configuration
    pu_config_name = alignn_pu_config_generator(
        args.experiment, cs, args.small_data, args.output_dir
    )
    pu_setup = loadjson(pu_config_name)
    pu_setup["start_of_iterations"] = args.startIt

    print(
        f"Now we run calculations for iterations {pu_setup['start_of_iterations']} till {pu_setup['max_num_of_iterations']}"
    )
    start_time = time.time()

    # Run the training iterations
    run_training_iterations(
        pu_setup, args, cs, split_id_path, start_time, args.output_dir
    )


if __name__ == "__main__":
    main()
