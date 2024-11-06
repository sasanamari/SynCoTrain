def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError(f'Boolean value expected for --small_data and --ehull015. Insead we got {value} with type{type(value)}.')

def str_to_int(value):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f'Integer value expected, but received {value} with type {type(value)}.')



def current_setup(small_data, experiment, ehull015):
    """
    Set up the current experiment configuration based on input parameters.

    Args:
        small_data (bool): Indicates whether to use a small subset of data.
        experiment (str): The name of the experiment.
        ehull015 (bool): Indicates whether to use a specific stability cutoff.

    Returns:
        dict: A dictionary containing the configuration setup, including:
            - propDFpath: Path to the property DataFrame.
            - result_dir: Directory for saving results.
            - prop: The property name being analyzed.
            - TARGET: The training label column based on the experiment.
            - dataPrefix: A prefix for the dataset based on input parameters.

    Raises:
        Exception: If both small_data and ehull015 are set to True.
        KeyError: If the experiment is not recognized in the mapping.
    """
    if str_to_bool(ehull015) and str_to_bool(small_data):
        raise Exception("small_data and ehull015 cannot be set to True at the same time.")

    elif small_data:
        propDFpath = 'data/clean_data/small_synthDF'
        result_dir = 'data/results/small_data_synth'
        prop = 'synth'
    elif ehull015:
        propDFpath = 'data/clean_data/stabilityDF015' 
        result_dir = 'data/results/stability015'
        prop = 'stability'
    else:
        propDFpath = 'data/clean_data/synthDF'
        result_dir = 'data/results/synth'
        prop = 'synth'

    experiment_target_match = { #output_dir: training_label_column
            'alignn0':prop, 
            'coAlignn1':'schnet0',
            'coAlignn2':'coSchnet1',
            'coAlignn3':'coSchnet2',
            'coAlignn4':'coSchnet3',
            'coAlignn5':'coSchnet4',
            'schnet0':prop, 
            'coSchnet1':'alignn0',
            'coSchnet2':'coAlignn1',
            'coSchnet3':'coAlignn2',
            'coSchnet4':'coAlignn3',
            'coSchnet5':'coAlignn4',
            'final_avg':'final_label',
    }
    data_prefix = "small_" if small_data else ""
    if ehull015:
        data_prefix = "15_"
    
    # Ensure the experiment is valid and retrieve the target
    if experiment not in experiment_target_match:
        raise KeyError(f"Unrecognized experiment: {experiment}")

    return {"propDFpath":propDFpath, "result_dir":result_dir, "prop":prop, 
            "TARGET":experiment_target_match[experiment], "dataPrefix":data_prefix}
