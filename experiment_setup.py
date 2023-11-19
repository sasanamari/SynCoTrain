def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError(f'Boolean value expected for --small_data and --ehull. Insead we got {value} with type{type(value)}.')

def str_to_int(value):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f'Integer value expected, but received {value} with type {type(value)}.')



def current_setup(ehull_test, small_data, experiment, ehull015):
# def current_setup(ehull_test, small_data, experiment, schnettest):
    if str_to_bool(ehull_test) and str_to_bool(small_data):
        error_message = "small_data and ehull_test are not allowed at the same time."
        raise Exception(error_message)
    if str_to_bool(ehull015) and str_to_bool(small_data):
        error_message = "small_data and ehull015 are not allowed at the same time."
        raise Exception(error_message)

    elif small_data:
        propDFpath = 'data/clean_data/small_synthDF'
        result_dir = 'data/results/small_data_synth'
        prop = 'synth'
    elif ehull015:
        propDFpath = 'data/clean_data/stabilityDF015' 
        result_dir = 'data/results/stability015'
        prop = 'stability'
    elif ehull_test:
        propDFpath = 'data/clean_data/stabilityDF' 
        result_dir = 'data/results/stability'
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
    
    
    return {"propDFpath":propDFpath, "result_dir":result_dir, "prop":prop, 
            "TARGET":experiment_target_match[experiment], "dataPrefix":data_prefix}