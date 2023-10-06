def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError(f'Boolean value expected for --small_data and --ehull. Insead we got {value} with type{type(value)}.')


def current_setup(ehull_test, small_data, experiment):
# def current_setup(ehull_test, small_data, experiment, schnettest):
    if str_to_bool(ehull_test) and str_to_bool(small_data):
        error_message = "small_data and ehull_test are not allowed at the same time."
        raise Exception(error_message)
    # if small_data and schnettest:
    #     propDFpath = 'data/clean_data/small_synthDFTest'
    #     result_dir = 'data/results/small_stabilityTest'
    #     prop = 'synth'
    elif small_data:
        propDFpath = 'data/clean_data/small_synthDF'
        result_dir = 'data/results/small_data_synth'
        prop = 'synth'
    # elif schnettest:
    #     propDFpath = 'data/clean_data/stabilityTest' 
    #     result_dir = 'data/results/stabilityTest'
    #     prop = 'stability'
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
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
            'coAlSch3':'coSchAl2',
            'coAlSch4':'coSchAl3',
            'coAlSch5':'coSchAl4',
            'schnet0':prop, 
            'coSchAl1':'alignn0',
            'coSchAl2':'coAlSch1',
            'coSchAl3':'coAlSch2',
            'coSchAl4':'coAlSch3',
            'coSchAl5':'coAlSch4',
            'final_class':'change_to_desired_label',
    }
    data_prefix = "small_" if small_data else ""
    
    
    return {"propDFpath":propDFpath, "result_dir":result_dir, "prop":prop, 
            "TARGET":experiment_target_match[experiment], "dataPrefix":data_prefix}