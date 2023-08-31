def current_setup(ehull_test, small_data, experiment):
    if small_data and ehull_test:
        error_message = "small_data and ehull_test are not allowed at the same time."
        raise Exception(error_message)
    if small_data:
        propDFpath = '/data/clean_data/small_synthDF'
        result_dir = 'data/results/small_data_synth'
        prop = 'synth'
    elif ehull_test:
        propDFpath = '/data/clean_data/stabilityDF' 
        result_dir = 'data/results/stability'
        prop = 'stability'
    else:
        propDFpath = '/data/clean_data/synthDF'
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
            'final_class':'change_to_desired_label'
    }
    data_prefix = "small_" if small_data else ""
    
    
    return {"propDFpath":propDFpath, "result_dir":result_dir, "prop":prop, 
            "TARGET":experiment_target_match[experiment], "dataPrefix":data_prefix}