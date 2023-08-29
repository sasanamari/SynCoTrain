def current_setup(ehull_test, small_data):
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
        
    return {"propDFpath":propDFpath, "result_dir":result_dir, "prop":prop}