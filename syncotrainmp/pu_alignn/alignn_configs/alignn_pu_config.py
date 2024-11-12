# Build a json file to configure SchNetPack.
import os
import json

def alignn_pu_config_generator(experiment, cs, small_data, output_dir):
    prop = cs["prop"]
    data_prefix = cs["dataPrefix"]
    max_num_of_iterations = 60#100
    start_of_iterations = 0  
    data_dir = "data/clean_data"
    root_dir = os.path.join(data_dir,"alignn_format")
    pu_setup = dict()
    alignn_dir = os.path.join(output_dir, "pu_alignn")
    default_class_config = os.path.join(alignn_dir, 'default_class_config.json')
    class_config_name = os.path.join(alignn_dir, f'class_config_{data_prefix}{experiment}_{prop}.json')
    pu_config_name = os.path.join(alignn_dir, f'pu_config_{data_prefix}{experiment}_{prop}.json')
    pu_setup["default_class_config"] = default_class_config
    pu_setup["pu_config_name"] = pu_config_name
    pu_setup["class_config_name"] = class_config_name
    pu_setup["data_dir"] = data_dir
    pu_setup["root_dir"] = root_dir
    pu_setup["file_format"] = "poscar"
    pu_setup["keep_data_order"] = False #overwrites this attrib in config
    pu_setup["classification_threshold"] = 0.5 #also overwrites if present
    pu_setup["batch_size"] = None
    pu_setup["output_dir"] = None
    pu_setup["epochs"] = 120
    pu_setup["max_num_of_iterations"] = max_num_of_iterations
    pu_setup["start_of_iterations"]= start_of_iterations
    pu_setup["small_data"]= small_data
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pu_setup["pu_config_name"], "w+") as configJson:
        json.dump(pu_setup, configJson, indent=2)

    print(f'New PU Alignn pu_config_{data_prefix}{experiment}_{prop}.json was generated.')
        
    return pu_config_name
