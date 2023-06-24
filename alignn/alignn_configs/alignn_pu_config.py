# Build a json file to configure SchNetPack.
import os
import json
# os.chdir("alignn/alignn_configs")
# experiment_name = "alignn0"
# experiment_name = 'coAlSch1'
experiment_name = 'coAlSch2'
cotraining = True
max_num_of_iterations = 99
start_of_iterations = 1  #default is 1
data_dir = "data/clean_data"
root_dir = os.path.join(data_dir,"alignn_format")
pu_setup = dict()
alignn_dir = "alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
coConfigPath = None
default_class_config = os.path.join(alignn_config_dir, 'default_class_config.json')
class_config_name = os.path.join(alignn_config_dir, 'class_config_'+experiment_name+'.json')
pu_config_name = os.path.join(alignn_config_dir, 'pu_config_'+experiment_name+'.json')
pu_setup["default_class_config"] =default_class_config
pu_setup["pu_config_name"] =pu_config_name
pu_setup["class_config_name"] =class_config_name
pu_setup["cotraining"] =cotraining
pu_setup["data_dir"]=data_dir
pu_setup["root_dir"]=root_dir
pu_setup["file_format"] = "poscar"
pu_setup["keep_data_order"]=False #overwrites this attrib in config
pu_setup["classification_threshold"] = 0.5 #also overwrites if present
pu_setup["batch_size"]=None
pu_setup["output_dir"] = None
pu_setup["epochs"]= None
pu_setup["max_num_of_iterations"]= max_num_of_iterations
pu_setup["start_of_iterations"]= start_of_iterations


with open(pu_setup["pu_config_name"], "w") as configJson:
    json.dump(pu_setup, configJson, indent=2)