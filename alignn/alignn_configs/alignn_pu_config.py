# Build a json file to configure SchNetPack.
import os
import json
# os.chdir("alignn/alignn_configs")
cotraining = False
experiment_name = 'coAlSch1'
if experiment_name != "alignn0":
    cotraining=True
small_data = True 
data_prefix = "small_" if small_data else ""
max_num_of_iterations = 100
start_of_iterations = 0  
data_dir = "data/clean_data"
root_dir = os.path.join(data_dir,"alignn_format")
pu_setup = dict()
alignn_dir = "alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
coConfigPath = None
default_class_config = os.path.join(alignn_config_dir, 'default_class_config.json')
class_config_name = os.path.join(alignn_config_dir, 'class_config_'+experiment_name+'.json')
pu_config_name = os.path.join(alignn_config_dir, f'pu_config_{data_prefix}{experiment_name}.json')
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
pu_setup["epochs"]= 120
pu_setup["max_num_of_iterations"]= max_num_of_iterations
pu_setup["start_of_iterations"]= start_of_iterations
pu_setup["small_data"]= small_data
# change here for final version
# f = open(os.path.join(data_dest,f"{data_prefix}synth_id_prop.csv"), "w")
pu_setup["id_prop_dat"] = os.path.join(root_dir,f"{data_prefix}synth_id_prop.csv") # for consistency
if cotraining:
    experiment_train_match = { #output_dir: training_label_column
                'alignn0':'synth',
                'coAlSch1':'schnet0',
                'coAlSch2':'coSchAl1',
                'coAlSch3':'coSchAl2',
                'coAlSch4':'coSchAl3',
                'coAlSch5':'coSchAl4',
        }
    reference_col = experiment_train_match[experiment_name]
    pu_setup["id_prop_dat"] = os.path.join(root_dir, 
                f"{data_prefix}synth_id_from_{reference_col}.csv") # for consistency
    

with open(pu_setup["pu_config_name"], "w") as configJson:
    json.dump(pu_setup, configJson, indent=2)


# # Build a json file to configure SchNetPack.
# import os
# import json
# # os.chdir("alignn/alignn_configs")
# # experiment_name = "alignn0"
# # experiment_name = 'coAlSch1'
# experiment_name = 'coAlSch2'
# cotraining = True
# max_num_of_iterations = 99
# start_of_iterations = 1  #default is 1
# data_dir = "data/clean_data"
# root_dir = os.path.join(data_dir,"alignn_format")
# pu_setup = dict()
# alignn_dir = "alignn"
# alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
# coConfigPath = None
# default_class_config = os.path.join(alignn_config_dir, 'default_class_config.json')
# class_config_name = os.path.join(alignn_config_dir, 'class_config_'+experiment_name+'.json')
# pu_config_name = os.path.join(alignn_config_dir, 'pu_config_'+experiment_name+'.json')
# pu_setup["default_class_config"] =default_class_config
# pu_setup["pu_config_name"] =pu_config_name
# pu_setup["class_config_name"] =class_config_name
# pu_setup["cotraining"] =cotraining
# pu_setup["data_dir"]=data_dir
# pu_setup["root_dir"]=root_dir
# pu_setup["file_format"] = "poscar"
# pu_setup["keep_data_order"]=False #overwrites this attrib in config
# pu_setup["classification_threshold"] = 0.5 #also overwrites if present
# pu_setup["batch_size"]=None
# pu_setup["output_dir"] = None
# pu_setup["epochs"]= None
# pu_setup["max_num_of_iterations"]= max_num_of_iterations
# pu_setup["start_of_iterations"]= start_of_iterations


# with open(pu_setup["pu_config_name"], "w") as configJson:
#     json.dump(pu_setup, configJson, indent=2)