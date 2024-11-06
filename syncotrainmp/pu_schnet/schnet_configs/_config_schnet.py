# #Can be used to produce different configs for schnet. 
# # Build a json file to configure SchNetPack.
# import os
# import sys
# import json
# import argparse
# from experiment_setup import current_setup, str_to_bool
# # %%
# parser = argparse.ArgumentParser(
#     description="Semi-Supervised ML for Synthesizability Prediction"
# )
# parser.add_argument(
#     "--experiment",
#     default="schnet0",
#     help="name of the experiment and corresponding config files.",
# )
# parser.add_argument(
#     "--ehull",
#     type=str_to_bool,
#     default=False,
#     help="Predicting stability to evaluate PU Learning's efficacy.",
# )
# parser.add_argument(
#     "--small_data",
#     type=str_to_bool,
#     default=False,
#     help="This option selects a small subset of data for checking the workflow faster.",
# )
# args = parser.parse_args(sys.argv[1:])
# experiment = args.experiment 
# ehull_test = args.ehull
# small_data = args.small_data

# cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment)
# propDFpath = cs["propDFpath"]
# result_dir = cs["result_dir"]
# prop = cs["prop"]
# TARGET = cs["TARGET"]
# data_prefix = cs["dataPrefix"]

# # os.chdir("schnet/schnet_configs")
# config = {
#   "experiment": experiment,
#   "epoch_num": 15,
#   "num_iter": 100,
#   "batch_size": 32,
#   "data_dir": "data/clean_data/",
#   "start_iter": 0,
#   "schnetDirectory": "pu_schnet",
#   "small_data": False
# }

# with open(os.path.join(config['schnetDirectory'],'schnet_configs',
#               f'pu_config_{config["experiment"]}.json'), "w") as configJson:
#     json.dump(config, configJson, indent=2)
    
# need to run this at the end to produce new configs.    

# # Build a json file to configure SchNetPack.
# import os
# import json
# os.chdir("schnet/schnet_configs")
# config = {
#   "experiment": "coSchAl1",
#   "cotraining": True,
#   "new_target": 'alignn0',
#   "epoch_num": 15,
#   "num_iter": 100,
#   "batch_size": 32,
#   "fulldatapath": "data/clean_data/synthDF",
#   "start_iter": 0,
#   "schnetDirectory": "schnet",
#   "small_data": False
# }

# with open(config["experiment"]+"_config.json", "w") as configJson:
#     json.dump(config, configJson, indent=2)