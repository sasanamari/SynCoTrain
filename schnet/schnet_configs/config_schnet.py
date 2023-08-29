# Build a json file to configure SchNetPack.
import os
import json
os.chdir("schnet/schnet_configs")
config = {
  "experiment": "coSchAl2",
  "cotraining": True,
  "new_target": 'coAlSch1',
  "epoch_num": 15,
  "num_iter": 100,
  "batch_size": 32,
  "data_dir": "data/clean_data/",
  "start_iter": 0,
  "schnetDirectory": "schnet",
  "small_data": False
}

with open(f'pu_config_{config["experiment"]}.json', "w") as configJson:
    json.dump(config, configJson, indent=2)

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