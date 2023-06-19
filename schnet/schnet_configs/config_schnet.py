# Build a json file to configure SchNetPack.
import os
import json
os.chdir("/home/samariam/projects/synth/schnet/schnet_configs")
config = {
  "experiment": "coSchAl1",
  "cotraining": True,
  "new_target": 'alignn0',
  "epoch_num": 15,
  "num_iter": 100,
  "batch_size": 32,
  "fulldatapath": "/home/samariam/projects/synth/data/clean_data/synthDF",
  "start_iter": 0,
  "schnetDirectory": "/home/samariam/projects/synth/schnet",
  "small_data": False
}

with open(config["experiment"]+"_config.json", "w") as configJson:
    json.dump(config, configJson, indent=2)