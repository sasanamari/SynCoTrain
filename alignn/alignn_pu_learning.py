# %%
# for cotraining I need to produce the csv files.
# I should probebly add a more unique way for multiple cotrainers.
# Need to modify data_size for selecting test-set :/
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from alignn_setup import *
from jarvis.db.jsonutils import loadjson, dumpjson
from synth.myjsonutil import loadjson, dumpjson
import time
import datetime
from jarvis.core.atoms import Atoms
# from alignn.data import get_train_val_loaders
# from alignn.train import train_dgl
from alignn.config import TrainingConfig
import argparse
import random
import pandas as pd
from experiment_setup import current_setup
# %%
parser = argparse.ArgumentParser(
    description="Semi-Supervised ML for Synthesizability Prediction"
)
parser.add_argument(
    "--experiment",
    default="alignn0",
    help="name of the experiment and corresponding config files.",
)
parser.add_argument(
    "--ehull",
    default=False,
    help="Predicting stability to evaluate PU Learning's efficacy.",
)
parser.add_argument(
    "--small_data",
    default=False,
    help="This option selects a small subset of data for checking the workflow faster.",
)
args = parser.parse_args(sys.argv[1:])
experiment = args.experiment 
ehull_test = args.ehull
small_data = args.small_data
cs = current_setup(ehull_test=ehull_test, small_data=small_data)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
data_prefix = "small_" if small_data else ""

data_dir = '/home/samariam/projects/synth/data/clean_data' #full path for now
# changed the target match below, for the constant/dynamic experiment.
experiment_target_match = { #output_dir: training_label_column
            'alignn0':prop, 
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
            'coAlSch3':'coSchAl2',
            'coAlSch4':'coSchAl3',
            'coAlSch5':'coSchAl4',
            'final_class':'change_to_desired_label'
    }

split_id_dir = f"{data_prefix}{experiment_target_match[experiment]}{prop}"
split_id_path = os.path.join(data_dir, split_id_dir)
    
alignn_dir = "alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")

pu_config_name = os.path.join(alignn_config_dir, 
                              f'pu_config_{data_prefix}{experiment}.json')

pu_setup = loadjson(pu_config_name)
cotraining  =pu_setup['cotraining']
id_prop_dat =pu_setup["id_prop_dat"] 
# %%
if cotraining:
    reference_col = experiment_target_match[experiment]
    csvPath = os.path.join(pu_setup['root_dir'], 
                f"{data_prefix}{prop}_id_from_{reference_col}.csv")
    print(f'You have selected {reference_col} as your reference column!!!!!!!!!')
    
else:
    csvPath = os.path.join(pu_setup['root_dir'], 
                f"{data_prefix}{prop}_ids.csv")
    
# %%
def config_generator(
    newConfigName,
    iterNum = 3,
    epochNum = 10,
    class_config='alignn/default_class_config.json',
    alignn_dir = alignn_dir,
    ehull_test = ehull_test
                     ):
    class_config_path = os.path.join(alignn_dir, 'alignn_configs',
                                     class_config+'.json')
    _config = loadjson(class_config_path)
    _config['random_seed'] = iterNum
    _config['epochs'] = epochNum
    _config['output_dir'] = os.path.join(alignn_dir,f'PUOutput_{data_prefix}{experiment}',
                                         f'{str(iterNum)}iter/')
    if ehull_test:
        _config['output_dir'] = os.path.join(alignn_dir,f'PUehull_{experiment}',
                                         f'{str(iterNum)}iter/')

    dumpjson(_config, filename=newConfigName)
    print('Config file for iteratin {} was generated.'.format(iterNum))
    return 
# %%

print("Now we run calculations for iterations", 
      pu_setup['start_of_iterations']," till",pu_setup['max_num_of_iterations'])
# %%
for iterNum in range(pu_setup['start_of_iterations'], 
                     pu_setup['max_num_of_iterations']):
    config_generator(iterNum = iterNum,
                     epochNum= pu_setup['epochs'],
                     class_config='class_config_'+experiment,
                    newConfigName =pu_setup["class_config_name"],
                    alignn_dir = alignn_dir,
                                    )
    
    
    train_for_folder(
        root_dir=pu_setup["root_dir"],
        config_name=pu_setup["class_config_name"],
        keep_data_order=pu_setup["keep_data_order"],
        classification_threshold=pu_setup["classification_threshold"],
        output_dir=pu_setup["output_dir"],
        # batch_size=pu_setup["batch_size"],
        batch_size=None,#this is read separately for each iteration from the generated config file.
        epochs=pu_setup["epochs"],
        file_format=pu_setup["file_format"],
        ehull_test = ehull_test,
        cotraining = cotraining,
        small_data = data_prefix,
        train_id_path = os.path.join(split_id_path, f'train_id_{iterNum}.txt'),
        test_id_path = os.path.join(split_id_path, f'test_id_{iterNum}.txt'),
        id_prop_dat = id_prop_dat,
        # test_strategy = test_strategy,
        csvPath = csvPath, 

    )

# %%
print('PU Learning concluded.')