# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from alignn_setup import *
from jarvis.db.jsonutils import loadjson, dumpjson
from synth.myjsonutil import loadjson, dumpjson
import time
import argparse
from experiment_setup import current_setup
from alignn_configs.alignn_pu_config import alignn_pu_config_generator
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
cs = current_setup(ehull_test=ehull_test, small_data=small_data, experiment=experiment)
propDFpath = cs["propDFpath"]
result_dir = cs["result_dir"]
prop = cs["prop"]
TARGET = cs["TARGET"]
data_prefix = cs["dataPrefix"]

data_dir = os.path.dirname(propDFpath)
split_id_dir = f"{data_prefix}{TARGET}{prop}"
split_id_path = os.path.join(data_dir, split_id_dir)
    
alignn_dir = "alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
pu_config_name = alignn_pu_config_generator(experiment, small_data, ehull_test)
pu_setup = loadjson(pu_config_name)
    
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
start_time = time.time()
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
        small_data = data_prefix,
        train_id_path = os.path.join(split_id_path, f'train_id_{iterNum}.txt'),
        test_id_path = os.path.join(split_id_path, f'test_id_{iterNum}.txt'),
        experiment = experiment, 
        # ehull_test = ehull_test, 
        # small_data = small_data,
                    )


    elapsed_time = time.time() - start_time
    remaining_iterations = pu_setup['max_num_of_iterations'] - iterNum - 1
    time_per_iteration = elapsed_time / (iterNum - pu_setup['start_of_iterations'] + 1)
    estimated_remaining_time = remaining_iterations * time_per_iteration
    remaining_days = int(estimated_remaining_time // (24 * 3600))
    remaining_hours = int((estimated_remaining_time % (24 * 3600)) // 3600)

    with open(f'alignn_remaining_time_{data_prefix}{experiment}_{prop}.txt', 'w') as file:
        file.write(f"Iterations completed: {iterNum - pu_setup['start_of_iterations']}\n")
        file.write(f"Iterations remaining: {remaining_iterations}\n")
        file.write(f"Estimated remaining time: {remaining_days} days, {remaining_hours} hours\n")

    print(f"Iteration {iterNum} completed. Remaining time: {remaining_days} days, {remaining_hours} hours")

# %%
# Final summary
elapsed_days = int(elapsed_time // (24 * 3600))
elapsed_hours = int((elapsed_time % (24 * 3600)) // 3600)

with open(f'alignn_remaining_time_{data_prefix}{experiment}_{prop}.txt', 'w') as file:
    file.write(f"Iterations completed: {pu_setup['max_num_of_iterations'] - pu_setup['start_of_iterations']}\n")
    file.write(f"Total time taken: {elapsed_days} days, {elapsed_hours} hours\n")

print(f"PU Learning completed. Total time taken: {elapsed_days} days, {elapsed_hours} hours")