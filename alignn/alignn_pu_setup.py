# %%
# for cotraining I need to produce the csv files.
# I should probebly add a more unique way for multiple cotrainers.
# Need to modify data_size for selecting test-set :/
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
# %%
# ##config file (class) does not accept precsion/accuracy as criterion.
start_time =time.time()
normal_experiment = False
ehull_test = False
cotraining = True #choose the desired reference_col below.
reference_col = "coSchAl1"
reverse_label=False #to use the prtrained model
if sum([cotraining, reverse_label, ehull_test, normal_experiment])!= 1:
    print("Contradiction in experiment setup!!!")
    exit()
    
experiment_str = "_coAlSch2"#"_100runs"#"longlightDBug"#
max_num_of_iterations = 99#9#
start_of_iterations = 1  #default is 1
epochs = 120#150
batch_size = 64
n_early_stopping = 25
num_workers = 4
# root_dir = "./alignn/sample_synth/"
# root_dir = "/home/samariam/projects/chemheuristics/data/alignn_full_data/"
data_dir = "/home/samariam/projects/synth/data/clean_data"
root_dir = os.path.join(data_dir,"alignn_format")
ml_setup = dict()
alignn_layers = 4
gcn_layers = 4
hidden_features = 256
weight_decay = 1e-5 #default is 1e-5
save_dataloader = False
learning_rate = 0.001
warmup_steps = 0
# earlyStopMetric = "recall"  # or "accuracy"?
# %%
# ##something is wrong with root_dir, it is saved as tuple!
# alignn_dir = "/home/samariam/projects/chemheuristics/alignn"
alignn_dir = "/home/samariam/projects/synth/alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
# ml_setup["root_dir"] = "./alignn/sample_synth/",
ml_setup["config_name"] =os.path.join(alignn_config_dir, "new_config"+experiment_str+".json")
ml_setup["root_dir"]=root_dir
ml_setup["file_format"] = "poscar"
ml_setup["keep_data_order"]=False #overwrites this attrib in config
ml_setup["classification_threshold"] = 0.5 #also overwrites if present
ml_setup["batch_size"]=None
# ml_setup["batch_size"]=batch_size
ml_setup["output_dir"] = None
ml_setup["epochs"]= None

configName = os.path.join(alignn_config_dir, 'config_example_class.json')
# configName = '/home/samariam/projects/chemheuristics/alignn/configForDebug.json'
newConfigName = os.path.join(alignn_config_dir, 'new_config'+experiment_str+'.json')
coConfigPath = None
# %%
if cotraining:
    origdataPath = os.path.join(
        data_dir,'synthDF'
        )
    csvPath = os.path.join(
            root_dir,"synth_id_from_"+reference_col+".csv")
    
    print(f'You have selected {reference_col} as your reference column!!!!!!!!!')
    def cotrain_config(
        csvPath,
        origdataPath,
        reference_col,
        dest,
    ):
        coConfig = {}
        coConfig['csvPath'] = csvPath
        synthDF = pd.read_pickle(origdataPath)
        coConfig['experimentalDataSize'] = int(synthDF.synth.sum())
        coConfig['idsPositiveLabel'] = list(synthDF.loc[synthDF[reference_col]==1].index)
        refCoConfigPath =os.path.join(dest, reference_col+'_coConfig.json')
        dumpjson(coConfig, filename=refCoConfigPath)
        print('Cotrain config was saved.')
        return refCoConfigPath
    
    coConfigPath = cotrain_config(csvPath=csvPath, origdataPath=origdataPath,
                reference_col = reference_col, dest = root_dir)
    # sys.exit()

    
# %%
def config_generator(
    iterNum = 3,
    epochNum = 10,
    batchSize = 64,
    n_early_stopping = None,
    configName='alignn/config_example.json',
    newConfigName =newConfigName,
    rootDir = "./alignn/sample_synth/",
    num_workers = num_workers,
    # earlyStopMetric = "recall",
    alignn_layers= 4,
    gcn_layers= 4,
    hidden_features = 256,
    weight_decay = 1e-5,
    save_dataloader = False,
    learning_rate = 0.0001,
    warmup_steps = 0,
    alignn_dir = alignn_dir,
                     ):
    
    _config = loadjson(configName)
    _config['random_seed'] = iterNum
    _config['epochs'] = epochNum
    _config['batch_size'] = batchSize
    _config["n_early_stopping"] = n_early_stopping
    _config['output_dir'] = os.path.join(alignn_dir,'PUOutput'+experiment_str+'/'+str(iterNum)+'iter/')
    _config['num_workers'] = num_workers
    _config["model"]['alignn_layers'] = alignn_layers
    _config["model"]['gcn_layers'] = gcn_layers
    _config["model"]['hidden_features'] = hidden_features
    _config["weight_decay"] = weight_decay
    _config["save_dataloader"] = save_dataloader
    _config["learning_rate"] = learning_rate
    _config["warmup_steps"] = warmup_steps
    # _config['earlyStopMetric'] = earlyStopMetric

    
    dumpjson(_config, filename=newConfigName)
    
    return print('Config file for iteratin {} was generated.'.format(iterNum))
# %%

print("Now we run calculations for iterations", 
      start_of_iterations," till",max_num_of_iterations+1)
# %%
for iterNum in range(start_of_iterations, max_num_of_iterations+1):
    config_generator(iterNum = iterNum,
                    epochNum = epochs,
                    batchSize = batch_size,
                    n_early_stopping = n_early_stopping,
                    configName=configName,
                    newConfigName =newConfigName,
                    rootDir = root_dir,
                    alignn_layers = alignn_layers,
                    gcn_layers = gcn_layers,
                    hidden_features = hidden_features,
                    weight_decay = weight_decay,
                    save_dataloader = save_dataloader,
                    learning_rate = learning_rate,
                    warmup_steps = warmup_steps,
                    alignn_dir = alignn_dir,
                    # earlyStopMetric = earlyStopMetric
                                    )
    
    
    train_for_folder(
        root_dir=ml_setup["root_dir"],
        config_name=ml_setup["config_name"],
        keep_data_order=ml_setup["keep_data_order"],
        classification_threshold=ml_setup["classification_threshold"],
        output_dir=ml_setup["output_dir"],
        batch_size=ml_setup["batch_size"],
        epochs=ml_setup["epochs"],
        file_format=ml_setup["file_format"],
        reverse_label=reverse_label,
        ehull_test = ehull_test,
        cotraining = cotraining,
        coConfigPath = coConfigPath,
    )

    try:  #just reporting progress in a small text file.
        current_time = time.time()
        deltaT = current_time - start_time
        num_of_iterations = max_num_of_iterations - start_of_iterations +1
        avgIterTime = deltaT/num_of_iterations
        estimateRemain = (max_num_of_iterations - iterNum) * avgIterTime
        convertToMinute = str(datetime.timedelta(seconds = estimateRemain))


        with open("iterReport"+experiment_str+".txt", "w") as text_file:
            print(f"So far we have finished iteration #{iterNum}.", file=text_file)
            print(F"Estimated remaining time is {convertToMinute}", file=text_file)
    except:
        pass
# %%
end_time = time.time()
total_time = end_time-start_time
print('The script took {} seconds for {} iterations of {} epochs of batch size {}.'.format(
    total_time, max_num_of_iterations, epochs, batch_size
))
print('Average time per iteration was {}'.format(avgIterTime))