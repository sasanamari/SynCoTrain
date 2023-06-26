# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from alignn_setup import *
# from jarvis.db.jsonutils import loadjson, dumpjson
from myjsonutil import loadjson, dumpjson
from jarvis.core.atoms import Atoms
from alignn.config import TrainingConfig
import argparse
import pandas as pd
# %%
experiment_name = "coAlSch2" #this could be the argument.
# normal_experiment = False
ehull_test = False
reverse_label=False #to use the prtrained model
# cotraining = True #choose the desired reference_col below.
# if sum([cotraining, reverse_label, ehull_test, normal_experiment])!= 1:
#     print("Contradiction in experiment setup!!!")
#     exit()
    
alignn_dir = "alignn"
alignn_config_dir = os.path.join(alignn_dir,"alignn_configs")
pu_config_name = os.path.join(alignn_config_dir, 'pu_config_'+experiment_name+'.json')

pu_setup = loadjson(pu_config_name)
cotraining  =pu_setup['cotraining']
# %%
coConfigPath = None
if cotraining:
    experiment_train_match = { #output_dir: training_label_column
            'alignn0':'synth',
            'coAlSch1':'schnet0',
            'coAlSch2':'coSchAl1',
                            }
    train_col = experiment_train_match[experiment_name]
    origdataPath = os.path.join(
        pu_setup['data_dir'],'synthDF'
        )
    csvPath = os.path.join(
            pu_setup['root_dir'],"synth_id_from_"+train_col+".csv")
    
    print(f'You have selected {train_col} as your reference column!!!!!!!!!')
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
                reference_col = train_col, dest = pu_setup['root_dir'])
    
# %%
def config_generator(
    newConfigName,
    iterNum = 3,
    epochNum = 10,
    default_class_config='alignn/default_class_config.json',
    alignn_dir = alignn_dir,
                     ):
    
    _config = loadjson(default_class_config)
    _config['random_seed'] = iterNum
    _config['epochs'] = epochNum
    _config['output_dir'] = os.path.join(alignn_dir,'PUOutput_'+experiment_name+'/'+str(iterNum)+'iter/')

    dumpjson(_config, filename=newConfigName)
    print('Config file for iteratin {} was generated.'.format(iterNum))
    return 
# %%

print("Now we run calculations for iterations", 
      pu_setup['start_of_iterations']," till",pu_setup['max_num_of_iterations']+1)
# %%
for iterNum in range(pu_setup['start_of_iterations'], 
                     pu_setup['max_num_of_iterations']+1):
    config_generator(iterNum = iterNum,
                    newConfigName =pu_setup["class_config_name"],
                    alignn_dir = alignn_dir,
                                    )
    
    
    train_for_folder(
        root_dir=pu_setup["root_dir"],
        config_name=pu_setup["class_config_name"],
        keep_data_order=pu_setup["keep_data_order"],
        classification_threshold=pu_setup["classification_threshold"],
        output_dir=pu_setup["output_dir"],
        batch_size=pu_setup["batch_size"],
        epochs=pu_setup["epochs"],
        file_format=pu_setup["file_format"],
        reverse_label=reverse_label,
        ehull_test = ehull_test,
        cotraining = cotraining,
        coConfigPath = coConfigPath,
    )

# %%
print('PU Learning concluded.')