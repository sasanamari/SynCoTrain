import os
from pathlib import Path 
from torch import nn
 

def directory_setup(res_dir,dataPath,save_dir, bestModelPath, iteration_num=None):
    if iteration_num == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('Logging directory was created.')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            print('Result directory was created.')
        
    splitFile_path = Path('split.npz') #should this be split.lock?
    try:
        splitFile_path.unlink()
    except OSError as e:
        print(e)
        splitFile_path = Path(os.path.join(save_dir,str(splitFile_path)))
        try:
            splitFile_path.unlink()
        except OSError as e:
            print(e)
            
    datapathObj = Path(dataPath)
    try:
        datapathObj.unlink()
        print('unlinked')
    except OSError as e:
        print(e)        
        
    bestModelPath_obj = Path(bestModelPath)
    try:
        bestModelPath_obj.unlink()
    except OSError as e:
        print(e)       
         
    return str(splitFile_path)

def predProb(score): 
    """returns class label from network score"""
    prob = nn.Sigmoid()
    pred_prob = prob(score)     
    if 0<=pred_prob< 0.5:
        return 0
    else:
        return 1