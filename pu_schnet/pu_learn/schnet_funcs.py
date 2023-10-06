import os
from pathlib import Path 
from torch import nn
 
def directory_setup(res_dir, save_dir, dataPath, bestModelPath, split_file_name='split.npz'):
    for dir_path in [save_dir, res_dir, os.path.dirname(dataPath)]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'{dir_path} directory was created.')

    def delete_file(file_path):
        try:
            Path(file_path).unlink()
            print(f'{file_path} unlinked')
        except OSError as e:
            print(e)

    for path in [os.path.join(save_dir, split_file_name), dataPath, bestModelPath]:
        delete_file(path)

    return os.path.join(save_dir, split_file_name)



def predProb(score): 
    """returns class label from network score"""
    prob = nn.Sigmoid()
    pred_prob = prob(score)     
    if 0<=pred_prob< 0.5:
        return 0
    else:
        return 1
    

# def directory_setup(res_dir,dataPath,save_dir, bestModelPath, iteration_num=None):
#     if iteration_num == 0:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             print('Logging directory was created.')
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
#             print('Result directory was created.')
        
#     splitFile_path = Path('split.npz') 
#     try:
#         splitFile_path.unlink()
#     except OSError as e:
#         print(e)
#         splitFile_path = Path(os.path.join(save_dir,str(splitFile_path)))
#         try:
#             splitFile_path.unlink()
#         except OSError as e:
#             print(e)
            
#     datapathObj = Path(dataPath)
#     try:
#         datapathObj.unlink()
#         print('unlinked')
#     except OSError as e:
#         print(e)        
        
#     bestModelPath_obj = Path(bestModelPath)
#     try:
#         bestModelPath_obj.unlink()
#     except OSError as e:
#         print(e)       
         
#     return str(splitFile_path)    