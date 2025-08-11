import os
from pathlib import Path
from torch import nn


def directory_setup(
    res_dir, save_dir, dataPath, bestModelPath, split_file_name="split.npz"
):
    for dir_path in [save_dir, res_dir, os.path.dirname(dataPath)]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"{dir_path} directory was created.")

    def delete_file(file_path):
        try:
            Path(file_path).unlink()
            print(f"{file_path} unlinked")
        except OSError as e:
            print(e)

    for path in [os.path.join(save_dir, split_file_name), dataPath, bestModelPath]:
        delete_file(path)

    return os.path.join(save_dir, split_file_name)


def predProb(score):
    """returns class label from network score"""
    prob = nn.Sigmoid()
    pred_prob = prob(score)
    if 0 <= pred_prob < 0.5:
        return 0
    else:
        return 1


def ProbnPred(score):
    """returns class label from network score"""
    prob = nn.Sigmoid()
    pred_prob = prob(score)
    if 0 <= pred_prob < 0.5:
        pred = 0
    else:
        pred = 1
    return {"pred": pred, "pred_prob": pred_prob.item()}  # Convert tensor to a scalar
