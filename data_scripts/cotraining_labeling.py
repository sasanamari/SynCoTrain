# %%
import os
import sys
import numpy as np
import pandas as pd
# %%
theoretical_path = "/home/samariam/projects/chemheuristics/data/theoretical"
experimental_path = "/home/samariam/projects/chemheuristics/data/experimental"
new_labels_path = '/home/samariam/projects/chemheuristics/data/alignn_0.pkl'
# %%
np.random.seed(42)
The following should be removed!
pAtoms = np.load(os.path.join(experimental_path,"schnet_experimental_data.npy"), allow_pickle=True)
tAtoms = np.load(os.path.join(theoretical_path,"schnet_theoretical_data.npy"), allow_pickle=True)
print("size of positive data is", len(pAtoms))
print("size of unlabeled data is", len(tAtoms))

# %%
def cotrain_labeling_schnet(positive_data, unlabeled_data, new_labels_path):
    df0 = pd.read_pickle(new_labels_path)
    df0 = df0.reset_index()
    # df0_pred_pos = df0[(df0.target == 0) & (df0.prediction == 1)]
    df0_pred_pos = df0[(df0.original_target == 0) & (df0.prediction == 1)]
    new_labels = [str(datum) for datum in df0_pred_pos.material_id.values]
    move_to_positive = []
    remove_from_unlabeled = []
    for i,d in enumerate(unlabeled_data):
        if str(d["material_id"]) in new_labels:
            move_to_positive.append(d)
            remove_from_unlabeled.append(i)
            
    if len(move_to_positive) < len(positive_data)*0.005:
        print("Can't detect enough new labels for cotraining!")
        sys.exit()
    newtAtoms = np.delete(unlabeled_data, remove_from_unlabeled)
    newpAtoms = np.append(positive_data,move_to_positive)
    print("size of positively labeled data is", len(newpAtoms))
    print("size of remaining unlabeled data is", len(newtAtoms))
    return newpAtoms , newtAtoms
# %%
# newp, newt = cotrain_labeling_schnet(pAtoms, tAtoms, new_labels_path)
# %%

# %%
