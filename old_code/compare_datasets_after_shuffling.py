# %%
import numpy as np
import pickle
import os
import collections
# %%
os.chdir('/home/samariam/projects/chemheuristics/alignn/checking_datasets')
# %%
dFileList = sorted(os.listdir())
# %%
dlist = ['test0.data',
 'test1.data',
 'train0.data',
 'train1.data',
 'val0.data',
 'val1.data']
ddict = {key:"" for key in dlist}
# %%
for i in range(len(dFileList)):
    with open(dFileList[i], 'rb') as f:
        ddict[dlist[i]] = pickle.load(f)
# %%
t0 = ddict['train0.data']
t1 = ddict['train1.data']
# %%
def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched
# %%
equal_ignore_order(t0, t1)
# %%
t0t = [d['target'] for d in t0]
t1t = [d['target'] for d in t1]
# %%
sorted(t0t) == sorted(t1t)
# %%
sum(t1t)/len(t1t)
# %%
sum(t0t)/len(t0t)

# %%
