# This script can be beneficial if the data is quite imbalanced. We did not end up using it for ICSD data.
#  %%
import pandas as pd
# %%
label_path = '/home/samariam/projects/final_syncotrain/SynCoTrain/data/results/synth/synth_labels'
# Write dynamic address before pushing.
synthLab = pd.read_pickle(label_path)
# %%
# Determine the minority class
minority_class = synthLab['synth_labels'].value_counts().idxmin()

# Calculate the difference between the minority class count and majority class
count_diff = len(synthLab) - 2*(synthLab['synth_labels'].value_counts()[minority_class])

# Duplicate rows with the minority class label
minority_rows = synthLab[synthLab['synth_labels'] == minority_class].sample(
    random_state = 42,n=int(count_diff), replace=False)
balanced_data = pd.concat([synthLab, minority_rows])

# Verify the class distribution
# balanced_data['synth_labels'].value_counts()
# %%
balanced_label_path = '/home/samariam/projects/final_syncotrain/SynCoTrain/data/results/synth/synth_balanced_labels'
# %%
balanced_data.to_pickle(balanced_label_path)
# %%
