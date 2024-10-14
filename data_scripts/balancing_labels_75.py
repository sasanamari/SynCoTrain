# %%
import pandas as pd
import os
from pathlib import Path
# %%
# current_dir = Path().resolve() #Path() returns the current directory and resolve() is used to get the absolute path.
current_dir = os.path.dirname(__file__) 
# %%
df = pd.read_pickle(f'{current_dir}/../data/results/synth/synth_labels_2_75')
# %%
min_class = df['synth_labels'].value_counts().idxmin()
max_count = df['synth_labels'].value_counts().max()

df_minority = df[df['synth_labels'] == min_class]
df_majority = df[df['synth_labels'] != min_class]

# Oversample the minority class
df_minority_oversampled = df_minority.sample(n=max_count, replace=True, random_state=42)

# Combining majority class with oversampled minority class
df_balanced = pd.concat([df_majority, df_minority_oversampled])
df_balanced = df_balanced.sample(frac=1, random_state=43).reset_index(drop=True)

# %%
balanced_df_path = Path(f'{current_dir}/../data/results/synth/synth_labels_2_75_balanced').resolve()
df_balanced.to_pickle(balanced_df_path)
# %%
print(f"Balanced data saved at {balanced_df_path}")