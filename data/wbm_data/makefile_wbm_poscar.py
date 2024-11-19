import pandas as pd
import os

# create output directory
save_dir = "wbm_oxides"
os.makedirs(save_dir, exist_ok=True)

# load wbm oxides data frame
df = pd.read_pickle("wbm_oxides.pkl")

# writing poscars into save_dir for prediction
for i, row in df.iterrows():
    crystal = row['structure']
    filename = f"POSCAR-{row['material_id']}.vasp"
    filepath = os.path.join(save_dir, filename)
    crystal.to(filename=filepath, fmt='poscar')
