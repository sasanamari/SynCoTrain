import json
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

from syncotrainmp.utility.crystal_funcs import clean_oxide

# load structures from json
with open("2022-10-19-wbm-computed-structure-entries+init-structs.json") as file:
    data = json.load(file)

wbm_data = list(data['computed_structure_entry'].values())

# convert structure dicts
for d in wbm_data:
    d['material_id'] = d['entry_id']
    del d['entry_id']
    d['structure'] = Structure.from_dict(d['structure'])

# filter oxides
oxs = [d for d in wbm_data if Element('O') in d['structure'].elements]

good_data = clean_oxide(experimental=False, pymatgenArray = oxs, reportBadData=False, read_oxide_type = False)

# create data frame
df = pd.DataFrame.from_records(good_data)
df = df.drop(columns=['@module', '@class', 'data'])

df.to_pickle('wbm_oxides.pkl')
