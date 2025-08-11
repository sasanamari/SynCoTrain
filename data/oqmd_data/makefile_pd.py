import numpy as np
import pandas as pd

from jarvis.core.atoms import Atoms


def struct_to_atoms(s):
    s['elements'] = [ item.strip() for item in s['elements'] ]
    atoms = Atoms.from_dict(s)
    atoms = atoms.ase_converter(pbc=True)

    return atoms


array = np.load('oqmd_raw_oxygen.npy', allow_pickle=True)

df = pd.DataFrame(data={
    'atoms': [ struct_to_atoms(item['atoms']) for item in array ],
    'material_id': [ item['_oqmd_entry_id'] for item in array ],
    'band_gap': [ item['_oqmd_band_gap'] for item in array ],
    'delta_e': [ item['_oqmd_delta_e'] for item in array ],
    'stability': [ item['_oqmd_stability'] for item in array ],
})

df.to_pickle('oqmd_df.pkl')
