import numpy as np
import sys
import os
from pymatgen.ext.matproj import MPRester
import requests
import json
import pathlib
pathlib.Path().resolve()
from pymatgen.core.structure import Structure
import pickle

# print(os.getcwd())

from pymatgen.ext.optimade import OptimadeRester
import qmpy_rester as qr

from pymatgen.ext.optimade import OptimadeRester



with OptimadeRester(aliases_or_resource_urls=["oqmd"],timeout=150) as r:

    struct=r.get_structures(elements=['O'], nelements= [2,4], nsites = [2,5])

struct = struct['oqmd']

theo_data = np.empty(0)
dict_data = dict()
for s in struct:
    dict_data['material_id'] = str(s)
    dict_data['structure'] = struct[s]
    theo_data = np.append(theo_data, dict_data)

np.save('theoretical_data/theoretical_structures', theo_data)
