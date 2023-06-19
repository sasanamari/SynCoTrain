# This supresses warnings.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.ext.matproj import MPRester
import requests
import json

MPID = "Q0tUKnAE52sy7hVO"

# request below is just made to print the version of the database and pymatgen
response = requests.get(
    "https://www.materialsproject.org/rest/v2/materials/mp-1234/vasp",
    {"API_KEY": MPID})
response_data = json.loads(response.text)
print(response_data.get('version'))
# request above is just made to print the version of the database and pymatgen


with MPRester(api_key=MPID) as mpr:

    data = mpr.query(
    
        criteria={
            "icsd_ids": {"$ne": []}, #allows data with existing "icsd_ids" tag
            "theoretical": {"$ne": True}, #allows data without the "theoretical" tag
            "elements": {"$all": ["O"]}, #allows for crystals with Oxygen present
            "oxide_type": {"$all": ["oxide"]}, #allows for oxides (e.g. not peroxide)
            "nelements": {"$gte": 2} #allows crystals with at least 2 elements
                  },
        
        properties=[
            "exp.tags", "icsd_ids", "formula", "pretty_formula", "structure",
            "material_id", "theoretical"
                    ]   
        
                    )
        
arrdata = np.array(data) #converts list to array, much faster to work with
del data #free up memory

np.save("arrdata", arrdata)
