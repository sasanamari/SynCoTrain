1- Download data.
a) download experimental data (less than 150 atoms) from icsd by data_scripts.icsd_data_query.py, 
filter for oxygen, save it in data/raw/experimental_raw_oxygen.npy
b) download theoretical OQMD data from Jarvis through data_scripts.jarvis_oqmd_query, 
filter for oxygen and less than 150 atoms, save it in data/raw/theoretical_raw_oxygen.npy

2- Clean data.
data_scripts/pymatgen_oxide_clean.py script converts structure to pymatgen if they are not.
It uses the clean_oxide function to clean data for oxides.
We'll save the data in ASE format in a dataframe. 
np.nan columns are place-holders for results.
We'll save it to data/clean_data/synthDFin .pickle format