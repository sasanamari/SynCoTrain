Here I write the step by step instructions for predicting synthesizability. From obtaining the data to plotting the results.

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
We'll save it to data/clean_data/synthDF (pickle format)
Remember to drop duplucate materials via the script in the same directory.
Otherwise the data handling doesn't work!

!Alignn can read ASE instead of poscar. (id will be material id).

!!!I can keep a column placeholder for results. Let PU work produce a dataframe for results.
Replace the place holder column with the resulst in analysis.

!!!sort out directories to store Schnet databases and logs;
You don't need all databases, but each ensamble should have 
its own directory!
!!!REMEMBER to have a separate test set, not a changing one!!!
!!!We need to check at least some of the outliers in the positive class!!!

!!!make sure in the new data hadling, schnet builds..
the new labels in the dictionary form before test and train.

!I think I modified the alignn_setup script to read data from 
the pickled df. Didn't check by running it though.
!!should remove atoms with atomic numbers above 100.
!!Correct the number of epocjs for 3runs of schnet.
!!throw out duplicate materials!
!!!NEED TO CHANGE THE TEST&TRAIN SET IN ALIGNN TO MATCH SCHNET! 