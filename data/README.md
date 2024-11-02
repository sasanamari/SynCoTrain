## 1. Download Data

1. **Download Experimental Data**
   - Use `data_scripts/icsd_data_query.py` to download experimental data (structures with fewer than 150 atoms) from ICSD.
   - Filter for structures containing oxygen, then save the data to `data/raw/experimental_raw_oxygen.npy`.

2. **Download Theoretical Data from OQMD**
   - Use `data_scripts/jarvis_oqmd_query.py` to download theoretical data from the OQMD via Jarvis.
   - Filter for structures containing oxygen and fewer than 150 atoms, then save the data to `data/raw/theoretical_raw_oxygen.npy`.

## 2. Clean Data

- Run `data_scripts/pymatgen_oxide_clean.py` to ensure all structures are converted to Pymatgen format if not already.
- This script applies the `clean_oxide` function to prepare oxide data by removing unwanted elements and features.
- The cleaned data will be saved in an ASE-compatible format within a DataFrame. Placeholder columns (`np.nan`) will be added for future results.
- Save the final cleaned data to `data/clean_data/synthDF` in `.pickle` format.
