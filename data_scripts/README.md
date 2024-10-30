# Querying the Data
The data for this project was obtained from the Materials Project API. A pickled DataFrame containing this data is available in `data/clean_data/synthDF`.

To reproduce the data query, you can use the `icsd_data_query.py` script. Follow these steps to set up the necessary environment and API access:

1. **Install the Materials Project API**:

    - Due to conflicting dependencies—specifically, different `pydantic` versions required by `ALIGNN` and the API—it’s recommended to create a separate Conda (or Mamba) environment.
    - Sign up at the [Materials Project website](https://next-gen.materialsproject.org/api)  and obtain an API key for access.
2. **Set Up the Environment**:

```bash
cd SynCoTrain
mamba create -n query python=3.10 numpy pandas requests typing pymatgen ase jarvis-tools mp-api
mamba activate query
pip install -e . #to enable relative paths
```
3. **Run the Query Script**: By default, `icsd_data_query.py` will query and save a small sample DataFrame to demonstrate the data pipeline. To download the full dataset, uncomment the lines `# num_sites = (1,150)` and `# dataFrame_name = 'synthDF'` in the script.

```bash
python data_scripts/icsd_data_query.py --MPID <your_api_key>
```