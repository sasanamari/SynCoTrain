from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
import pandas as pd
import time


# Function to convert seconds to readable time (days, hours, minutes, seconds)
def format_time(seconds):
    days = seconds // (24 * 3600)
    hours = (seconds % (24 * 3600)) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"


synthDF = pd.read_pickle("data/clean_data/synthDF")
oqmd = pd.read_pickle("data/clean_data/oqmd_df.pkl")
# Convert ASE Atoms to Pymatgen Structures and calculate composition
oqmd["structure"] = oqmd["atoms"].apply(lambda x: AseAtomsAdaptor.get_structure(x))
oqmd["composition"] = oqmd["structure"].apply(lambda x: x.composition.reduced_formula)

synthDF["structure"] = synthDF["atoms"].apply(
    lambda x: AseAtomsAdaptor.get_structure(x)
)
synthDF["composition"] = synthDF["structure"].apply(
    lambda x: x.composition.reduced_formula
)

# Create a StructureMatcher with tolerances
matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)


# Function to compare structures only if compositions match and record time
def find_duplicates_by_composition(df1, df2, matcher):
    start_time = time.time()
    duplicates = []
    num_rows = len(df1)

    # Group the materials by composition
    df1_grouped = df1.groupby("composition")
    df2_grouped = df2.groupby("composition")

    for i, comp in enumerate(df1_grouped.groups.keys()):
        if comp in df2_grouped.groups:
            df1_subset = df1_grouped.get_group(comp)
            df2_subset = df2_grouped.get_group(comp)

            for i, struct1 in df1_subset[["material_id", "structure"]].iterrows():
                for j, struct2 in df2_subset[["material_id", "structure"]].iterrows():
                    if matcher.fit(struct1["structure"], struct2["structure"]):
                        duplicates.append(
                            (struct1["material_id"], struct2["material_id"])
                        )

        # Estimate time every 1000 rows of df1
        if (i + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_row = elapsed_time / (i + 1)
            remaining_rows = num_rows - (i + 1)
            estimated_time_left = remaining_rows * avg_time_per_row
            print(
                f"Processed {i + 1} compositions. Estimated time left: {format_time(estimated_time_left)}",
                flush=True,
            )

    total_time = time.time() - start_time
    print(f"Total time taken: {format_time(total_time)}")
    print(f"Total duplicates found: {len(duplicates)}")
    return duplicates


# Run the comparison using composition as filter and save duplicates
print("Finding duplicates by composition...", flush=True)
duplicates_by_composition = find_duplicates_by_composition(oqmd, synthDF, matcher)

# Save the duplicate material_id pairs to a file
duplicates_by_composition_df = pd.DataFrame(
    duplicates_by_composition, columns=["oqmd_material_id", "synthDF_material_id"]
)
duplicates_by_composition_df.to_csv(
    "schnet_pred/data/duplicate_materials_by_composition.csv", index=False
)
