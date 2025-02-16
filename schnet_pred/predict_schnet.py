# predict_schnet.py
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import schnetpack as spk
from pu_schnet.pu_learn.Datamodule4PU import DataModuleWithPred
from pu_schnet.pu_learn.schnet_funcs import ProbnPred
import torchmetrics
import pytorch_lightning as pl
import syncotrainmp.pu_schnet.pu_learn.int2metric as int2metric
import numpy as np

# Set up argument parser
parser = argparse.ArgumentParser(description="Predict using pre-trained SchNet model.")
parser.add_argument(
    "--input_file",
    type=str,
    default="test_df_aug_75_symmetrical",
    help="Name of the .pkl file containing the test data.",
)
parser.add_argument(
    "--gpu", type=str, default="0", help="CUDA device to use, e.g., 'cuda:0'."
)
args = parser.parse_args()

# Use the provided arguments
test_filename = args.input_file
cuda_device = args.gpu
# threshold = args.threshold


# from schnet_drop_mod import SchNetWithDropout #need to import/define this class to read the model

# Quick debug flag
quick_debug = False
noise_level = 0.05
dropout_embedding = 0.1
dropout_interaction = 0.2
# make sure to use test_df_aug_75_symmetrical.pkl for cos sym experiment.
# Directory setup
base_dir = Path(__file__).parent.resolve()
data_dir = base_dir / "data"
model_dir = base_dir / "models"
result_dir = base_dir / "results"
os.makedirs(result_dir, exist_ok=True)

exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_cos_sym_sc"

# Define paths
model_name = f"schnet_model_{exp_id}"

output_file_base = Path(f"{test_filename}_{exp_id}")

prop = "synth"  # The property to predict

# Adjust paths and filenames for quick debug
if quick_debug:
    model_name = "schnet_debug_model"
    test_filename = "debug_test_df"

# Load the test data
test_df = pd.read_pickle(data_dir / f"{test_filename}.pkl")

# Prepare the data module for predictions
test_db_path = data_dir / f"{test_filename}_{exp_id}.db"
if test_db_path.exists():
    test_db_path.unlink()  # Remove old test data if it exists
# Remove the split file if it exists to prevent interference
split_file_path = data_dir / f"split_{test_filename}_{exp_id}.npz"
if split_file_path.exists():
    split_file_path.unlink()  # Remove the existing split file


# Load the saved model (no need to set up architecture manually)
model_path = model_dir / model_name
device = torch.device(f"cuda:{cuda_device}")
nnpot = torch.load(model_path, map_location=device)

# Set the model to evaluation mode (disables dropout)
nnpot.eval()

# Prepare the task (LightningModule) with the loaded model
output_prop = int2metric.ModelOutput4ACC(
    name=prop,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    loss_weight=1.0,
    metrics={
        "Accuracy": torchmetrics.Accuracy("binary"),
        "recalll": torchmetrics.Recall("binary"),
    },
)

# Prepare necessary parameters
cutoff = 5  # Use the same cutoff value as in training
batch_size = 32  # Use the same batch size as in training
lr = 1e-3

scheduler = {
    "scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "scheduler_args": {
        "mode": "max",  # Mode is 'max' because we monitor accuracy
        "factor": 0.7,
        "patience": 10,
        "threshold": 0.01,
        "min_lr": 1e-6,
    },
    "scheduler_monitor": "val_synth_Accuracy",
}

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_prop],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": lr},
    scheduler_monitor=scheduler["scheduler_monitor"],
    scheduler_cls=scheduler["scheduler_cls"],
    scheduler_args=scheduler["scheduler_args"],
)

test_dataset = spk.data.ASEAtomsData.create(
    str(test_db_path), distance_unit="Ang", property_unit_dict={"dummy": int(1)}
)

# Add systems to the dataset with a minimal dummy property

dummy_properties = [
    {"dummy": np.array([0.0])} for _ in range(len(test_df))
]  # Minimal property to satisfy requirement
test_dataset.add_systems(np.array(dummy_properties), np.array(test_df.atoms))


crysTest = DataModuleWithPred(
    datapath=str(test_db_path),
    batch_size=batch_size,
    num_train=0,
    num_val=0,
    num_test=len(test_df),
    transforms=[
        spk.transform.ASENeighborList(cutoff=float(cutoff)),
        spk.transform.CastTo32(),
    ],
    property_units={"dummy": int(1)},  # Still using the dummy property
    num_workers=4,
    pin_memory=True,
    split_file=str(split_file_path),
    load_properties=[],  # No properties to load
)


crysTest.prepare_data()
crysTest.setup("test")

# Make predictions
trainer = pl.Trainer(accelerator="gpu", gpus=1, auto_select_gpus=True, precision=16)
predictions = trainer.predict(
    model=task, dataloaders=crysTest.predict_dataloader(), return_predictions=True
)

# Process predictions and extract both prediction and score
results = []
for batch in predictions:
    for datum in batch[prop]:
        result = ProbnPred(datum.float())
        results.append(result)

# Recreate the results dataframe with indices and separate the columns for prediction and probability
res_list = []
for i, datum in enumerate(crysTest.test_dataset):
    ind = int(datum["_idx"])  # Get the original index
    pred_synth = results[i]["pred"]
    synth_score = results[i]["pred_prob"]
    res_list.append([ind, pred_synth, synth_score])
    if i % 1000 == 0:
        print(f"Processed {i} out of {len(crysTest.test_dataset)} predictions.")

# Create a DataFrame from the results
resdf = pd.DataFrame(res_list, columns=["testIndex", "pred_synth", "synth_score"])
resdf = resdf.set_index("testIndex").sort_index()

# Merge predictions back to the original test dataframe
test_df = test_df.merge(
    resdf[["pred_synth", "synth_score"]], left_index=True, right_index=True, how="outer"
)


# Save results to CSV
output_path = result_dir / f"{output_file_base}_predictions.csv"
if quick_debug:
    output_path = result_dir / "debug_predictions.csv"

test_df.to_csv(output_path, index=False)

print("Predictions:")
print(test_df.head())
print(f"The average prediction was {test_df['pred_synth'].mean():.4f}")
print(f"Predictions saved to {output_path}")
