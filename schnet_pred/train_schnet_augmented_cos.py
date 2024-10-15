import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(2) # use before loading lightning.gpu
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import schnetpack as spk
from schnetWithDrop import SchNet
from schnetpack.data import ASEAtomsData, AtomsDataModule
import schnetpack.transform as trn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,Callback
import pu_schnet.pu_learn.int2metric as int2metric
from data_scripts.add_noise import format_and_add_noise
import torch.nn as nn
from torch import tensor
# from custom_loss_s import FocalLoss

# from schnet_drop_mod import SchNetWithDropout
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# model1 = {'noise_level':0, 'dropout_embedding':.1, 'dropout_interaction':.25}
# model2 = {'noise_level':0.05, 'dropout_embedding':.1, 'dropout_interaction':.2}
quick_debug = False
noise_level = 0.05 #already applied to the data before augmentation.
dropout_embedding=0.1#0.1
dropout_interaction=0.2#0.2 
epoch_num = 200
# weight_loss = tensor([0.35 / 0.65]).to(device)
# weight_loss = None
weight_loss = tensor([0.45 / 0.55]).to(device)
# ES_patience = 100 #best model is saved anyway
ES = {'patience':70, 'monitor':'val_synth_Accuracy', 'mode':'max'}
# ES = {'patience':60, 'monitor':'val_loss', 'mode':'min'}

## Define the CosineAnnealingWarmRestarts scheduler arguments
# scheduler_args = {
#     "T_0": 3,  # Number of epochs before the first restart
#     "T_mult": 2,  # Multiplier to increase the period length after each restart
#     "eta_min": 1e-6  # Minimum learning rate
# }

# Define the CosineAnnealingLR scheduler arguments
scheduler_args = {
    "T_max": epoch_num,  # Number of epochs for the annealing process
    "eta_min": 1e-6  # Minimum learning rate
}
my_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight_loss)
# my_loss = FocalLoss(gamma=2).to(device)
# my_loss = FocalLoss(gamma=1, pos_weight=weight_loss).to(device)
# my_loss = FocalLoss(gamma=0.5, pos_weight=weight_loss).to(device)
n_interactions = 3
batch_size = 32

# train_val_filename = 'augmented_data.pkl'
# train_val_filename = f'augmented_data_75_{str(int(noise_level * 100))}_noise.pkl'
train_val_filename = 'augmented_data_75_symmetrical.pkl'
# train_val_filename = 'augmented_data_75_balanced.pkl'
# train_val_filename = 'augmented_data_75_10_noise.pkl'



# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_w_cos_sym"
exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_cos_sym_LR"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_cos_sym_sc"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_cos_sym"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_w_cos_loss_sym"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_w_cyclic"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_w_cos_loss"
# exp_id = f"{str(int(noise_level * 100))}_noise_{str(int(dropout_embedding * 100))}_DOE_{str(int(dropout_interaction * 100))}_DOI_aug_75_w_cos"



model_name = f"schnet_model_{exp_id}"
print(f"Model name: {model_name}")

# Directory setup
base_dir = Path(__file__).parent.resolve()
data_dir = base_dir / "data"
model_dir = base_dir / "models"
logs_dir = base_dir / "logs"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Load dataset
train_val_df = pd.read_pickle(data_dir / train_val_filename)
train_length = int(len(train_val_df) * 0.85) #the rest is for validation.

# SchNet Model Configuration
cutoff = 5
n_rbf = 30
n_atom_basis = 64
n_filters = 64
# n_interactions = 3
lr = 1e-3
weight_decay = 1e-6
if quick_debug:
    epoch_num = 2


pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)


# class CustomAtomisticTask(spk.task.AtomisticTask):
#     def __init__(self, model, outputs, optimizer_args=None, scheduler_args=None):
#         super().__init__(model=model, outputs=outputs, optimizer_args=optimizer_args)
#         self.optimizer_args = optimizer_args
#         self.scheduler_args = scheduler_args

#     def configure_optimizers(self):
#         # Define the optimizer using the passed arguments
#         optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)

#         # Define the CosineAnnealingWarmRestarts scheduler
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=self.scheduler_args["T_0"],
#             T_mult=self.scheduler_args["T_mult"],
#             eta_min=self.scheduler_args["eta_min"]
#         )

#         # Return the optimizer and scheduler
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,  # Scheduler used for learning rate adjustments
#                 "interval": "epoch",     # Adjust learning rate after every epoch
#                 "frequency": 1
#             }
#         }

class CustomAtomisticTask(spk.task.AtomisticTask):
    def __init__(self, model, outputs, optimizer_args=None, scheduler_args=None):
        super().__init__(model=model, outputs=outputs, optimizer_args=optimizer_args)
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

    def configure_optimizers(self):
        # Define the optimizer using the passed arguments
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)

        # Define the CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_args["T_max"],  # Set to match the total number of epochs
            eta_min=self.scheduler_args["eta_min"]
        )

        # Return the optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # Scheduler used for learning rate adjustments
                "interval": "epoch",     # Adjust learning rate after every epoch
                "frequency": 1
            }
        }


# Define the optimizer arguments
optimizer_args = {
    "lr": 1e-3
}


# Define the SchNet model
schnet = SchNet(
    n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=n_interactions, radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
    dropout_rate_interaction=dropout_interaction,  # Dropout rate for interaction blocks
    dropout_rate_embedding=dropout_embedding,  # Dropout rate for embedding
)

# Define the Atomwise output layer
pred_prop = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="synth")

# Define the NeuralNetworkPotential
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_prop],
)

# Define the model output with loss function and metrics
output_prop = int2metric.ModelOutput4ACC(
    name="synth",
    loss_fn=my_loss,
    loss_weight=1,
    metrics={
        "Accuracy": torchmetrics.Accuracy("binary"),
        "recall": torchmetrics.Recall("binary"),
    }
)

# Create the AtomisticTask with the custom optimizer configuration
task = CustomAtomisticTask(
    model=nnpot,
    outputs=[output_prop],
    optimizer_args=optimizer_args,
    scheduler_args=scheduler_args
)



# Define the converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.), 
    dtype=torch.float32
)

# Check if the dataset file already exists and delete it if necessary
db_path = data_dir / f"train_dataset_{exp_id}.db"
if db_path.exists():
    db_path.unlink()  # Delete the existing file
    
split_file_path = data_dir / f"split_{exp_id}.npz"
if split_file_path.exists():
    split_file_path.unlink()  # Remove the existing split file
    
# Create the new dataset
class_dataset = ASEAtomsData.create(str(db_path),
                                    distance_unit='Ang',
                                    property_unit_dict={"synth": int(1)})

# Add systems to the dataset
class_dataset.add_systems(np.array(train_val_df.targets), np.array(train_val_df.atoms))


crysData = AtomsDataModule(
    datapath=str(db_path),
    batch_size=batch_size,
    num_train=train_length,
    num_val=len(train_val_df)-train_length,
    transforms=[trn.ASENeighborList(cutoff=float(cutoff)), trn.CastTo32()],
    property_units={"synth": int(1)},
    num_workers=4,
    pin_memory=True,
    split_file=str(split_file_path),
    load_properties=["synth"],
)

crysData.prepare_data()
crysData.setup()

# Callback to save validation accuracy history
class ValidationHistory(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.history = []

    def on_validation_end(self, trainer, pl_module):
        val_acc = trainer.callback_metrics.get("val_synth_Accuracy")
        if val_acc:
            self.history.append(val_acc.item())
            pd.DataFrame(self.history, columns=["val_synth_Accuracy"]).to_csv(self.filename, index=False)

validation_history_callback = ValidationHistory(str(logs_dir / f"validation_history_{model_name}.csv"))

# Callbacks and Trainer Setup
early_stopping = EarlyStopping(
    verbose=2,
    mode=ES['mode'],
    monitor=ES['monitor'],
    min_delta=0.01,
    patience=ES['patience'],
)

logger = pl.loggers.TensorBoardLogger(save_dir=logs_dir, name=model_name)
callbacks = [
    early_stopping,
    validation_history_callback,
    spk.train.ModelCheckpoint(
        monitor="val_synth_Accuracy",
        mode='max',
        save_top_k=1,
        inference_path=model_dir / model_name,
        save_last=False
    )
]


trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    auto_select_gpus=True,
    precision=16,
    callbacks=callbacks,
    logger=logger,
    max_epochs=epoch_num,
    default_root_dir=model_dir,
)

# Train the Model
trainer.fit(task, datamodule=crysData)

# Save the model
model_path = model_dir / model_name
# torch.save(nnpot.state_dict(), f"{model_path}.pt")
# torch.save(nnpot, f"{model_path}.pt")

# print(f"Training completed. Model saved to {model_path}.pt")
