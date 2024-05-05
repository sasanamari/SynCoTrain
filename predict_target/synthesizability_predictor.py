#!/usr/bin/env python

"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
from tqdm import tqdm
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.data import get_torch_dataset
from torch.utils.data import DataLoader
import tempfile
import torch
import sys
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson

# from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson
import pandas as pd
import csv

tqdm.pandas()

"""
Name of the model, figshare link, number of outputs,
extra config params (optional)
"""


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network Pretrained Models"
)
parser.add_argument(
    "--model_name",
    default="syncotrain",
    # help="Choose a model from these "
    # + str(len(list(all_models.keys())))
    # + " models:"
    # + ", ".join(list(all_models.keys())),
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--directory_name",
    # default="alignn/examples/sample_data/POSCAR-JVASP-10.vasp",
    default="predictor_toy_data",
    help="Path to file.",
)


parser.add_argument(
    "--cutoff",
    default=8,
    help="Distance cut-off for graph constuction"
    + ", usually 8 for solids and 5 for molecules.",
)

parser.add_argument(
    "--max_neighbors",
    default=12,
    help="Maximum number of nearest neighbors in the periodic atomistic graph"
    + " construction.",
)
parser.add_argument(
    "--config_name",
    default="predict_target/config.json",
    help="Name of the config file",
)
parser.add_argument(
    "--iter",
    default="2",
    help="Iteration of co-training to use for labels.",
)
parser.add_argument(
    "--output_name",
    default="synth_preds",
    help="name of the csv file to save the predictions",
)

args = parser.parse_args(sys.argv[1:])


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_config(config_name = "predict_target/config.json"):
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check this expection here", exp)
    return config

def load_model_from_checkpoint(checkpoint_file = f"predict_target/synth_final_preds_{args.iter}/best_model.pt",  
# def load_model_from_checkpoint(checkpoint_file = "predict_target/synth_final_preds/checkpoint_120.pt",  
# def load_model_from_checkpoint(checkpoint_file = "predict_target/synth_final_preds/unbalanced_checkpoint_120.pt",  
                               output_features=1):
    config = get_config()
    model = ALIGNN(config=config.model)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model


def get_prediction(
    model_name="synchotrain",
    atoms=None,
    cutoff=8,
    max_neighbors=12,
):
    """Get model prediction on a single structure."""
    # model = get_figshare_model(model_name)
    model = load_model_from_checkpoint()
    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(
        atoms, cutoff=float(cutoff), max_neighbors=max_neighbors,
    )
    # out_data = (
    #     model([g.to(device), lg.to(device)])
    #     .detach()
    #     .cpu()
    #     .numpy()
    #     .flatten()
    #     .tolist()
    # )
    out_data = model([g.to(device), lg.to(device)])
    top_p, top_class = torch.topk(torch.exp(out_data), k=1)
    out_class = top_class.cpu().numpy().flatten().tolist()[0]
    # return out_data
    return out_class


def get_multiple_predictions(
    atoms_array=[],
    cutoff=8,
    neighbor_strategy="k-nearest",
    max_neighbors=12,
    use_canonize=True,
    target="prop",
    atom_features="cgcnn",
    line_graph=True,
    workers=0,
    filename="pred_data.json",
    include_atoms=True,
    pin_memory=False,
    output_features=1,
    batch_size=1,
    model=None,
    model_name="jv_formation_energy_peratom_alignn",
    print_freq=100,
):
    """Use pretrained model on a number of structures."""
    # import glob
    # atoms_array=[]
    # for i in glob.glob("alignn/examples/sample_data/*.vasp"):
    #      atoms=Atoms.from_poscar(i)
    #      atoms_array.append(atoms)
    # get_multiple_predictions(atoms_array=atoms_array)

    mem = []
    for i, ii in enumerate(atoms_array):
        info = {}
        info["atoms"] = ii.to_dict()
        info["prop"] = -9999  # place-holder only
        info["jid"] = str(i)
        mem.append(info)

    if model is None:
        try:
            # model = get_figshare_model(model_name)
            model = load_model_from_checkpoint()
        except Exception as exp:
            raise ValueError(
                'Check is the model name exists using "pretrained.py -h"', exp
            )
            pass

    # Note cut-off is usually 8 for solids and 5 for molecules
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=True,
            use_canonize=use_canonize,
        )

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, id in zip(test_loader, ids):
            g, lg, target = dat
            out_data = model([g.to(device), lg.to(device)])
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()
            info = {}
            info["id"] = id
            info["pred"] = out_data
            results.append(info)
            print_freq = int(print_freq)
            if len(results) % print_freq == 0:
                print(len(results))
    df1 = pd.DataFrame(mem)
    df2 = pd.DataFrame(results)
    df2["jid"] = df2["id"]
    df3 = pd.merge(df1, df2, on="jid")
    save = []
    for i, ii in df3.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["atoms"] = ii["atoms"]
        info["pred"] = ii["pred"]
        save.append(info)

    dumpjson(data=save, filename=filename)


if __name__ == "__main__":
    # args = parser.parse_args(sys.argv[1:])
    model_name = args.model_name
    directory_name = args.directory_name
    file_format = args.file_format
    cutoff = args.cutoff
    max_neighbors = args.max_neighbors
    config_name = args.config_name
    output_name = args.output_name
    
    directory_path = "predict_target/label_alignn_format/poscars_for_synth_prediction"    
    poscars_dir = os.path.join(directory_path, directory_name)
    synth_preds = []
    # Iterate over all files in the poscars_for_synth_prediction directory
    for file_name in os.listdir(poscars_dir):
        file_path = os.path.join(poscars_dir, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            atoms = Atoms.from_poscar(file_path)
            out_class = get_prediction(
                model_name=model_name,
                cutoff=float(cutoff),
                max_neighbors=int(max_neighbors),
                atoms=atoms,
            )
            synth_preds.append([file_name, out_class])
        print("Predicted value:", model_name, file_name, out_class)
        
        

    csv_path = os.path.join(os.path.dirname(directory_path),f'{output_name}.csv')
    
    labels = [item[1] for item in synth_preds]
    avg_label = sum(labels) / len(labels)
    avg_label_percent = avg_label*100
    print(f"{avg_label_percent:.2f}% of the structures were predicted to be synthesizable.")
# Open the file in write mode
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['file_name', 'synth_prediction'])

        # Write each row
        for row in synth_preds:
            writer.writerow(row)
        
