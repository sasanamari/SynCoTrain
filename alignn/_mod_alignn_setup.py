#!/usr/bin/env python
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

"""Module to train for a folder with formatted dataset."""
import csv
import sys
import time
from jarvis.core.atoms import Atoms
### from alignn.data import get_train_val_loaders
from PU_data_handling import get_train_val_loaders_PU
### from alignn.train import train_dgl
from myTrain import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
import torch 
import pandas as pd
from synth.data_scripts.crystal_structure_conversion import ase_to_jarvis

# %%
# import subprocess as sp
# %%
# parser = argparse.ArgumentParser(
#     description="Atomistic Line Graph Neural Network"
# )
# parser.add_argument(
#     "--root_dir",
#     default="./alignn/sample_synth",
#     help="Folder with id_props.csv, structure files",
# )
# parser.add_argument(
#     "--config_name",
#     # default="alignn/examples/sample_data/config_example.json",
#     # default="alignn/config_example.json",
#     default="./alignn/config_example_class.json",
#     help="Name of the config file",
# )

# parser.add_argument(
#     "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
# )

# parser.add_argument(
#     "--keep_data_order",
#     default=False,
#     help="Whether to randomly shuffle samples, True/False",
# )

# parser.add_argument(
#     "--classification_threshold",
#     default=None,
#     help="Floating point threshold for converting into 0/1 class"
#     + ", use only for classification tasks",
# )

# parser.add_argument(
#     "--batch_size", default=None, help="Batch size, generally 64"
# )

# parser.add_argument(
#     "--epochs", default=None, help="Number of epochs, generally 300"
# )

# parser.add_argument(
#     # "--output_dir", default="./",
#     "--output_dir", default=None, #set to none, to be read from config
#     help="Folder to save outputs",
# )

# # parser.add_argument(
# #     "--shuffle_seed", default="42",
# #     help="seed to shuffle data before splitting train-test splitting",
# # )

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
    reverse_label=False,
    ehull_test = False,
    cotraining = False,
):
    # """Train for a folder.""" not a folder anymore.
    # config_dat=os.path.join(root_dir,config_name)
    # if reverse_label:
    #     id_prop_dat = os.path.join(root_dir, "synth_id_prop_rev.csv")
    # elif cotraining:
    #     id_prop_dat = os.path.join(root_dir, "synth_id_prop_cotraining.csv")
    # elif ehull_test:
    #     id_prop_dat = os.path.join(root_dir, "ehull_test.csv")
    # else:
    #     id_prop_dat = os.path.join(root_dir, "synth_id_prop.csv")
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check this expection here", exp)

    config.keep_data_order = keep_data_order
    # trying to iterate
    # config.output_dir = config.output_dir+shuffle_seed
    # config.random_seed = shuffle_seed
    
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
        
    data_df = pd.read_pickle(os.path.join(root_dir, 'synthDF'))
    data_df["TARGET"] = data_df["synth"] #later modify for cotraining
    # with open(id_prop_dat, "r") as f:
    #     reader = csv.reader(f)
    #     data = [row for row in reader]
        
    # tmptargets = [float(d[1]) for d in data]
    # data_size = {"positive_data_size":int(sum(tmptargets)),
    #          "unlabeled_data_size":int(len(tmptargets)-sum(tmptargets))}
    # del tmptargets
    data_size = {"positive_data_size":int(sum(data_df["TARGET"])),
             "unlabeled_data_size":int(len(data_df)-sum(data_df["TARGET"]))}
                     
    # random.Random(config.random_seed).shuffle(data)
    # # shuffles data before test/train split

    dataset = []
    n_outputs = []
    multioutput = False #I keep the multioutput in case of future inspiration.
    lists_length_equal = True
    # for i in data:
    for _, row in data_df.iterrows():
        info = {}
        if file_format == "ase_atom":
            info['atoms']= ase_to_jarvis(row["atoms"]).to_dict()
            # The library is made for dictionary form.
        else:
            print('Undersired file format!')
            sys.exit()
        info['jid']=row['material_id']
        tmp=row['TARGET']
        # file_name = i[0]
        # file_path = os.path.join(root_dir, file_name)
        # if file_format == "poscar":
        #     atoms = Atoms.from_poscar(file_path)
        # elif file_format == "cif":
        #     atoms = Atoms.from_cif(file_path)
        # elif file_format == "xyz":
        #     # Note using 500 angstrom as box size
        #     atoms = Atoms.from_xyz(file_path, box_size=500)
        # elif file_format == "pdb":
        #     # Note using 500 angstrom as box size
        #     # Recommended install pytraj
        #     # conda install -c ambermd pytraj
        #     atoms = Atoms.from_pdb(file_path, max_lat=500)
        # else:
        #     raise NotImplementedError(
        #         "File format not implemented", file_format
        #     )

        # info["atoms"] = atoms.to_dict()
        # info["jid"] = file_name

        # tmp = [float(j) for j in i[1:]]  # float(i[1])
        # if len(tmp) == 1:
        #     tmp = tmp[0]
        # else:
        #     multioutput = True
        if isinstance(tmp, list): #in case of future multiouput.
            if len(tmp)>1:
                multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
    
    # data_size = loadjson("/home/samariam/projects/chemheuristics/alignn/dataSize.json")   
    
    print(data_size["positive_data_size"])     
    print(data_size["unlabeled_data_size"])     
    # all_targerts = [datum['target'] for datum in dataset]
        
    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

    # print ('n_outputs',n_outputs[0])
    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.model.output_features = 1
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders_PU(
    # ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        split_seed=config.random_seed,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        data_portion_dict= data_size,
    )
    t1 = time.time()
    if reverse_label:
        from alignn.models.alignn import ALIGNN
        model = ALIGNN(config=config.model)
        model_path = (
    "/home/samariam/projects/chemheuristics/alignn/pretrain_test/checkpoint_177.pt"
                    )
        model.load_state_dict(torch.load(model_path, map_location=device)["model"])
        
    else:
        model=None
    
    train_dgl(
        config,
        model = model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )
    t2 = time.time()
    print("Time taken (s):", t2 - t1)

    # train_data = get_torch_dataset(


# if __name__ == "__main__":
#     args = parser.parse_args(sys.argv[1:])
#     train_for_folder(
#         root_dir=args.root_dir,
#         config_name=args.config_name,
#         keep_data_order=args.keep_data_order,
#         classification_threshold=args.classification_threshold,
#         output_dir=args.output_dir,
#         batch_size=(args.batch_size),
#         epochs=(args.epochs),
#         file_format=(args.file_format),
#     )
