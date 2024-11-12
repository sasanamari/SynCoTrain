#!/usr/bin/env python
"""Module to train for a folder with formatted dataset."""
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# import csv
# import sys
# import time
# from jarvis.core.atoms import Atoms
# ### from alignn.data import get_train_val_loaders
# from pu_alignn.pu_learn.PU_data_handling import get_train_val_loaders_PU
# ### from alignn.train import train_dgl
# from pu_alignn.pu_learn.Train_stop import train_dgl
# from alignn.config import TrainingConfig
# from jarvis.db.jsonutils import loadjson
# import argparse
# import torch 
# import numpy as np
# from experiment_setup import current_setup
# # %%
# device = "cpu"
# if torch.cuda.is_available():
#     device = torch.device("cuda")


def train_for_folder(gpu_id,
    experiment,
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
    small_data = False,
    ehull015 = False,
    train_id_path = 'data/clean_data/alignn0/train_id_1.txt',
    test_id_path ='data/clean_data/alignn0/test_id_1.txt',
):
    """Train for a folder."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import csv
    import time
    import torch
    import numpy as np
    from jarvis.core.atoms import Atoms
    from syncotrainmp.pu_alignn.pu_learn.PU_data_handling import get_train_val_loaders_PU
    from syncotrainmp.pu_alignn.pu_learn.Train_stop import train_dgl
    from alignn.config import TrainingConfig
    from jarvis.db.jsonutils import loadjson

    from syncotrainmp.experiment_setup import current_setup
    
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")    
    # if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # config_dat=os.path.join(root_dir,config_name)
    cs = current_setup(small_data=small_data, experiment=experiment, ehull015 = ehull015)
    # propDFpath = cs["propDFpath"]
    # result_dir = cs["result_dir"]
    prop = cs["prop"]
    TARGET = cs["TARGET"]
    data_prefix = cs["dataPrefix"]
    
    print(f'You have selected {TARGET} as your reference column!!!!!!!!!')

    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check this expection here", exp)
            
        
    config.keep_data_order = keep_data_order
    if small_data:
        config.epochs = int(config.epochs/3)

    
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        if small_data:
            epochs = int(epochs/3)
            config.epochs = int(epochs)
        else:
            config.epochs = int(epochs)
            
    with open(os.path.join(os.path.dirname(train_id_path), 
                           'experimentalDataSize.txt')) as eds:
        experimentalDataSize = int(float(eds.read().strip()))
    
    data_csv_path = os.path.join(root_dir, 
                    f"{data_prefix}{prop}_id_from_{TARGET}.csv")
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        
    tmptargets = [float(d[1]) for d in data]
    
    positiveSize = int(np.nansum(tmptargets))
    
    data_portion_dict = {"positive_data_size":positiveSize,
            "unlabeled_data_size":int(len(tmptargets)-positiveSize),
            "experimentalDataSize": experimentalDataSize}

    del tmptargets
        
    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True
    for i in data:
        info = {}
        file_name = i[0]
        # file_path = os.path.join(root_dir, file_name)  
        file_path = os.path.join(root_dir,f"{data_prefix}atomistic_{prop}_{experiment}", file_name)
        
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
    
    
    print("The size of positive labeled data is ",
          data_portion_dict["positive_data_size"])     
    print("The size of experimental data is ",
          data_portion_dict["experimentalDataSize"])     
    print("The size of unlabeled data is ",
          data_portion_dict["unlabeled_data_size"])     
        
    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

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
        val_ratio=config.val_ratio,
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
        train_id_path = train_id_path,
        test_id_path =test_id_path
    )
    t1 = time.time()
    # if reverse_label:
    #     from alignn.models.alignn import ALIGNN
    #     model = ALIGNN(config=config.model)
    #     model_path = (
    # "alignn/pretrain_test/checkpoint_177.pt"
    #                 )
    #     model.load_state_dict(torch.load(model_path, map_location=device)["model"])
        
    # else:
    model=None
    print(f"The length of train data is {len(train_loader.dataset)},")
    print(f"The length of val data is {len(val_loader.dataset)},")
    print(f"The length of test data is {len(test_loader.dataset)}.")
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


# %%