
import os
import json
import time
import torch
import numpy as np
import pandas as pd
import argparse
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import schnetpack as spk
from schnetpack import transform as trn
from schnetpack.data import ASEAtomsData, AtomsDataModule

from syncotrainmp.experiment_setup import current_setup, str_to_bool
from syncotrainmp.pu_schnet.pu_learn.schnet_funcs import directory_setup, predProb
from syncotrainmp.pu_schnet.pu_learn.Datamodule4PU import DataModuleWithPred
from syncotrainmp.pu_schnet.pu_learn import int2metric


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Semi-Supervised ML for Synthesizability Prediction -- SchNet PU Step"
    )
    parser.add_argument("--experiment", default="schnet0", help="Name of the experiment and config files.")
    parser.add_argument("--ehull015", type=str_to_bool, default=False, help="Use 0.015 eV cutoff for stability.")
    parser.add_argument("--small_data", type=str_to_bool, default=False, help="Use a small dataset for testing.")
    parser.add_argument("--startIt", type=int, default=0, help="Starting iteration number.")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID to use for training.")
    return parser.parse_args()


def initialize_environment(args):
    """
    Sets the environment variable for GPU configuration based on the provided GPU ID.

    Args:
        args (argparse.Namespace): Parsed arguments containing the GPU ID to set.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


def load_configuration(args, config_path='syncotrainmp/pu_schnet/schnet_configs/pu_config_schnetpack.json'):
    """
    Loads configuration settings for the PU-SchNet model from a JSON file.

    Args:
        args (argparse.Namespace): Parsed arguments containing the starting iteration.
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary with updated start iteration.
    """
    with open(config_path, "r") as read_file:
        config = json.load(read_file)

    config["start_iter"] = args.startIt

    return config


def load_crysdf(cs):
    """
    Loads the crystal dataset for synthesizability prediction, with target properties reshaped 
    for model compatibility.

    Args:
        cs (dict): Dictionary with configuration settings including properties and target paths.

    Returns:
        pandas.DataFrame: DataFrame containing crystal structures and associated targets.
    """
    prop = cs["prop"]
    TARGET = cs["TARGET"]

    crysdf = pd.read_pickle(cs["propDFpath"])
    crysdf["targets"] = crysdf[TARGET].map(lambda target: np.array(target).flatten())
    # We need the array to have the shape (1,), hence we use flatten()
    crysdf["targets"] = crysdf.targets.map(lambda target: {prop: np.array(target)})  

    return crysdf


def get_res_dir(args, config, cs):
    """
    Constructs the directory paths for storing experiment results and checkpoints.

    Args:
        args (argparse.Namespace): Parsed arguments with experiment settings.
        config (dict): Configuration dictionary with directory paths.
        cs (dict): Configuration settings including data prefix.

    Returns:
        tuple: Paths for result directory and save directory.
    """
    save_dir = os.path.join(config["schnetDirectory"], f'PUOutput_{cs["dataPrefix"]}{args.experiment}')
    if args.ehull015:
        save_dir = os.path.join(config["schnetDirectory"], f'PUehull015_{args.experiment}')
    res_dir = os.path.join(save_dir,'res_df')

    return res_dir, save_dir


def create_model(prop, cutoff=5, n_rbf=30, n_atom_basis=64, n_filters=64, n_interactions=3, lr=1e-3):
    """
    Creates and initializes the PU-SchNet model for synthesizability prediction.

    Args:
        prop (str): Property to be predicted.
        cutoff (float): Cutoff distance for interatomic interactions.
        n_rbf (int): Number of radial basis functions.
        n_atom_basis (int): Size of atom embedding space.
        n_filters (int): Number of filters in interaction blocks.
        n_interactions (int): Number of interaction blocks.
        lr (float): Learning rate for the optimizer.

    Returns:
        spk.task.AtomisticTask: Configured model for training and prediction.
    """
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_filters=n_filters, n_interactions=n_interactions, radial_basis=radial_basis,    
        cutoff_fn = spk.nn.CosineCutoff(cutoff),
    )

    pred_prop = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=prop)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_prop],
    )

    output_prop = int2metric.ModelOutput4ACC(
        name=prop,
        loss_fn=torch.nn.BCEWithLogitsLoss(), 
        loss_weight=1.,
        metrics={
            "Accuracy": torchmetrics.Accuracy("binary"),
            "recalll" : torchmetrics.Recall  ("binary")
        }
    )

    scheduler = {
        'scheduler_cls':torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_args':{
            "mode"     : "max", #mode is min for loss, max for merit
            "factor"   : 0.5,
            "patience" : 15,
            "threshold": 0.01,
            "min_lr"   : 1e-6
        },
        'scheduler_monitor': f"val_{prop}_Accuracy"
    }

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_prop],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": lr},
        scheduler_monitor=scheduler['scheduler_monitor'],
        scheduler_cls=scheduler['scheduler_cls'],
        scheduler_args=scheduler['scheduler_args'],
    )

    return task


def create_trainer(config, cs, save_dir, save_it_dir):
    """
    Sets up the PyTorch Lightning Trainer with early stopping, model checkpointing, and TensorBoard logging.

    Args:
        config (dict): Configuration settings for training.
        cs (dict): Configuration settings including property to be predicted.
        save_dir (str): Directory for saving training logs and checkpoints.
        save_it_dir (str): Directory for saving iteration-specific results.

    Returns:
        pl.Trainer: Configured PyTorch Lightning Trainer.
    """
    # This doesn't work when no test data is given.
    early_stopping = EarlyStopping(
        verbose=2,
        mode= 'max', #min for loss, max for merit.
        monitor=f"val_{cs['prop']}_Accuracy",  #if it works, also change in ModelCheckpoint?
        min_delta=0.02,
        patience=30,
    )
    model_checkpoint = spk.train.ModelCheckpoint(
        inference_path=os.path.join(save_it_dir, "best_inference_model"),
        save_top_k=1,
        monitor=f"val_{cs['prop']}_Accuracy"
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=1,
        auto_select_gpus = True,
        strategy=None,
        precision=16,
        callbacks=[early_stopping, model_checkpoint],
        logger=logger,
        default_root_dir=save_it_dir,
        max_epochs=config["epoch_num"],
    )

    return trainer


def get_test_train_data(it, config, cs, crysdf, trainDataPath, testDatapath, splitFilestring, split_id_dir_path, res_dir, save_it_dir, bestModelPath, cutoff):
    """
    Prepares the training and test datasets for each iteration, including data shuffling and 
    splitting into training, validation, and test sets.

    Args:
        it (int): Current iteration number.
        config (dict): Configuration settings for training.
        cs (dict): Configuration settings for the dataset.
        crysdf (pandas.DataFrame): DataFrame containing crystal structures and targets.
        trainDataPath (str): Path to save training dataset.
        testDatapath (str): Path to save test dataset.
        splitFilestring (str): String representing the split configuration.
        split_id_dir_path (str): Path to the split ID files.
        res_dir (str): Path to save results.
        save_it_dir (str): Directory for saving iteration-specific files.
        bestModelPath (str): Path to save the best model.
        cutoff (float): Cutoff distance for neighbor interactions.

    Returns:
        tuple: Data loaders for training (crysData) and testing (crysTest).
    """
    prop = cs["prop"]
    TARGET = cs["TARGET"]

    train_id_path = os.path.join(split_id_dir_path, f'train_id_{it}.txt')
    test_id_path  = os.path.join(split_id_dir_path, f'test_id_{it}.txt')

    with open(train_id_path, "r") as f:
        id_val_train = [int(line.strip()) for line in f]

    with open(test_id_path, "r") as f2:
        id_test = [int(line.strip()) for line in f2]

    it_traindf = crysdf.loc[id_val_train]
    it_testdf  = crysdf.loc[id_test]
    
    valLength       = int(len(it_traindf)*.1)-5
    trainLength     = int(len(it_traindf)*.9)-5 
    innerTestLength = len(it_traindf)-(trainLength+valLength) # Fatal error without internal test set.
    
    positivePredictionLength = it_testdf[TARGET].sum()
    unlabeledPredictionLength = len(it_testdf)-positivePredictionLength
    testLength = len(it_testdf)

    it_traindf = it_traindf.sample(frac=1,random_state=it, ignore_index=True) # Shuffling for each iteration.
    it_traindf.reset_index(drop=True, inplace=True)
    it_testdf.reset_index(drop=True, inplace=True)
    
    if it==1:
        print(f"The #training data is {trainLength}, #validation data {valLength} and #internal test data {innerTestLength}.")
        print(f"The total number of test-set (predictions) is {testLength}, out of which {unlabeledPredictionLength} are unlabeled and {positivePredictionLength} are labeled positive.")

    class_dataset = ASEAtomsData.create(trainDataPath, 
                        distance_unit='Ang',
                        property_unit_dict={prop:int(1)} # The unit is int(1); aka unitless.
    )
    print('adding systems to dataset')
    class_dataset.add_systems(np.array(it_traindf.targets), np.array(it_traindf.atoms))
    print('creating data module')
    crysData = AtomsDataModule(datapath=trainDataPath,
        batch_size=config["batch_size"],
        num_train=trainLength,
        num_val=valLength,
        transforms=[
            trn.ASENeighborList(cutoff=float(cutoff)),
            trn.CastTo32(), 
        ],
        property_units={prop:int(1)},
        num_workers=4,    
        split_file = splitFilestring, 
        pin_memory=True, # set to false, when not using a GPU
        load_properties=[prop], 
    )
    
    crysData.prepare_data()
    crysData.setup()
    
    splitFilestringTest = directory_setup(res_dir = res_dir, 
                                          dataPath = testDatapath,save_dir = save_it_dir, 
                                          bestModelPath= bestModelPath,
    ) # iteration_num=it)

    test_dataset = ASEAtomsData.create(
        testDatapath, 
        distance_unit='Ang',
        property_unit_dict={prop:int(1)},
    )
    print('adding systems to the test dataset')
    test_dataset.add_systems(np.array(it_testdf.targets), np.array(it_testdf.atoms))  

    print('creating data module')

    crysTest = DataModuleWithPred(
        datapath=testDatapath,
        batch_size=config["batch_size"],
        num_train=0,
        num_val=0, 
        num_test=len(it_testdf),
        transforms=[
            trn.ASENeighborList(cutoff=float(cutoff)),
            trn.CastTo32(), 
        ],
        property_units={prop:int(1)},
        num_workers=4,
        split_file = splitFilestringTest, 
        pin_memory=True, # set to false, when not using a GPU
        load_properties=[prop], 
    )

    crysTest.prepare_data()
    crysTest.setup("test")

    return crysData, crysTest, it_traindf, it_testdf


def run_iteration(it, iteration_results, args, config, cs, crysdf, start_time, cutoff = 5):
    """
    Runs a single iteration of the PU-SchNet training and prediction, including data loading, 
    training, and prediction on the test set.

    Args:
        it (int): Current iteration number.
        args (argparse.Namespace): Parsed arguments with experiment settings.
        config (dict): Configuration dictionary.
        cs (dict): Configuration settings for the dataset.
        crysdf (pandas.DataFrame): DataFrame containing crystal structures and targets.
        start_time (float): Start time for tracking total elapsed time.
        cutoff (float): Cutoff distance for interatomic interactions.

    Returns:
        pandas.DataFrame: DataFrame containing the iteration results with predictions.
    """
    prop = cs["prop"]
    TARGET = cs["TARGET"]
    data_prefix = cs["dataPrefix"]

    split_id_dir = f"{data_prefix}{TARGET}_{prop}"
    split_id_dir_path = os.path.join(config["data_dir"], split_id_dir)        

    res_dir, save_dir = get_res_dir(args, config, cs)

    print('we started iteration {}'.format(it))
    np.random.seed(it) 

    save_it_dir = os.path.join(save_dir, f'iter_{it}')
    dataDir = os.path.join(save_it_dir,"schnetDatabases")

    testDatapath  = os.path.join(dataDir,f"{data_prefix}{args.experiment}_{prop}_test_dataset.db")
    trainDataPath = os.path.join(dataDir,f"{data_prefix}{args.experiment}_{prop}_train_dataset.db")
    bestModelPath = os.path.join(save_it_dir,'best_inference_model')

    splitFilestring = directory_setup(
        res_dir = res_dir, 
        dataPath = trainDataPath,
        save_dir = save_it_dir,
        bestModelPath= bestModelPath,
    )# iteration_num=it)

    crysData, crysTest, _, it_testdf = get_test_train_data(
        it, config, cs, crysdf, trainDataPath, testDatapath,
        splitFilestring, split_id_dir_path, res_dir, save_it_dir, bestModelPath, cutoff)

    means, stddevs = crysData.get_stats(
        prop, divide_by_atoms=True, remove_atomref=True)

    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())

    model = create_model(prop, cutoff=cutoff)

    trainer = create_trainer(config, cs, save_dir, save_it_dir)
    trainer.fit(model, datamodule=crysData)

    predictions = trainer.predict(
        model=model,
        dataloaders= crysTest.predict_dataloader(),
        return_predictions=True
    )

    results = []
    for batch in predictions:    
        for datum in batch[prop]:
            results = results+[predProb(datum.float())]
            
    res_list = []
    for i, datum in enumerate(crysTest.test_dataset):
        groundTruth = int(datum[prop].detach())
        ind = int(datum['_idx'])
        res_list.append([ind,groundTruth,results[i]])

    resdf = pd.DataFrame(res_list, columns=['testIndex','GT','pred_'+str(it)])  #GT is a duplicate
    resdf = resdf.set_index('testIndex').sort_index() 
        
    it_testdf = it_testdf[['material_id']] 
    it_testdf = it_testdf.merge(resdf['pred_'+str(it)], left_index=True, right_index=True)

    iteration_results = iteration_results.merge(
        it_testdf,
        left_on='material_id',
        right_on='material_id',
        how='outer'
    )
    
    try:
        os.remove(testDatapath)
        os.remove(trainDataPath)
        print("File removed successfully!")
    except Exception as e:
        print("An error occurred:", str(e))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("===the {}th iteration is done.".format(it))

    save_result(args, config, cs, iteration_results, tmp=True)

    elapsed_time = time.time() - start_time
    remaining_iterations = config["num_iter"] - it - 1
    time_per_iteration = elapsed_time / (it - config["start_iter"] + 1)
    estimated_remaining_time = remaining_iterations * time_per_iteration
    remaining_days = int(estimated_remaining_time // (24 * 3600))
    remaining_hours = int((estimated_remaining_time % (24 * 3600)) // 3600)

    time_log_path = os.path.join('time_logs',f'schnet_remaining_time_{data_prefix}{args.experiment}_{prop}.txt')
    with open(time_log_path, 'w') as file:
        file.write(f"Iterations completed: {it - config['start_iter']}\n")
        file.write(f"Iterations remaining: {remaining_iterations}\n")
        file.write(f"Estimated remaining time: {remaining_days} days, {remaining_hours} hours\n")

    print(f"Iteration {it} completed. Remaining time: {remaining_days} days, {remaining_hours} hours")   

    return iteration_results


def save_result(args, config, cs, iteration_results, tmp=False):
    """
    Saves the results of the current iteration to a file.

    Args:
        args (argparse.Namespace): Parsed arguments with experiment settings.
        config (dict): Configuration dictionary.
        cs (dict): Configuration settings for the dataset.
        iteration_results (pandas.DataFrame): DataFrame containing predictions and ground truth.
        tmp (bool): Whether to save as a temporary file.
    """
    res_dir, _ = get_res_dir(args, config, cs)

    res_df_fileName = f"{cs['dataPrefix']}{args.experiment}_{str(config['start_iter'])}_{str(config['num_iter'])}ep{str(config['epoch_num'])}"

    if tmp:
        res_df_fileName = res_df_fileName+'tmp'

    iteration_results.to_pickle(os.path.join(res_dir, res_df_fileName))

    print(f"Saved PU-SchNet results to: {res_df_fileName}")


def save_time_log(args, config, cs, start_time):
    """
    Logs the total time taken for the PU-SchNet training and testing iterations.

    Args:
        args (argparse.Namespace): Parsed arguments with experiment settings.
        config (dict): Configuration dictionary.
        cs (dict): Configuration settings for the dataset.
        start_time (float): Start time for tracking total elapsed time.
    """
    elapsed_time  = time.time() - start_time
    elapsed_days  = int(elapsed_time // (24 * 3600))
    elapsed_hours = int((elapsed_time % (24 * 3600)) // 3600)

    time_log_path = os.path.join('time_logs',f'schnet_remaining_time_{cs["dataPrefix"]}{args.experiment}_{cs["prop"]}.txt')

    with open(time_log_path, 'w') as file:
        file.write(f"Iterations completed: {config['num_iter'] - config['start_iter']}\n")
        file.write(f"Total time taken: {elapsed_days} days, {elapsed_hours} hours\n")

    print(f"PU Learning completed. Total time taken: {elapsed_days} days, {elapsed_hours} hours")


def main():
    """
    Main function for running PU-SchNet training and prediction. Parses arguments, loads 
    configurations, initializes data, and runs iterative training and testing.
    """
    args   = parse_arguments()
    config = load_configuration(args)
    cs     = current_setup(small_data=args.small_data, experiment=args.experiment, ehull015=args.ehull015)
    crysdf = load_crysdf(cs)

    initialize_environment(args)

    # Initialize result df
    if config["start_iter"] != 0:
        res_dir, _        = get_res_dir(args, config, cs)
        iteration_results = pd.read_pickle(os.path.join(res_dir, f"{cs['dataPrefix']}{args.experiment}_0_{str(config['num_iter'])}ep{str(config['epoch_num'])}"+"tmp"))
    else:
        iteration_results = crysdf[["material_id", cs["prop"], cs["TARGET"]]]
        iteration_results = iteration_results.loc[:, ~iteration_results.columns.duplicated()]

    start_time = time.time()
    for it in range(config["start_iter"], config["num_iter"]):
        iteration_results = run_iteration(it, iteration_results, args, config, cs, crysdf, start_time)
        print(f"Iteration {it} completed.")

    save_result  (args, config, cs, iteration_results)
    save_time_log(args, config, cs, start_time)

if __name__ == "__main__":
    main()
