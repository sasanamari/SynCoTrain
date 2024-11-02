# SynCoTrain 
Co-training enhanced PU-learning for crystal synthesizability prediction.


## Introduction
SynthCoTrain is a materials-informatics package which predicts the synthesizability of crystals. The nature of the problem is a semi-supervised classification, in which we access only to positively labeled and unlabeled data points. SynCoTrain does this classification task by combining two semi-supervised classification methods: **Positive and Unlabeled (PU) Learning** and **Co-training**. The classifiers used in this package are [ALIGNN](https://github.com/usnistgov/alignn) and [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack).

<!-- ![cotraining scheme](figures/cotraining_scheme.jpg) -->
<div style="text-align:center">
<img src="figures/cotraining_scheme_new.png" alt="cotraining scheme" width="550" height="350">
</div>

The final model achieves a notable true-positive rate of 96% for the experimentally synthesized test-set and predicts that 29% of the theoretical crystals are synthesizable. These results go beyond the scope of thermodynamic stability analysis alone. This work carries significant implications, including the filtration of structural predictions from high-throughput simulations to identify synthesizable candidates.


## Installation
It is recommended to create a virtual environment with mamba and miniforge to install the different packages easily. Start by installing mamba according to the instructions [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

Start by cloning this repository in your preferred path as shown below:
```bash
cd /path/to/parent/directory
git clone git@github.com:BAMeScience/SynCoTrainMP.git
```
Next, navigate to the cloned directory. You can create the appropriate mamba environment there based on the `sync.yml` file:
```bash
cd SynCoTrain
mamba env create -f condaEnvs/sync.yml
mamba activate sync
```
This might take a while, as all the required packages are being installed. Please note that you may need to change ther exact version of dgl and cudatoolkit based on your current setup. You can check your current cuda version using the `nvidia-smi` command. Then, you can search for a compatible dgl with cuda using the command `mamba search dgl --channel conda-forge`. Pick a version of dgl earlier than 2.0.0 which is compatible with your cudatoolkit.

Once the packages are installed, you may activate the `sync` conda environment and install this repository with the following commands:
```bash
pip install -e .
```
## Predicting Synthesizability of Oxides
You don't need to train the model from scratch if you are only interested in predicting synthesizability. The current version of SynCoTrain has been trained to predict the synthesizability of oxide crystals. We use the SchNet as our classifier.

In order to predict synthesizability results for your own data, place a pickled DataFrame inside the `schnet_pred/data` directory, e.g. `schnet_pred/data/<your_crsytal_data>.pkl`. Next, you can feed this DataFrame as the input to the model:
```bash
python schnet_pred/predict_schnet.py --input_file <your_crsytal_data>
```
The result will be saved in `schnet_pred/results/<your_crsytal_data>_predictions.csv`.


## Auxiliary Experiments

This package includes two auxiliary experiments to further evaluate model performance:

1. **Reduced Data Experiment**: Runs the regular experiment on only 5% of the available data. This is useful for testing the code workflow without long computation times. Note that the model’s performance may decrease due to the reduced data.

2. **Stability Classification Experiment**: Classifies the stability of crystals based on their energy above hull using the same PU Learning and Co-training approach. This experiment differs from typical Positive and Unlabeled (PU) Learning because we have correct labels for the “unlabeled” class. Since stability correlates with synthesizability, this experiment provides a proxy for assessing the quality of the main experiment by comparing true-positive rates from PU Learning.

## Training the Models

To replicate the results from this library, follow the steps for running each PU experiment. Three predefined experiments are available for each base classifier, with each experiment comprising 60 iterations of PU learning.

1. **Run Base Experiment**: Begin by running the base experiment for each model. Afterward, each model is trained on pseudo-labels provided by the other model to improve co-training.

> **Note**: These experiments are computationally intensive. For example, on an NVIDIA A100 80GB PCIe GPU, each experiment takes approximately one week. If running on full data, consider using the `nohup` command to allow the process to run in the background.

> **Recommendation**: Avoid running multiple experiments simultaneously on the same GPU to prevent memory overflow, which could crash the experiment.

### Step 1: Initial Model Training (Iteration "0")

Before co-training, train each model separately on the PU data. For example, to train the SchNet model:
```bash
mamba activate sync
syncotrainmp_data_selection --experiment schnet0
nohup syncotrainmp_pu_schnet --experiment schnet0 --gpu_id 0 > nohups/schnet0_synth_gpu0.log &
```
In case you have access to multiple GPUs, the `--gpu_id` parameter can be changed accordingly. Similarly for the ALIGNN experiment we have:
```bash
mamba activate sync
syncotrainmp_data_selection --experiment alignn0
nohup syncotrainmp_pu_alignn --experiment alignn0 --gpu_id 0 > nohups/alignn0_synth_gpu0.log &
```

### Step 2: Analyze Results and Generate Labels

After each experiment is concluded, the data needs to be analyzed to produce the relevant labels for the next step of co-training. The code for the analysis of results of SchNetPack is
```bash
python pu_schnet/schnet_pu_analysis.py --experiment schnet0 
```
and for ALIGNN:
```bash
python pu_alignn/alignn_pu_analysis.py --experiment alignn0 
```

### Subsequent Steps

From this point, it matters that the experiments are executed in their proper order. Before each PU experiment, the relevant data selection needs to be performed. After each PU experiment, the analysis of the results are needed to produce the labels for the next iteration. The commands to run these experiments can be found on `synth_commands.txt`.

The correct order of running the experiments starting from alignn0 is:
```
alignn0 > coSchnet1 > coAlignn2 > coSchnet3
```
and for the other view, starting from schnet0:
```
schnet0 > coAlignn1 > coSchnet2 > coAlignn3
```

## Stability experiments
The auxiliary stability experiments can be run with almost the same commands, except for an extra `--ehull015 True` flag. The relavant commands are stored in `stability_commands.txt`.

## Training the predictor
After the final round of predictions, the predictions are averaged and the classification threshold is applied to produce training labels. Next, the data is augmented to improve model generalization.
```bash
python schnet_pred/label_by_average.py
python schnet_pred/data_augment.py
```
Now our training data is ready. We can train a SchNet classifier on these augmented data.
```bash
python schnet_pred/train_schnet.py
```
After the training is complete, we can predict the results for the test-set:
```bash
python schnet_pred/predict_schnet.py
```





