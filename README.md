# SynthCoTrain 
Co-Training for Crystal Synthesizability Prediction
## Intoroduction
SynthCoTrain is a materials-informatics package which predicts the synthesizability of crystals. The nature of the problem is a semi-supervised classification, in which we access only to positively labeled and unlabeled data points. SynthCoTrain does this classification task by combining two semi-supervised classification methods: **Positive and Unlabeled (PU) Learning** and **Co-training**. The classifiers used in this package are the ALIGNN https://github.com/usnistgov/alignn and the SchNetPack https://github.com/atomistic-machine-learning/schnetpack.

<!-- ![cotraining scheme](figures/cotraining_scheme.jpg) -->
<div style="text-align:center">
<img src="figures/cotraining_scheme.jpg" alt="cotraining scheme" width="550" height="350">
</div>

The final model achieves a notable true-positive rate of nearly 95% for the experimentally synthesized test set and predicts that 17% of the theoretical crystals are synthesizable. These results go beyond the scope of thermodynamic stability analysis alone. This work carries significant implications, including the filtration of structural predictions from high-throughput simulations to identify synthesizable candidates.

## Installation
First we need conda environments. This package used two separate models, and each of them rquire their own conda environment. For running the ALIGNN model, create a conda env for ALIGNN and activate the environment:
```bash
conda create -n puAlignn python=3.9
conda activate puAlignn
```
Then, you can install the required packages for ALIGNN from the Inline code `requirements_alignn.txt`.
```bash
pip install -r requirements_alignn.txt
```
If `requirements_alignn.txt` doe not work, you can also try `alignn_pipfreee.txt` instead.
After installing the required packges, you can install SynthCoTrain by
```
cd /path/to/SynthCoTrain
pip install .
```

Now you have installed ALIGNN and SynthCoTrain in your new puAlignn env. Similarly, you can create a separate conda env for the SchNetPack and install the corresponding packages there:
```bash
conda create -n puSchnet python=3.9
conda activate puSchnet
pip install -r requirements_schnet.txt
cd /path/to/SynthCoTrain
pip install .
```



## Using the pre-trained model
## Training the models
To replicate the results of this library, you need to run the scripts made for running each PU experiment. There are three experiments for each of the models, with pre-defined data handling. Each experimment consists of 100 iterations of PU learning.

First, the base experiment is run with each model. Next, each model can be trained on the additional psuedo-labels provided by the other model. 

Please note that these experiments are rather long. Using a NVIDIA A100 80GB PCIe GPU, each experiment took an average of one week to conclude. So, for the full-data experiment, you may want to use the `nohup` command, as shown later.

It is recommended not to run simultanous experiments on the same gpu, since you run the risk of overflowing the gpu memory and crashing the experiment mid-way.
### ALIGNN
Before each experiment with ALIGNN, the the suitable data format and labels need to be produced. The `preparing_data_byFile.py` provides the correct data format. For the base experiment run the following command:
```
python alignn/preparing_data_byFile.py
```
Then, the base experiment with ALIGNN can be excuted with this command:
```
conda activate puAlignn
python alignn/alignn_pu_learning.py --experiment alignn0
```
If you want to use `nohup`, it could be done as
```
conda activate puAlignn
nohup python alignn/alignn_pu_learning.py --experiment alignn0 > alignn0.log &
```
After the base experiments are done for both models, you can run the first cotraining experiment for ALIGNN called `coAlSch1`. You need to provide the new labels for ALIGNN:
```
python alignn/preparing_data_byFile.py --experiment coAlSch1
```
Then, `coAlSch1` can be executed by
```
nohup python alignn/alignn_pu_learning.py --experiment coAlSch1 > coAlSch1.log &
```
After the first cotraining of SchNet `coSchAl1`, the second and final cotraining for ALIGNN can be executed. Again, first prepare the new labels, then run the experiment:
```
python alignn/preparing_data_byFile.py --experiment coAlSch2
nohup python alignn/alignn_pu_learning.py --experiment coAlSch2 > coAlSch2.log &
```
### SchNetPack
The experiments which use the SchNetPack can be executed similar to those using ALIGNN. Here, there is no need to provide new labels before each experiment, as the script can directly read the pickled data. Do not forget to change conda environments as you switch between each model.
The base experiment with the SchNetPack can be executed by:
```
conda activate puSchnet
python schnet/schnet_pu_learning.py --experiment schnet0
```
If you want to use `nohup`, it could be done as
```
conda activate puSchnet
nohup python schnet/schnet_pu_learning.py --experiment schnet0 > schnet0.log &
```
Just like ALIGNN, you need to execute the corrsponding label source experiment on the other classifier before you run the cotraining experiments. Once you have, they can be executed by

```
nohup python alignn/schnet_pu_learning.py --experiment coSchAl1 > coSchAl1.log &
```
and
```
nohup python alignn/schnet_pu_learning.py --experiment coSchAl2 > coSchAl2.log &
```

## Data


