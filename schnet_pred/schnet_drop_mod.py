# schnet_drop_mod.py
import schnetpack as spk
import torch.nn as nn
# from train_schnet import SchNetWithDropout #need to import/define this class to read the model

class SchNetWithDropout(nn.Module):
    def __init__(self, n_atom_basis, n_filters, n_interactions, n_rbf, cutoff, dropout_rate):
        super(SchNetWithDropout, self).__init__()
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_filters=n_filters,
            n_interactions=n_interactions,
            radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff)
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs):
        x = self.schnet(inputs)
        x['scalar_representation'] = self.dropout(x['scalar_representation'])  # Apply dropout to the relevant tensor
        return x