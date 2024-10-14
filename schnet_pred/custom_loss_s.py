import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight  # pos_weight for the positive class
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the binary cross-entropy loss with logits and pos_weight
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # Get the probabilities of the correct class
        probs = torch.exp(-ce_loss)

        # Apply the focal loss modulation: (1 - p_t)^gamma * CE loss
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        # Apply the reduction (mean or sum)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
