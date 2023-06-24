from typing import Optional, Dict, List
from torchmetrics import Metric
import torch
from torch import nn

# ###This has been copied from spk.task.ModelOutput, except for the commented line near the end.
class ModelOutput4ACC(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.metrics = nn.ModuleDict(metrics)
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def calculate_metrics(self, pred, target):
        prob = nn.Sigmoid()
        pred_prob = prob(pred[self.name])
        metrics = {
            metric_name: metric(pred_prob, target[self.target_property].type(torch.int))
            # metric_name: metric(pred[self.name], target[self.target_property])
            for metric_name, metric in self.metrics.items()
        }
        return metrics