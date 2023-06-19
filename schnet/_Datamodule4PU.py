import logging
import os
import shutil
from copy import copy
from typing import Optional, List, Dict, Tuple, Union


import numpy as np
import fasteners

import pytorch_lightning as pl
from pytorch_lightning.accelerators import GPUAccelerator
import torch

from schnetpack.data import (
    AtomsDataFormat,
    resolve_format,
    load_dataset,
    BaseAtomsData,
    AtomsLoader,
    calculate_stats,
    SplittingStrategy,
    RandomSplit,
   AtomsDataModule,
)

__all__ = ["DataModuleWithPred", "AtomsDataModuleError"]


class AtomsDataModuleError(Exception):
    pass

class DataModuleWithPred(AtomsDataModule):
    """Adding predict_dataloader method to AtomsDataModule.

    Args:
        AtomsDataModule (_type_): _description_
    """
    
    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = None,
        # shuffle: bool = True
    ):
    

        super().__init__(
        datapath,
        batch_size,
        num_train,
        num_val,
        num_test,
        split_file,
        format,
        load_properties,
        val_batch_size,
        test_batch_size,
        transforms,
        train_transforms,
        val_transforms,
        test_transforms,
        num_workers,
        num_val_workers,
        num_test_workers,
        property_units,
        distance_unit,
        data_workdir,
        cleanup_workdir_stage,
        splitting,
        pin_memory,
        
        )

    def predict_dataloader(self) -> AtomsLoader:
                return AtomsLoader(
                    self.test_dataset,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_test_workers,
                    shuffle=False,
                    pin_memory=self._pin_memory,
            )
