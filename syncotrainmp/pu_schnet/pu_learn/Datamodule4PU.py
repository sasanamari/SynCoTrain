import logging
import os
from typing import Tuple
import numpy as np
import fasteners
import torch
from schnetpack.data import (
    AtomsLoader,
    calculate_stats,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_partitions(self) -> Tuple[list, list, list]:
        # I commented out the check for training partition, as this module is only used for testing.
        lock = fasteners.InterProcessLock("splitting.lock")

        with lock:
            self._log_with_rank("Enter splitting lock")

            if self.split_file is not None and os.path.exists(self.split_file):
                self._log_with_rank("Load split")

                S = np.load(self.split_file)
                self.train_idx = S["train_idx"].tolist()
                self.val_idx = S["val_idx"].tolist()
                self.test_idx = S["test_idx"].tolist()
                if self.num_train and self.num_train != len(self.train_idx):
                    logging.warning(
                        f"Split file was given, but `num_train ({self.num_train})"
                        + f" != len(train_idx)` ({len(self.train_idx)})!"
                    )
                if self.num_val and self.num_val != len(self.val_idx):
                    logging.warning(
                        f"Split file was given, but `num_val ({self.num_val})"
                        + f" != len(val_idx)` ({len(self.val_idx)})!"
                    )
                if self.num_test and self.num_test != len(self.test_idx):
                    logging.warning(
                        f"Split file was given, but `num_test ({self.num_test})"
                        + f" != len(test_idx)` ({len(self.test_idx)})!"
                    )
            else:
                self._log_with_rank("Create split")

                # if not self.num_train or not self.num_val:
                #     raise AtomsDataModuleError(
                #         "If no `split_file` is given, the sizes of the training and"
                #         + " validation partitions need to be set!"
                #     )

                self.train_idx, self.val_idx, self.test_idx = self.splitting.split(
                    self.dataset, self.num_train, self.num_val, self.num_test
                )

                if self.split_file is not None:
                    self._log_with_rank("Save split")
                    np.savez(
                        self.split_file,
                        train_idx=self.train_idx,
                        val_idx=self.val_idx,
                        test_idx=self.test_idx,
                    )

        self._log_with_rank("Exit splitting lock")

    def predict_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            shuffle=False,
            pin_memory=self._pin_memory,
        )

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool, mode: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # was modified to use the test_dataloader.
        if mode == "train":
            dataloader = self.train_dataloader()
        elif mode == "val":
            dataloader = self.val_dataloader()
        elif mode == "test":
            dataloader = self.test_dataloader()
        else:
            raise ValueError(
                "Invalid dataloader type specified. Must be 'train', 'val', or 'test'."
            )

        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats:
            return self._stats[key]

        stats = calculate_stats(
            dataloader,
            divide_by_atoms={property: divide_by_atoms},
            atomref=self.train_dataset.atomrefs if remove_atomref else None,
        )[property]
        self._stats[key] = stats
        return stats
