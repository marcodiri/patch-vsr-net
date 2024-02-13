from typing import List

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from data.folder_dataset import VideoFolder, VideoFolderPaired


class VideoFolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        hr_path,
        lr_path="",
        *,
        patch_size=None,
        augment=False,
        tempo_extent=None,
        hr_path_filter="",
        lr_path_filter="",
        num_classes=None,
        validation_classes: List[str] = [],
        jump_frames=1,
        dataset_upscale_factor=4,
        batch_size=32,
        pin_memory=True,
    ):
        """
        Custom PyTorch Lightning DataModule.

        See :class:`~folder_dataset.VideoFolder` for details on args.

        Args
            train_pct (float):
                Percentage of the training data to use as validation.
            batch_size (int):
                Size of every training batch.
        """

        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_set = VideoFolderPaired(**self.hparams, train=True)
            if len(self.hparams.validation_classes) > 0:
                self.valid_set = VideoFolderPaired(**self.hparams, train=False)
            else:
                self.valid_set = None
        if stage == "predict":
            self.predict_set = VideoFolder(
                self.hparams.hr_path,
                tempo_extent=self.hparams.tempo_extent,
                hr_path_filter=self.hparams.hr_path_filter,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=20,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
        )
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader_eval = []
        if self.valid_set is not None:
            data_loader_eval = DataLoader(
                dataset=self.valid_set,
                batch_size=self.hparams.batch_size,
                num_workers=20,
                shuffle=True,
                pin_memory=self.hparams.pin_memory,
            )
        return data_loader_eval

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        data_loader_predict = DataLoader(
            dataset=self.predict_set,
            num_workers=20,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )
        return data_loader_predict
