import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

from data.folder_dataset import FolderDataset


class FolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        hr_path,
        lr_path="",
        extension="jpg",
        *,
        patch_size,
        tempo_extent=None,
        hr_path_filter="",
        lr_path_filter="",
        dataset_upscale_factor=4,
        train_pct=0.8,
        batch_size=32,
        pin_memory=True,
    ):
        """
        Custom PyTorch Lightning DataModule.

        See :class:`~folder_dataset.FolferDataset` for details on args.

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
            dataset = FolderDataset(**self.hparams)
            train_set_size = int(len(dataset) * self.hparams.train_pct)
            valid_set_size = len(dataset) - train_set_size

            # split the train set into two
            self.train_set, self.valid_set = random_split(
                dataset, [train_set_size, valid_set_size]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=40,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
        )
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader_eval = DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=40,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
        )
        return data_loader_eval


if __name__ == "__main__":
    dm = FolderDataModule(
        hr_path="/home/DATASETS/BVI_DVC/frames_HQ",
        lr_path="/home/DATASETS/BVI_DVC/frames/frames_CRF_22",
        extension="png",
        tempo_extent=5,
        hr_path_filter="1088",
        lr_path_filter="1088",
        patch_size=64,
        dataset_upscale_factor=2,
        train_pct=0.8,
        batch_size=4,
        pin_memory=False,
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    print()
