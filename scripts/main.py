from lightning.pytorch.cli import LightningCLI

from data.datamodule import VideoFolderDataModule
from utils.lit_callbacks import ImageLog, MemProfiler  # noqa: F401


def cli_main():
    cli = LightningCLI(
        datamodule_class=VideoFolderDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
