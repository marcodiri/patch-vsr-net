import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid

from utils.mem_profiler import *


class ImageLog(Callback):
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == trainer.num_val_batches[0] - 2:
            try:
                pl_module.logger.log_image(
                    key="samples",
                    images=[
                        make_grid(
                            torch.cat(outputs[i]),
                            nrow=outputs[i][0].shape[0],
                            normalize=True,
                            scale_each=True,
                        )
                        for i in range(len(outputs) - 1)
                    ],
                    caption=outputs[-1],
                )
            except Exception as e:
                print(e)


class MemProfiler(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Start recording memory snapshot history
        start_record_memory_history()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Create the memory snapshot file
        export_memory_snapshot()

        # Stop recording memory snapshot history
        stop_record_memory_history()
