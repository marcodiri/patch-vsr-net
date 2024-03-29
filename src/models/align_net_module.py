from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lpips import LPIPS

from archs.arch_utils import BaseGenerator
from optim import define_criterion
from optim.losses import SSIM, CharbonnierLoss


class AlignModule(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        *,
        losses: Dict,
        upscale_factor,
        gen_lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator"])
        self.G = generator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # validation losses
        self.pix_crit_val = CharbonnierLoss(reduction="mean")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)

        return optim_G

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # ------------ prepare data ------------ #
        gt_data, lr_data = batch

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        assert t > 1, "A temporal radius of at least 2 is needed"

        to_log, to_log_prog = {}, {}

        # ------------ forward G ------------ #
        align_res = self.G(lr_data)

        # ------------ optimize G ------------ #

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(
                align_res["aligned_patch"],
                F.interpolate(
                    gt_data[:, t // 2],
                    scale_factor=1 / self.hparams.upscale_factor,
                    mode="bicubic",
                ),
            )
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # feature (feat) loss
        if self.feat_crit is not None:
            loss_feat_G = self.feat_crit(
                align_res["aligned_patch"],
                F.interpolate(
                    gt_data[:, t // 2],
                    scale_factor=1 / self.hparams.upscale_factor,
                    mode="bicubic",
                ),
            ).mean()
            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

        return loss_G

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data, lr_data = batch
        _, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        align_res = self.G(lr_data)

        pix_loss_val = self.pix_crit_val(
            align_res["aligned_patch"],
            F.interpolate(
                gt_data[:, t // 2],
                scale_factor=1 / self.hparams.upscale_factor,
                mode="bicubic",
            ),
        )

        self.log_dict(
            {
                "val_pix_loss": pix_loss_val,
            },
            on_epoch=True,
            prog_bar=True,
        )

        return (
            (
                lr_data[:, t // 2 - 1],
                lr_data[:, t // 2],
                align_res["aligned_patch"],
                F.interpolate(
                    gt_data[:, t // 2],
                    scale_factor=1 / self.hparams.upscale_factor,
                    mode="bicubic",
                ),
            ),
            (gt_data[:, t // 2],),
            ("lq_t-1 vs lq_t vs aligned vs hq downscaled", "hq_t"),
        )
