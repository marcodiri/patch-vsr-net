from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lpips import LPIPS

from archs.arch_utils import BaseGenerator
from optim import define_criterion
from optim.losses import SSIM


class VSRSingle(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        *,
        losses: Dict,
        gen_lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator"])
        self.G = generator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # align criterion
        self.algn_crit, self.algn_w = define_criterion(losses.get("align_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # validation losses
        self.lpips_alex = LPIPS(net="alex", version="0.1")
        self.ssim = SSIM()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)

        return optim_G

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # ------------ prepare data ------------ #
        gt_data, lr_data = batch

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        assert t > 2, "A temporal radius of at least 3 is needed"

        lr_bic_data = F.interpolate(
            gt_data.view(-1, c, gt_h, gt_w),
            size=lr_data.shape[-2:],
            mode="bicubic",
        ).view(n, t, c, lr_h, lr_w)

        to_log, to_log_prog = {}, {}

        # ------------ forward G ------------ #
        hr_fake, align_res = self.G(lr_data, lr_bic_data=lr_bic_data)

        # ------------ optimize G ------------ #

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(hr_fake, gt_data[:, t // 2])
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # align loss
        if self.algn_crit is not None:
            loss_algn_G = self.algn_crit(
                align_res["aligned_patch"],
                F.interpolate(
                    gt_data[:, t // 2],
                    size=lr_data.shape[-2:],
                    mode="bicubic",
                ),
            )
            loss_G += self.algn_w * loss_algn_G
            to_log["G_align_loss"] = loss_algn_G

        # feature (feat) loss
        if self.feat_crit is not None:
            loss_feat_G = self.feat_crit(hr_fake, gt_data[:, t // 2].detach()).mean()

            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

        return loss_G

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data, lr_data = batch
        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        lr_bic_data = F.interpolate(
            gt_data.view(-1, c, gt_h, gt_w),
            size=lr_data.shape[-2:],
            mode="bicubic",
        ).view(n, t, c, lr_h, lr_w)

        hr_fake, align_res = self.G(lr_data, lr_bic_data=lr_bic_data)

        ssim_val = self.ssim(hr_fake, gt_data[:, t // 2]).mean()
        lpips_val = self.lpips_alex(hr_fake, gt_data[:, t // 2]).mean()

        self.log_dict(
            {
                "val_ssim": ssim_val,
                "val_lpips": lpips_val,
            },
            on_epoch=True,
            prog_bar=True,
        )

        return (
            (
                lr_data[:, t // 2],
                align_res["aligned_patch"],
                F.interpolate(
                    gt_data[:, t // 2],
                    size=lr_data.shape[-2:],
                    mode="bicubic",
                ),
            ),
            (
                gt_data[:, t // 2],
                hr_fake,
                F.interpolate(
                    lr_data[:, t // 2],
                    size=gt_data.shape[-2:],
                    mode="bicubic",
                ),
            ),
            ("lq vs aligned vs hq downscaled", "hq vs fake vs bicubic"),
        )

    def predict_step(self, batch, batch_idx) -> torch.Any:
        gt_data, lr_data = batch
        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        lr_bic_data = F.interpolate(
            gt_data.view(-1, c, gt_h, gt_w),
            size=lr_data.shape[-2:],
            mode="bicubic",
        ).view(n, t, c, lr_h, lr_w)

        hr_fake, align_res = self.G(lr_data, lr_bic_data=lr_bic_data)

        return hr_fake, align_res
