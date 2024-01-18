from typing import Dict

import lightning as L
import torch
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

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # ping-pong criterion
        self.pp_crit, self.pp_w = define_criterion(losses.get("pingpong_crit"))

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

        assert t > 1, "A temporal radius of at least 2 is needed"

        # # augment data for pingpong criterion
        # if self.pp_crit is not None:
        #     # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
        #     lr_rev = lr_data.flip(1)[:, 1:, ...]
        #     gt_rev = gt_data.flip(1)[:, 1:, ...]

        #     lr_data = torch.cat([lr_data, lr_rev], dim=1)
        #     gt_data = torch.cat([gt_data, gt_rev], dim=1)

        # ------------ forward G ------------ #
        hr_fake = self.G(lr_data)

        # ------------ optimize G ------------ #
        to_log, to_log_prog = {}, {}

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(hr_fake, gt_data[:, t // 2])
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # feature (feat) loss
        if self.feat_crit is not None:
            loss_feat_G = self.feat_crit(hr_fake, gt_data[:, t // 2].detach()).mean()

            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        # # ping-pong (pp) loss
        # if self.pp_crit is not None:
        #     hr_data_fw = hr_fake[:, : t - 1, ...]  #    -------->|
        #     hr_data_bw = hr_fake[:, t:, ...].flip(1)  # <--------|

        #     loss_pp_G = self.pp_crit(hr_data_fw, hr_data_bw)
        #     loss_G += self.pp_w * loss_pp_G
        #     to_log["G_ping_pong_loss"] = loss_pp_G

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data, lr_data = batch
        _, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        hr_fake = self.G(lr_data)

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

        return lr_data[:, t // 2], gt_data[:, t // 2], hr_fake
