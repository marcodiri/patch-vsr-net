from typing import Dict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lpips import LPIPS

from archs.arch_utils import BaseDiscriminator, BaseGenerator
from optim import define_criterion
from optim.losses import SSIM


class VSRGAN(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        *,
        losses: Dict,
        gen_lr: float = 5e-5,
        dis_lr: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.G = generator
        self.D = discriminator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # align criterion
        self.algn_crit, self.algn_w = define_criterion(losses.get("align_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))

        # gan criterion
        self.gan_crit, self.gan_w = define_criterion(losses.get("gan_crit"))

        # validation losses
        self.lpips_alex = LPIPS(net="alex", version="0.1")
        self.ssim = SSIM()

        self.automatic_optimization = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)
        optim_D = torch.optim.Adam(params=self.D.parameters(), lr=self.hparams.dis_lr)

        return optim_G, optim_D

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optim_G, optim_D = self.optimizers()

        # ------------ prepare data ------------ #
        gt_data, lr_data = batch

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        current_idx = t // 2
        gt_frame_t = gt_data[:, current_idx]
        lr_frame_t = lr_data[:, current_idx]

        assert t > 2, "A temporal radius of at least 3 is needed"

        to_log, to_log_prog = {}, {}

        # ------------ clear optimizers ------------ #
        optim_G.zero_grad()
        optim_D.zero_grad()

        # ------------ forward G ------------ #
        hr_fake, align_res = self.G(lr_data)

        # ------------ forward D ------------ #
        for param in self.D.parameters():
            param.requires_grad = True

        # forward real sequence (gt)
        real_pred = self.D(gt_frame_t)

        # forward fake sequence (hr)
        fake_pred = self.D(hr_fake.detach())

        # ------------ optimize D ------------ #
        to_log, to_log_prog = {}, {}

        loss_real_D = self.gan_crit(real_pred, True)
        loss_fake_D = self.gan_crit(fake_pred, False)
        loss_D = loss_real_D + loss_fake_D

        # update D
        self.manual_backward(loss_D)
        optim_D.step()
        to_log["D_real_loss"] = loss_real_D
        to_log["D_fake_loss"] = loss_fake_D

        to_log_prog["D_loss"] = loss_D

        # ------------ optimize G ------------ #
        for param in self.D.parameters():
            param.requires_grad = False

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
                    scale_factor=1 / self.G.hparams.scale_factor,
                    mode="bicubic",
                ),
            )
            loss_G += self.algn_w * loss_algn_G
            to_log["G_align_loss"] = loss_algn_G

        # feature (feat) loss
        if self.feat_crit is not None:
            loss_feat_G = self.feat_crit(hr_fake, gt_data[:, t // 2].detach()).mean()

            loss_G += self.feat_w * loss_feat_G
            to_log["G_lpip_loss"] = loss_feat_G

        # gan loss
        fake_pred = self.D(hr_fake)

        loss_gan_G = self.gan_crit(fake_pred, True)
        loss_G += self.gan_w * loss_gan_G
        to_log["G_gan_loss"] = loss_gan_G
        to_log_prog["G_loss"] = loss_G

        # update G
        self.manual_backward(loss_G)
        optim_G.step()

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        gt_data, lr_data = batch
        _, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        hr_fake, align_res = self.G(lr_data)

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
