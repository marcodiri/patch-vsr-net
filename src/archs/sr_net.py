import torch.nn as nn
import torch.nn.functional as F

from archs.arch_utils import BaseGenerator


class ResidualBlock(nn.Module):
    """Residual block without batch normalization"""

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(BaseGenerator):
    """Reconstruction & Upsampling network"""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        nf=64,
        nb=16,
        scale_factor=4,
        residual=False,
    ):
        super(SRNet, self).__init__()

        self.save_hyperparameters()

        self.residual = residual

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True),
            nn.ReLU(),
        )

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(),
        )

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(scale_factor),
            nn.ReLU(),
        )

        # output conv.
        self.conv_out = nn.Conv2d(
            nf // scale_factor**2, out_channels, 3, 1, 1, bias=True
        )

    def forward(self, lr_curr):
        """lr_curr: the current lr data in shape nchw"""

        out = self.conv_in(lr_curr)
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)

        if self.residual:
            out += F.interpolate(
                lr_curr, scale_factor=self.hparams.scale_factor, mode="bicubic"
            )

        return out
