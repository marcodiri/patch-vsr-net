import torch.nn as nn
import torch.nn.functional as F

from archs.arch_utils import BaseGenerator


class ResidualBlock(nn.Module):
    """Residual block without batch normalization"""

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(BaseGenerator):
    """Reconstruction & Upsampling network"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, residual=False):
        super(SRNet, self).__init__()

        self.save_hyperparameters()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )

        # output conv.
        self.conv_out = nn.Conv2d(nf // scale**2, out_nc, 3, 1, 1, bias=True)

    def forward(self, lr_curr):
        """lr_curr: the current lr data in shape nchw"""

        out = self.conv_in(lr_curr)
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)

        if self.hparams.residual:
            out += F.interpolate(
                lr_curr, scale_factor=self.hparams.scale, mode="bicubic"
            )

        out = F.tanh(out)

        return out
