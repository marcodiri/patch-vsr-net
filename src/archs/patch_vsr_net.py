import torch
import torch.nn.functional as F

from archs.align_net import AlignNet, AlignNet2
from archs.arch_utils import BaseGenerator
from archs.sr_net import SRNet


class PatchVSRNet(BaseGenerator):
    def __init__(
        self,
        in_channels=3,
        scale_factor=4,
        residual=True,
        *,
        align_net: BaseGenerator,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["align_net"])

        self.align_net = align_net
        # self.align_net = AlignNet2(3, 8)
        self.sr_net = SRNet(
            in_channels=in_channels * 2, scale_factor=scale_factor, residual=False
        )

    def forward(self, lr_data):
        n, t, c, lr_h, lr_w = lr_data.shape
        current_idx = t // 2
        frame_t = lr_data[:, current_idx]

        align_res = self.align_net(lr_data)
        out = torch.cat([frame_t, align_res["aligned_patch"]], dim=1)
        out = self.sr_net(out)

        if self.hparams.residual:
            out += F.interpolate(
                frame_t, scale_factor=self.hparams.scale_factor, mode="bicubic"
            )

        out = F.tanh(out)

        return out, align_res


if __name__ == "__main__":
    from torchsummary import summary

    net = PatchVSRNet(align_net=AlignNet2(3, 8))
    summary(net, (3, 3, 96, 96), device="cpu")
