import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, repeat


class BaseGenerator(L.LightningModule):
    pass


class BaseDiscriminator(L.LightningModule):
    pass


class CrossAttention(nn.Module):
    def __init__(self, in_channels, dim_head=32, heads=8, residual=False):
        super().__init__()

        self.in_channels = in_channels
        self.heads = heads
        self.residual = residual
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads
        kv_in_channels = in_channels

        self.to_q = nn.Conv2d(in_channels, dim_inner, 1, bias=False)
        self.to_k = nn.Conv2d(kv_in_channels, dim_inner, 1, bias=False)
        self.to_v = nn.Conv2d(kv_in_channels, dim_inner, 1, bias=False)
        self.to_out = nn.Conv2d(dim_inner, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, structure_image, appearance_images):
        """
        Forward pass of the CrossAttention module.

        Args:
        - structure_image (torch.Tensor): Input tensor of shape (batch, m, in_channels, height, width)
            representing a set of m source images.
        - appearance_images (torch.Tensor): Input tensor of shape (batch, n, in_channels, height, width)
            representing a set of n target images related to the m source images.

        Returns:
        - torch.Tensor: Output tensor of shape (batch, m, in_channels, height, width)
            after applying cross-attention.
        """
        """
        Einstein Notation:
            b - batch
            m - number of source images
            n - number of target images related to each source image
            h - heads
            x - height
            y - width
            d - dimension (in_channels)
            i - source image (attend from)
            j - target image (attend to)
        """
        assert structure_image.shape[:2] == appearance_images.shape[:2]

        b, m, c, x1_h, x1_w = structure_image.shape
        b, m, n, c, x2_h, x2_w = appearance_images.shape

        structure_image = structure_image.view(-1, c, x1_h, x1_w)
        appearance_images = appearance_images.view(-1, c, x2_h, x2_w)

        q, k, v = (
            self.to_q(structure_image),
            self.to_k(appearance_images),
            self.to_v(appearance_images),
        )

        k, v = map(
            lambda t: rearrange(
                t, "(b m n) (h d) x y -> (b m h) (n x y) d", m=m, n=n, h=self.heads
            ),
            (k, v),
        )
        q = rearrange(q, "(b m) (h d) x y -> (b m h) (x y) d", m=m, h=self.heads)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(
            out,
            "(b m h) (x y) d -> (b m) (h d) x y",
            m=m,
            h=self.heads,
            x=x1_h,
            y=x1_w,
        )
        out = self.to_out(out)
        if self.residual:
            out = self.gamma * out + structure_image

        out = rearrange(out, "(b m) d x y -> b m d x y", m=m).contiguous()

        return out, attn


if __name__ == "__main__":
    cross = CrossAttention(3, heads=2).cuda(1)

    out = cross(
        torch.rand((6, 4, 3, 48, 48)).cuda(1), torch.rand((6, 4, 5, 3, 48, 48)).cuda(1)
    )
    print(out.shape)
