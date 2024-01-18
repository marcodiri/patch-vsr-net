import lightning as L
import torch
import torch.nn as nn
from einops import rearrange


class BaseGenerator(L.LightningModule):
    pass


class BaseDiscriminator(L.LightningModule):
    pass


class CrossAttention(nn.Module):
    def __init__(self, in_channels, dim_head=32, heads=8):
        super().__init__()

        self.in_channels = in_channels
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads
        kv_in_channels = in_channels

        self.to_q = nn.Conv2d(in_channels, dim_inner, 1, bias=False)
        self.to_kv = nn.Conv2d(kv_in_channels, dim_inner * 2, 1, bias=False)
        self.to_out = nn.Conv2d(dim_inner, in_channels, 1, bias=False)

    def forward(self, x1, x2):
        """
        Forward pass of the CrossAttention module.

        Args:
        - x1 (torch.Tensor): Input tensor of shape (batch, m, in_channels, height, width)
            representing a set of m source images.
        - x2 (torch.Tensor): Input tensor of shape (batch, n, in_channels, height, width)
            representing a set of n target images related to the m source images.

        Returns:
        - torch.Tensor: Output tensor of shape (batch, m, in_channels, height, width)
            after applying cross-attention.

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
        assert x1.shape[:2] == x2.shape[:2]

        b, m, c, x1_h, x1_w = x1.shape
        b, m, n, c, x2_h, x2_w = x2.shape

        # x1 = x1.view(-1, c, x1_h, x1_w)
        # x2 = x2.view(-1, c, x2_h, x2_w)
        x1 = rearrange(x1, "b m c x y -> (b m) c x y")
        x2 = rearrange(x2, "b m n c x y -> (b m n) c x y")

        q, k, v = (self.to_q(x1), *self.to_kv(x2).chunk(2, dim=1))

        k, v = map(
            lambda t: rearrange(
                t, "(b m n) (h d) x y -> (b m h) (n x y) d", m=m, n=n, h=self.heads
            ),
            (k, v),
        )
        q = rearrange(q, "(b m) (h d) x y -> (b m h) (x y) d", m=m, h=self.heads)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        # ((b m h) (x y) (n x y))
        # one attn matrix for each of the m blocks and for each head
        # relating pixels in a block m to all the pixels in the n similar blocks

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

        out = rearrange(out, "(b m) d x y -> b m d x y", m=m)
        return out


if __name__ == "__main__":
    cross = CrossAttention(3)

    out = cross(torch.rand((6, 4, 3, 6, 6)), torch.rand((6, 4, 3, 3, 6, 6)))
    print(out.shape)
