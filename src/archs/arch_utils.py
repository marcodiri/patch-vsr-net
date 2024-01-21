from collections import namedtuple
from typing import Iterable, List, Optional

import lightning as L
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import models as tv


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
        # self.to_out = nn.Conv2d(dim_inner, in_channels, 1, bias=False)
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

        structure_image_v = structure_image.view(-1, c, x1_h, x1_w)
        appearance_images_v = appearance_images.view(-1, c, x2_h, x2_w)

        q, k, v = (
            self.to_q(structure_image_v),
            self.to_k(appearance_images_v),
            appearance_images_v,
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
        if self.residual:
            out = self.gamma * out + structure_image_v

        out = rearrange(out, "(b m) d x y -> b m d x y", m=m).contiguous()

        return out, attn


class CrossAttention2(nn.Module):
    def __init__(self, in_channels, dim_head=32, heads=1):
        super().__init__()

        self.in_channels = in_channels
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads
        kv_in_channels = in_channels

    def forward(self, reference_blocks, reference_feat_blocks, target_feat_blocks):
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
        assert reference_feat_blocks.shape[:2] == target_feat_blocks.shape[:2]

        b, m, c, x_h, x_w = reference_blocks.shape
        b, m, cb, x1_h, x1_w = reference_feat_blocks.shape
        b, m, cb, x2_h, x2_w = target_feat_blocks.shape

        q, k, v = (
            target_feat_blocks.view(-1, cb, x2_h, x2_w),
            reference_feat_blocks.view(-1, cb, x1_h, x1_w),
            reference_blocks.view(-1, c, x_h, x_w),
        )

        q, k, v = map(
            lambda t: rearrange(t, "b c x y -> b (x y) c"),
            (q, k, v),
        )

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(
            out,
            "(b m) (x y) d -> (b m) d x y",
            m=m,
            x=x_h,
            y=x_w,
        )

        out = rearrange(out, "(b m) d x y -> b m d x y", m=m).contiguous()

        return out, attn


class Conv3Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
    ):
        super().__init__()

        self.block1 = Conv3Block(dim, dim_out)
        self.block2 = Conv3Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        return h + self.res_conv(x)


class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(ResNet, self).__init__()
        self.net = tv.resnet18(weights="DEFAULT")

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ["relu1", "conv2", "conv3", "conv4", "conv5"])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out


if __name__ == "__main__":
    # cross = CrossAttention(3, heads=1).cuda(1)

    # out, attn_mat = cross(
    #     torch.rand((6, 4, 3, 48, 48)).cuda(1), torch.rand((6, 4, 5, 3, 48, 48)).cuda(1)
    # )
    # print(out.shape)
    # print(attn_mat.shape)

    cross = CrossAttention2(3, heads=1)

    out, attn_mat = cross(
        torch.rand((6, 256, 3, 6, 6)),
        torch.rand((6, 256, 32, 6, 6)),
        torch.rand((6, 256, 32, 6, 6)),
    )
    print(out.shape)
    print(attn_mat.shape)

    from torchsummary import summary

    net = ResNet()
    feats = net(torch.rand(2, 3, 48, 48))
    print([feat.shape for feat in feats])
    print(feats.conv4.shape)
    print(feats.conv4.requires_grad)

    resblock = ResnetBlock(3, 3)
    summary(resblock, (3, 96, 96), device="cpu")
