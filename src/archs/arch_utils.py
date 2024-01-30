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
    """
    CrossAttention module for performing cross-attention mechanism.

    Args:
        in_channels (int): Number of input channels for the feature network.
        dim (int): Dimensionality of the output feature space (default is 32).

    Methods:
        forward(target_blocks, reference_blocks):
            Perform forward pass through the cross-attention mechanism.

            Args:
                target_blocks (torch.Tensor): Tensor of shape (batch, m, in_channels, height, width)
                representing a set of m source images.
                reference_blocks (torch.Tensor): tensor of shape (batch, m, n, in_channels, height, width)
                representing a set of n target images related to the m source images.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor after cross-attention
                with shape (batch, m, in_channels, height, width) and the attention weights tensor with
                shape (batch, m, n).
    """

    def __init__(self, in_channels, dim=32):
        super().__init__()

        self.scale = dim**-0.5

        self.feat_net = nn.Sequential(
            *[
                Conv3Block(in_channels, 32),
                ResnetBlock(32, 64),
                ResnetBlock(64, 64),
                nn.Conv2d(64, dim, 1),
            ]
        )

    def forward(self, target_blocks, reference_blocks):
        assert reference_blocks.shape[:2] == target_blocks.shape[:2]

        b, m, n, c, h, w = reference_blocks.shape
        b, m, _, _, _ = target_blocks.shape

        q, k, v = (
            self.feat_net(target_blocks.view(-1, c, h, w)),
            self.feat_net(reference_blocks.view(-1, c, h, w)),
            reference_blocks.view(-1, c, h, w),
        )

        k, v = map(
            lambda t: rearrange(t, "(b m n) d x y -> (b m) (n x y) d", m=m, n=n),
            (k, v),
        )
        q = rearrange(q, "(b m) d x y -> (b m) (x y) d", m=m)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(
            out,
            "(b m) (x y) d -> b m d x y",
            m=m,
            x=h,
            y=w,
        ).contiguous()

        return out, attn


class CrossAttention2(nn.Module):
    """
    CrossAttention2 Module: Performs cross-attention between reference and target feature blocks.

    Args:
        dim_feat (int, optional): Dimension of the feature vectors. Defaults to 32.

    Methods:
        forward(reference_blocks, reference_feat_blocks, target_feat_blocks):
            Performs cross-attention and computes the output feature blocks.

    Input:
        - reference_blocks (torch.Tensor): Input reference blocks with shape (batch, num_blocks, channels, height, width).
        - reference_feat_blocks (torch.Tensor): Input reference feature blocks with shape (batch, num_blocks, feat_channels, feat_height, feat_width).
        - target_feat_blocks (torch.Tensor): Input target feature blocks with shape (batch, num_blocks, feat_channels, feat_height, feat_width).

    Output:
        - out (torch.Tensor): Output feature blocks after cross-attention with shape (batch, num_blocks, channels, height, width).
        - attn (torch.Tensor): Attention scores with shape (batch * num_blocks, feat_height * feat_width, feat_height * feat_width).

    Note:
        The attention mechanism is applied across feature vectors of reference and target blocks.
    """

    def __init__(self, dim_feat=32):
        super().__init__()

        self.scale = dim_feat**-0.5

    def forward(self, reference_blocks, reference_feat_blocks, target_feat_blocks):
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
