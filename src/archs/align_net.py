import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from archs.arch_utils import (
    BaseGenerator,
    Conv3Block,
    CrossAttention,
    CrossAttention2,
    ResNet,
    ResnetBlock,
)
from utils.data_utils import blocks_to_tensor, similarity_matrix, tensor_to_blocks


class AlignNet(BaseGenerator):
    def __init__(self, in_channels, top_k, attn_residual=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.cross_attn = CrossAttention(
            in_channels=in_channels,
            residual=attn_residual,
            **kwargs,
        )

        self.feat_net = ResNet()

    def forward(self, lr_data, block_size, stride):
        b, t, c, lr_h, lr_w = lr_data.shape
        current_idx = t // 2
        frame_t = lr_data[:, current_idx]
        frame_tm1 = lr_data[:, current_idx - 1]

        kernel_size_t = block_size
        stride_t = kernel_size_t

        blocks_t = tensor_to_blocks(frame_t, kernel_size_t, stride_t)
        _, n_blocks_t, _, bh, bw = blocks_t.shape

        kernel_size_tm1, stride_tm1 = block_size, stride
        blocks_tm1 = tensor_to_blocks(frame_tm1, kernel_size_tm1, stride_tm1)
        _, n_blocks_tm1, _, _, _ = blocks_tm1.shape

        # extract block features
        blocks_t_feat = self.feat_net(blocks_t.view(-1, c, bh, bw))[3]
        blocks_tm1_feat = self.feat_net(blocks_tm1.view(-1, c, bh, bw))[3]
        blocks_t_feat = rearrange(
            blocks_t_feat, "(b n) c_f h_f w_f -> b n c_f h_f w_f", n=n_blocks_t
        )
        blocks_tm1_feat = rearrange(
            blocks_tm1_feat, "(b n) c_f h_f w_f -> b n c_f h_f w_f", n=n_blocks_tm1
        )

        sim = similarity_matrix(blocks_t_feat, blocks_tm1_feat)

        # get top k for each block (row)
        _, topk_idx = torch.topk(sim, self.hparams.top_k)

        topk_blocks = blocks_tm1[torch.arange(b)[:, None, None], topk_idx]

        recons_blocks, attn_mat = self.cross_attn(blocks_t, topk_blocks)

        reassembled = blocks_to_tensor(
            recons_blocks, frame_t.shape, kernel_size_t, stride_t
        )

        return {"aligned_patch": reassembled, "attn_mat": attn_mat}


class InterframeAligner(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.cross_attn = CrossAttention2(
            in_channels=in_channels,
            **kwargs,
        )

        self.feat_net = nn.Sequential(
            *[
                Conv3Block(in_channels, 32),
                ResnetBlock(32, 64),
                ResnetBlock(64, 64),
                nn.Conv2d(64, 32, 1),
            ]
        )

    def forward(self, frame_tm1, frame_t, kernel_size):
        b, c, lr_h, lr_w = frame_t.shape

        frame_t_feat = self.feat_net(frame_t)
        frame_tm1_feat = self.feat_net(frame_tm1)

        stride = kernel_size

        blocks_tm1 = tensor_to_blocks(frame_tm1, kernel_size, stride)

        blocks_t_feat = tensor_to_blocks(frame_t_feat, kernel_size, stride)
        _, n_blocks_t, _, bh, bw = blocks_t_feat.shape

        blocks_tm1_feat = tensor_to_blocks(frame_tm1_feat, kernel_size, stride)
        _, n_blocks_tm1, _, _, _ = blocks_tm1_feat.shape

        recons_blocks, attn_mat = self.cross_attn(
            blocks_tm1, blocks_tm1_feat, blocks_t_feat
        )

        reassembled = blocks_to_tensor(
            recons_blocks, frame_t.shape, kernel_size, stride
        )

        reassembled = F.tanh(reassembled)

        return {"aligned_patch": reassembled, "attn_mat": attn_mat}


class AlignNet2(BaseGenerator):
    def __init__(self, in_channels, num_blocks, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.aligner1 = InterframeAligner(in_channels, **kwargs)
        self.aligner2 = InterframeAligner(in_channels, **kwargs)
        self.aligner3 = InterframeAligner(in_channels, **kwargs)
        self.aligner4 = InterframeAligner(in_channels, **kwargs)
        self.aligner5 = InterframeAligner(in_channels, **kwargs)
        self.aligner6 = InterframeAligner(in_channels, **kwargs)

        self.fusion = nn.Sequential(
            *[
                Conv3Block(in_channels * 7, 32),
                Conv3Block(32, 32),
                Conv3Block(32, 6),
            ]
        )

    def forward(self, lr_data):
        b, t, c, lr_h, lr_w = lr_data.shape

        current_idx = t // 2
        input_frames = lr_data[:, current_idx - 1 : current_idx + 2]
        _, n, _, _, _ = input_frames.shape

        kernel_size = lr_h // self.hparams.num_blocks

        # frame t-1
        aligned1 = self.aligner1(
            input_frames[:, 0],
            input_frames[:, 1],
            kernel_size,
        )["aligned_patch"]
        assert aligned1.shape == (b, c, lr_h, lr_w)

        input_frames2 = F.interpolate(
            input_frames.view(-1, c, lr_h, lr_w),
            scale_factor=1 / 2,
            mode="bicubic",
        ).view(b, n, c, lr_h // 2, lr_w // 2)
        aligned2 = self.aligner2(
            input_frames2[:, 0],
            input_frames2[:, 1],
            kernel_size,
        )["aligned_patch"]
        aligned2 = F.interpolate(
            aligned2,
            scale_factor=2,
            mode="bicubic",
        )
        assert aligned2.shape == (b, c, lr_h, lr_w)

        input_frames3 = F.interpolate(
            input_frames.view(-1, c, lr_h, lr_w),
            scale_factor=1 / 4,
            mode="bicubic",
        ).view(b, n, c, lr_h // 4, lr_w // 4)
        aligned3 = self.aligner3(
            input_frames3[:, 0],
            input_frames3[:, 1],
            kernel_size,
        )["aligned_patch"]
        aligned3 = F.interpolate(
            aligned3,
            scale_factor=4,
            mode="bicubic",
        )
        assert aligned3.shape == (b, c, lr_h, lr_w)

        # frame t+1
        aligned4 = self.aligner4(
            input_frames[:, 2],
            input_frames[:, 1],
            kernel_size,
        )["aligned_patch"]
        assert aligned4.shape == (b, c, lr_h, lr_w)

        aligned5 = self.aligner5(
            input_frames2[:, 2],
            input_frames2[:, 1],
            kernel_size,
        )["aligned_patch"]
        aligned5 = F.interpolate(
            aligned5,
            scale_factor=2,
            mode="bicubic",
        )
        assert aligned5.shape == (b, c, lr_h, lr_w)

        aligned6 = self.aligner6(
            input_frames3[:, 2],
            input_frames3[:, 1],
            kernel_size,
        )["aligned_patch"]
        aligned6 = F.interpolate(
            aligned6,
            scale_factor=4,
            mode="bicubic",
        )
        assert aligned6.shape == (b, c, lr_h, lr_w)

        aligned_all = torch.cat(
            [
                aligned1,
                aligned2,
                aligned3,
                aligned4,
                aligned5,
                aligned6,
                lr_data[:, current_idx],
            ],
            dim=1,
        )

        M = self.fusion(aligned_all)
        M = M.softmax(dim=1)

        aligned_all = (
            aligned1 * M[:, 0][:, None]
            + aligned2 * M[:, 1][:, None]
            + aligned3 * M[:, 2][:, None]
            + aligned4 * M[:, 3][:, None]
            + aligned5 * M[:, 4][:, None]
            + aligned6 * M[:, 5][:, None]
        )

        aligned_all = F.tanh(aligned_all)

        return {"aligned_patch": aligned_all}


if __name__ == "__main__":
    from torchsummary import summary

    # net = AlignNet(3, 5)
    # summary(net, (2, 3, 64, 64), 64 // 2, 64 // 16, device="cpu")

    net = AlignNet2(3, 16)
    summary(net, (3, 3, 96, 96), device="cpu")
