import lightning as L
import torch
from einops import rearrange

from archs.arch_utils import BaseGenerator, CrossAttention, ResNet
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


if __name__ == "__main__":
    from torchsummary import summary

    net = AlignNet(3, 5)
    summary(net, (2, 3, 64, 64), 64 // 2, 64 // 16, device="cpu")
