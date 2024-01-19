import lightning as L
import torch

from archs.arch_utils import CrossAttention
from utils.data_utils import blocks_to_tensor, similarity_matrix, tensor_to_blocks


class AlignNet(L.LightningModule):
    def __init__(self, in_channels, top_k, attn_residual=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.cross_attn = CrossAttention(
            in_channels=in_channels,
            residual=attn_residual,
            **kwargs,
        )

    def forward(self, lr_data, block_size, stride):
        n, t, c, lr_h, lr_w = lr_data.shape
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

        sim = similarity_matrix(blocks_t, blocks_tm1)

        # get top k for each block (row)
        _, topk_idx = torch.topk(sim, self.hparams.top_k)

        topk_blocks = blocks_tm1[torch.arange(n)[:, None, None], topk_idx]
        topk_blocks = topk_blocks.view(n, n_blocks_t, self.hparams.top_k, c, bh, bw)

        recons_blocks, attn_mat = self.cross_attn(blocks_t, topk_blocks)

        reassembled = blocks_to_tensor(
            recons_blocks, frame_t.shape, kernel_size_t, stride_t
        )

        return {"aligned_patch": reassembled, "attn_mat": attn_mat}
