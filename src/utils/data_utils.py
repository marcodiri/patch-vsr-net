from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def normalize_img(x):
    return (x - 0.5) * 2.0


def denormalize_img(x):
    return x * 0.5 + 0.5


def downsample(img):
    w, h = img.size
    img = img.resize((w // 2, h // 2), Image.ANTIALIAS)
    return img


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        normalize_img,
    ]
)

transform_from_np = torchvision.transforms.Compose(
    [
        lambda x: x.permute(2, 1, 0),
        normalize_img,
    ]
)

de_normalize = denormalize_img
de_transform = torchvision.transforms.Compose(
    [de_normalize, torchvision.transforms.ToPILImage()]
)


def get_pics_in_subfolder(path, ext="jpg"):
    return list(Path(path).rglob(f"*.{ext}"))


def parse_frame_title(filename: str):
    file_parts = filename.split("_")
    seq = "_".join(file_parts[:-1])
    _, dim, _, _, _, frm_ext = file_parts
    (hr_w, hr_h) = dim.split("x")
    frm, _ = frm_ext.split(".")
    return seq, (int(hr_w), int(hr_h)), int(frm)


def load_img(path):
    img = Image.open(path)
    return img


def tensor_to_blocks(input_tensor, kernel_size, stride=1):
    """
    Divide a tensor into blocks of size kernel_size along the last two dimensions with a specified stride.

    Parameters:
    - input_tensor (torch.Tensor): Input tensor of shape (n c *).
    - kernel_size (int): Spatial dimension of the blocks.
    - stride (int, optional): Stride for overlapping blocks. Default is 1.

    Returns:
    - torch.Tensor: Tensor with blocks of size kernel_size. The last two dimensions represent the spatial dimensions of the blocks.
    """

    unfolded = F.unfold(input_tensor, (kernel_size, kernel_size), stride=stride)

    unfolded = unfolded.transpose(-1, -2)
    # Reshape the unfolded tensor to (batch_size, n_blocks, channels, block_size_n, block_size_m)
    unfolded = unfolded.view(
        input_tensor.size(0),
        -1,
        input_tensor.size(1),
        kernel_size,
        kernel_size,
    )

    return unfolded.contiguous()


def blocks_to_tensor(
    input_tensor,
    original_shape,
    kernel_size,
    stride,
    normalize=False,
):
    """
    Perform a folding operation on the input tensor.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to be folded.
    - original_shape (tuple): The original shape of the input tensor before unfolding.
    - kernel_size (int or tuple of ints): The size of the convolutional kernel.
    - stride (int or tuple of ints): The stride of the folding operation.
    - normalize (bool, default=False): If set to True, the folded result is normalized
        based on the number of contributions to each position.
        Useful if the blocks are overlapping.

    Returns:
    - torch.Tensor: The folded tensor resulting from the folding operation.
    """
    output_size = original_shape[-2:]

    # Reshape the input_tensor tensor to (batch_size, n_blocks, block_size_n*block_size_m*channels)
    input_tensor = input_tensor.view(
        original_shape[0],
        -1,
        kernel_size * kernel_size * original_shape[1],
    )
    input_tensor = input_tensor.transpose(-1, -2)

    folded = F.fold(input_tensor, output_size, kernel_size=kernel_size, stride=stride)
    if normalize:
        norm_map = F.fold(
            F.unfold(
                torch.ones(original_shape, device=input_tensor.device),
                kernel_size=kernel_size,
                stride=stride,
            ),
            output_size,
            kernel_size=kernel_size,
            stride=stride,
        )
        folded /= norm_map

    return folded.contiguous()


def similarity_matrix(tensor1, tensor2):
    """
    Compute the cosine similarity matrix between between the row-th image in the first tensor with the col-th image in the second tensor.

    Args:
    - tensor1 (torch.Tensor): The first tensor containing images. Should have shape (batch_size, num_images1, num_channels, height, width).
    - tensor2 (torch.Tensor): The second tensor containing images. Should have shape (batch_size, num_images2, num_channels, height, width).

    Returns:
    - torch.Tensor: The cosine similarity matrix. Has shape (batch_size, num_images1, num_images2).

    Example:
    ```python
    # Create random tensors for demonstration purposes
    tensor1 = torch.randn((2, 9, 3, 4, 4))
    tensor2 = torch.randn((2, 9, 3, 4, 4))

    # Compute cosine similarity matrix
    result = similarity_matrix(tensor1, tensor2)

    print(result.shape)  # Output: torch.Size([2, 9, 9])
    ```
    """
    return F.cosine_similarity(
        tensor1[:, :, None],
        tensor2[:, None],
        dim=-1,
    ).mean(dim=(-1, -2))
