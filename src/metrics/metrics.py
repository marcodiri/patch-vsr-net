import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS

from metrics.metrics_utils import center_corners_crop, reorder_image, to_y_channel
from utils import data_utils
from utils.color_utils import rgb2ycbcr_pt


def calculate_psnr(
    img, img2, crop_border, input_order="HWC", test_y_channel=False, **kwargs
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def calculate_psnr_video(
    seq1, seq2, crop_border, input_order="HWC", test_y_channel=False, **kwargs
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        seq1 (List[ndarray]): List of images with range [0, 255].
        seq2 (List[ndarray]): List of images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert len(seq1) == len(seq2)

    mse_list = []
    for i in range(len(seq1)):
        img = np.array(seq1[i])
        img2 = np.array(seq2[i])

        assert (
            img.shape == img2.shape
        ), f"Image shapes are different: {img.shape}, {img2.shape}."
        if input_order not in ["HWC", "CHW"]:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
            )
        img = reorder_image(img, input_order=input_order)
        img2 = reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel:
            img = to_y_channel(img)
            img2 = to_y_channel(img2)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img - img2) ** 2)
        mse_list.append(mse)

    mse = np.array(mse_list).mean()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def calculate_ssim(
    img, img2, crop_border, input_order="HWC", test_y_channel=False, **kwargs
):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_ssim_video(
    seq1, seq2, crop_border, input_order="HWC", test_y_channel=False, **kwargs
):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert len(seq1) == len(seq2)

    ssim_list = []
    for i in range(len(seq1)):
        img = np.array(seq1[i])
        img2 = np.array(seq2[i])

        assert (
            img.shape == img2.shape
        ), f"Image shapes are different: {img.shape}, {img2.shape}."
        if input_order not in ["HWC", "CHW"]:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
            )
        img = reorder_image(img, input_order=input_order)
        img2 = reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel:
            img = to_y_channel(img)
            img2 = to_y_channel(img2)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)

        ssims = []
        for i in range(img.shape[2]):
            ssims.append(_ssim(img[..., i], img2[..., i]))
        ssim = np.array(ssims).mean()
        ssim_list.append(ssim)
    return np.array(ssim_list).mean()


def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255.0, img2 * 255.0)
    return ssim


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = (
        torch.from_numpy(window)
        .view(1, 1, 11, 11)
        .expand(img.size(1), 1, 11, 11)
        .to(img.dtype)
        .to(img.device)
    )

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(
        img2, window, stride=1, padding=0, groups=img2.shape[1]
    )  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2
    )

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def calculate_lpips_video(seq1, seq2, net="alex", device="cpu"):
    lpips = LPIPS(net=net, version="0.1").to(device)

    seq1_pt = torch.stack([data_utils.transform(frm) for frm in seq1])
    seq2_pt = torch.stack([data_utils.transform(frm) for frm in seq2])

    return lpips(seq1_pt, seq2_pt).mean()


def calculate_arniqa_video(seq, device="cpu"):
    import torchvision.transforms as transforms

    model = torch.hub.load(
        repo_or_dir="miccunifi/ARNIQA",
        source="github",
        model="ARNIQA",
        regressor_dataset="kadid10k",
    )
    model.eval().to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    score_list = []
    for img in seq:
        img = img.convert("RGB")
        # Get the half-scale image
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

        img = center_corners_crop(img, crop_size=224)
        img_ds = center_corners_crop(img_ds, crop_size=224)

        # Preprocess the images
        img = [transforms.ToTensor()(crop) for crop in img]
        img = torch.stack(img, dim=0)
        img = normalize(img).to(device)
        img_ds = [transforms.ToTensor()(crop) for crop in img_ds]
        img_ds = torch.stack(img_ds, dim=0)
        img_ds = normalize(img_ds).to(device)

        # Compute the quality score
        with torch.no_grad(), torch.cuda.amp.autocast():
            score = model(img, img_ds, return_embedding=False, scale_score=True)
            # Compute the average score over the crops
            score = score.mean(0)
        score_list.append(score.cpu())

    return np.array(score_list).mean()
