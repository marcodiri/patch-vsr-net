import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from metrics.metrics import (
    calculate_arniqa_video,
    calculate_lpips_video,
    calculate_psnr_video,
    calculate_ssim_video,
)
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s1",
    "--seq1",
    type=str,
    help="Sequence one frames folder",
)
parser.add_argument(
    "-s2",
    "--seq2",
    type=str,
    required=False,
    help="Sequence two frames folder",
)
parser.add_argument(
    "-m",
    "--metrics",
    nargs="+",
    default=["psnr"],
    help="Metrics to calculate. List of [psnr, ssim]. Default: psnr",
)
parser.add_argument(
    "--ext",
    type=str,
    default="png",
    help="Extension of frame images",
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    help="torch.device to use for computations",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output file",
)

args = parser.parse_args()

seq1_paths = sorted(data_utils.get_pics_in_subfolder(args.seq1, ext=args.ext))
if args.seq2 is not None:
    seq2_paths = sorted(data_utils.get_pics_in_subfolder(args.seq2, ext=args.ext))
    assert len(seq1_paths) == len(seq2_paths)

seq1_list = []
seq2_list = []
for i in range(len(seq1_paths)):
    frame1 = data_utils.load_img(seq1_paths[i])
    seq1_list.append(frame1)
    if args.seq2 is not None:
        frame2 = data_utils.load_img(seq2_paths[i])
        seq2_list.append(frame2)

print("Calculating...")

if "psnr" in args.metrics:
    psnr = calculate_psnr_video(seq1_list, seq2_list, 0)
    print(f"PSNR: {psnr}")
    with open(args.output, "a") as f:
        print(f"PSNR: {psnr}", file=f)

if "ssim" in args.metrics:
    ssim = calculate_ssim_video(seq1_list, seq2_list, 0)
    print(f"SSIM: {ssim}")
    with open(args.output, "a") as f:
        print(f"SSIM: {ssim}", file=f)

if "lpips" in args.metrics:
    lpips = calculate_lpips_video(seq1_list, seq2_list, "alex", device=args.device)
    print(f"LPIPS: {lpips}")
    with open(args.output, "a") as f:
        print(f"LPIPS: {lpips}", file=f)

if "arniqa" in args.metrics:
    arniqa = calculate_arniqa_video(seq1_list, device=args.device)
    print(f"ARNIQA: {arniqa}")
    with open(args.output, "a") as f:
        print(f"ARNIQA: {arniqa}", file=f)
