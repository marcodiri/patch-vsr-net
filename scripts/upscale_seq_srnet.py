import argparse
import os

import torch
import torch.nn.functional as F
import torchvision

from archs.sr_net import SRNet
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="Path to lightning model checkpoint",
)
parser.add_argument(
    "-s",
    "--seq",
    type=str,
    help="Sequence to upscale (folder with frames)",
)
parser.add_argument(
    "--save_bicubic",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Wether to alse save a bicubic upscaled sequence (requires more memory). Default: False",
)
parser.add_argument(
    "-u",
    "--scale",
    type=int,
    help="Upscale factor",
)
parser.add_argument(
    "--downscale",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Downscale original sequenze by --scale before upscaling. Dafault: False",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Output directory",
)
parser.add_argument(
    "--ext",
    type=str,
    default="png",
    help="Extension of frame images",
)

args = parser.parse_args()
generator = SRNet(
    scale_factor=args.scale,
    residual=True,
).cuda()

checkpoint = torch.load(args.ckpt)
state_dict = {
    ".".join(k.split(".")[1:]): v
    for k, v in checkpoint["state_dict"].items()
    if "G." in k
}
generator.load_state_dict(state_dict)

lr_paths = sorted(data_utils.get_pics_in_subfolder(args.seq, ext=args.ext))
lr_list = []
for p in lr_paths:
    lr = data_utils.load_img(p)
    lr = data_utils.transform(lr)
    lr_list.append(lr)
lr_seq = torch.stack(lr_list).cuda()
if args.downscale:
    lr_seq = F.interpolate(lr_seq, scale_factor=1 / args.scale, mode="bicubic")

generator.freeze()  # important: disables grads to free memory of intermediate computations
hr_list = []
for lr_t in lr_seq:
    hr_fake_t = generator(lr_t.unsqueeze(0)).squeeze(0)
    hr_list.append(hr_fake_t)
hr_fake = torch.stack(hr_list)

if args.save_bicubic:
    os.makedirs(f"./{args.output_dir}/bic/", exist_ok=True)

to_image = torchvision.transforms.ToPILImage()

print("Saving upscaled sequence...")
os.makedirs(f"./{args.output_dir}/fake/", exist_ok=True)

frm_idx_lst = ["{:04d}.png".format(i + 1) for i in range(hr_fake.size(0))]
for i in range(hr_fake.size(0)):
    hr_f = data_utils.de_transform(hr_fake[i])
    hr_f.save(f"./{args.output_dir}/fake/{frm_idx_lst[i]}")

    if args.save_bicubic:
        hr_bic = F.interpolate(
            lr_seq[:, i], scale_factor=args.scale, mode="bicubic"
        ).squeeze(0)
        hr_bic = torch.clamp(hr_bic, min=-1.0, max=1.0)
        hr_bic = data_utils.de_transform(hr_bic)
        hr_bic.save(f"./{args.output_dir}/bic/{frm_idx_lst[i]}")
