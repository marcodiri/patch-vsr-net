import argparse
import os

import torch
import torch.nn.functional as F
import torchvision

from archs.align_net import AlignNet2
from archs.patch_vsr_net import PatchVSRNet
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
    "--generate_bicubic",
    type=bool,
    default=False,
    help="Wether to alse generate a bicubic upscaled sequence (requires more memory). Default: False",
)
parser.add_argument(
    "-u",
    "--scale",
    type=int,
    help="Upscale factor",
)
parser.add_argument(
    "--ext",
    type=str,
    default="png",
    help="Extension of frame images",
)

args = parser.parse_args()
generator = PatchVSRNet(
    scale_factor=args.scale,
    align_net=AlignNet2(3, 128),
)  # .cuda()

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
lr_seq = torch.stack(lr_list).unsqueeze(0)  # .cuda()

generator.eval()
hr_fake = generator.forward_sequence(lr_seq)[0].squeeze(0)

if args.generate_bicubic:
    hr_bic = F.interpolate(lr_seq, scale_factor=4, mode="bicubic")
    hr_bic = torch.clamp(hr_bic, min=-1.0, max=1.0)
    os.makedirs("./output/bic/", exist_ok=True)

to_image = torchvision.transforms.ToPILImage()

print("Saving upscaled sequence...")
os.makedirs("./output/fake/", exist_ok=True)

frm_idx_lst = ["{:04d}.png".format(i + 1) for i in range(hr_fake.size(0))]
for i in range(hr_fake.size(0)):
    hr_f = data_utils.de_transform(hr_fake[i])
    hr_f.save(f"./output/fake/{frm_idx_lst[i]}")

    if args.generate_bicubic:
        hr_b = data_utils.de_transform(hr_bic[i])
        hr_b.save(f"./output/bic/{frm_idx_lst[i]}")
