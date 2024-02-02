import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from lightning import Trainer

from archs.align_net import AlignNet
from archs.patch_vsr_net import PatchVSRNet
from data.datamodule import VideoFolderDataModule
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="Path to lightning model checkpoint",
)
parser.add_argument(
    "-f",
    "--folder_name",
    type=str,
    help="Folder with sequences to align (folder with folders with frames)",
)
parser.add_argument(
    "-s",
    "--seq",
    type=str,
    default="",
    help="Sequence to align (folder with frames)",
)
parser.add_argument(
    "-t",
    "--tempo_extent",
    type=int,
    default=5,
    help="Number of adiacent frames to use",
)
parser.add_argument(
    "-b",
    "--block_size",
    type=int,
    default=16,
    help="",
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

args = parser.parse_args()

dm = VideoFolderDataModule(
    hr_path=args.folder_name,
    hr_path_filter=args.seq,
    tempo_extent=args.tempo_extent,
)

checkpoint = torch.load(args.ckpt)

aligner = AlignNet(
    in_channels=3,
    top_k=1,
    block_size=16,
    stride=8,
)
model = PatchVSRNet(
    scale_factor=args.scale,
    align_net=aligner,
)

state_dict = {
    ".".join(k.split(".")[1:]): v
    for k, v in checkpoint["state_dict"].items()
    if "G." in k
}
model.load_state_dict(state_dict)

model.freeze()
trainer = Trainer(devices=[0])
pred = trainer.predict(model, dm)

to_image = torchvision.transforms.ToPILImage()

print("Saving upscaled sequence...")
os.makedirs(f"./{args.output_dir}/fake/", exist_ok=True)

frm_idx_lst = ["{:04d}.png".format(i) for i in range(len(pred))]
for i in range(len(pred)):
    hr_f = data_utils.de_transform(pred[i][0].squeeze(0))
    hr_f.save(f"./{args.output_dir}/fake/{frm_idx_lst[i]}")
