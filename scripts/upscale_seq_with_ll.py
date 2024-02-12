import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from lightning import Trainer
from torch.utils.data import DataLoader

from archs.align_net import AlignNet
from archs.patch_vsr_net import PatchVSRNet
from data.datamodule import VideoFolderDataModule
from data.folder_dataset import VideoFolderPaired
from models.vsr_with_ll_single_module import VSRSingle
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="Path to lightning model checkpoint",
)
parser.add_argument(
    "-f1",
    "--folder_name1",
    type=str,
    help="Folder with sequences to upscale (folder with folders with frames)",
)
parser.add_argument(
    "-s1",
    "--seq1",
    type=str,
    default="",
    help="Sequence to upscale (folder with frames)",
)
parser.add_argument(
    "-f2",
    "--folder_name2",
    type=str,
    help="Folder with ground truths (folder with folders with frames)",
)
parser.add_argument(
    "-s2",
    "--seq2",
    type=str,
    default="",
    help="Ground truth sequence (folder with frames)",
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
    "-d",
    "--device",
    type=int,
    default=0,
    help="Device number to use for computations",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Output directory",
)

args = parser.parse_args()

dataset = VideoFolderPaired(
    hr_path=args.folder_name2,
    lr_path=args.folder_name1,
    hr_path_filter=args.seq2,
    lr_path_filter=args.seq1,
    tempo_extent=args.tempo_extent,
)
data_loader_predict = DataLoader(
    dataset=dataset,
    num_workers=20,
    shuffle=False,
    pin_memory=True,
)

checkpoint = torch.load(args.ckpt, map_location=f"cuda:{args.device}")

aligner = AlignNet(
    in_channels=3,
    top_k=1,
    block_size=16,
    stride=8,
)
generator = PatchVSRNet(
    scale_factor=args.scale,
    align_net=aligner,
)

state_dict = {
    ".".join(k.split(".")[1:]): v
    for k, v in checkpoint["state_dict"].items()
    if "G." in k
}
generator.load_state_dict(state_dict)

model = VSRSingle(generator, losses={})

model.freeze()
trainer = Trainer(devices=[args.device])
pred = trainer.predict(model, data_loader_predict)

print("Saving upscaled sequence...")
os.makedirs(f"./{args.output_dir}/fake/", exist_ok=True)

frm_idx_lst = ["{:04d}.png".format(i) for i in range(len(pred))]
for i in range(len(pred)):
    hr_f = data_utils.de_transform(pred[i][0].squeeze(0))
    hr_f.save(f"./{args.output_dir}/fake/{frm_idx_lst[i]}")
