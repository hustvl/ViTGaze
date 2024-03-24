# Convert official model weights to format that d2 receives
import argparse
from collections import OrderedDict
import torch


def convert(src: str, dst: str):
    checkpoint = torch.load(src)
    has_model = "model" in checkpoint.keys()
    checkpoint = checkpoint["model"] if has_model else checkpoint
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    out_cp = OrderedDict()
    for k, v in checkpoint.items():
        out_cp[".".join(["backbone", k])] = v
    out_cp = {"model": out_cp} if has_model else out_cp
    torch.save(out_cp, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", "-s", type=str, required=True, help="Path to src weights.pth"
    )
    parser.add_argument(
        "--dst", "-d", type=str, required=True, help="Path to dst weights.pth"
    )
    args = parser.parse_args()
    convert(
        args.src,
        args.dst,
    )
