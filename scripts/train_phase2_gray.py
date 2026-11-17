"""
ALCLO Phase 2 entry point for the CRMark grayscale pretrained model.

This script reuses the Phase 2 training implementation in scripts/train_phase2.py
with conservative fine-tuning defaults and the architecture required by
crmark_gray_size_512_bit_256.pth:
    image size: 512
    channels:   1
    bits:       256
    min_size:   32

Quality-prioritized restart: lr=1e-6, pixel MSE weight=1000, message weight=10,
weaker adversarial noise/z regularization. Starts from the original pretrained
weights in a new phase2_gray_quality directory, without auto-resuming model_70.
The existing guardrail does not roll back weights or guarantee a PSNR floor.

Typical usage:
    python scripts/train_phase2_gray.py \
        --dataset_path ../../DIV2K/train \
        --val_dataset_path ../../DIV2K/val
"""

import argparse
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_phase2 import train  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def check_checkpoint_compat(args):
    if not os.path.exists(args.phase1_checkpoint):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {args.phase1_checkpoint}")

    ckpt = torch.load(args.phase1_checkpoint, map_location="cpu", weights_only=False)
    params = ckpt.get("param_dict", {}) or {}
    expected = {
        "img_size": args.im_size,
        "channel_dim": args.channel_dim,
        "bit_length": args.bit_length,
        "min_size": args.min_size,
        "fc": args.fc,
        "k": args.k,
    }
    mismatches = []
    for key, expected_value in expected.items():
        if key in params and params[key] != expected_value:
            mismatches.append(f"{key}: checkpoint={params[key]!r}, args={expected_value!r}")

    if mismatches:
        joined = "\n  ".join(mismatches)
        raise ValueError(
            "Checkpoint parameters do not match grayscale Phase 2 args:\n"
            f"  {joined}\n"
            "Use matching --im_size/--channel_dim/--bit_length/--min_size/--k/--fc values."
        )


def main():
    train_noise_dict = {
        "Jpeg": 50,
        "GaussianBlur": 1.5,
        "GaussianNoise": 0.05,
        "SaltPepperNoise": 0.15,
        "MedianFilter": 7,
        "Dropout": 0.3,
    }
    test_noise_dict = {
        "Jpeg": 50,
        "GaussianBlur": 3.0,
        "GaussianNoise": 0.3,
        "SaltPepperNoise": 0.3,
        "MedianFilter": 9,
        "Dropout": 0.5,
    }

    parser = argparse.ArgumentParser(
        description="ALCLO Phase 2 for CRMark grayscale 512x512 / 256-bit model"
    )

    parser.add_argument(
        "--phase1_checkpoint",
        type=str,
        default=str(REPO_ROOT / "pretrain" / "crmark_gray_size_512_bit_256.pth"),
        help="CRMark grayscale Phase 1/pretrained checkpoint.",
    )

    # Approximate baselines from frozen-iIWN epochs 1-10 in phase2_gray logs.
    # Replace with a fresh Phase 1 measurement on the chosen validation set.
    parser.add_argument("--baseline_psnr", type=float, default=36.3)
    parser.add_argument("--baseline_seen_acc", type=float, default=0.892)
    parser.add_argument("--psnr_tolerance", type=float, default=0.2)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, default=r"../../DIV2K/train")
    parser.add_argument("--val_dataset_path", type=str, default=r"../../DIV2K/val")
    parser.add_argument("--im_size", type=int, default=512)
    parser.add_argument("--channel_dim", type=int, default=1)
    parser.add_argument("--train_noise_dict", type=dict, default=train_noise_dict)
    parser.add_argument("--test_noise_dict", type=dict, default=test_noise_dict)

    parser.add_argument("--bit_length", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=32)
    parser.add_argument("--fc", type=str2bool, default=False)
    parser.add_argument("--hard_round", type=str2bool, default=True,
                        help="Native hard-round STE; match inference and remove extra rounding noise")

    parser.add_argument("--train_name", type=str, default="phase2_gray")
    parser.add_argument("--num_epoch", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--val_save_epoch", type=int, default=1)
    parser.add_argument("--queue_len", type=int, default=3)
    parser.add_argument("--continue_train", type=str2bool, default=False)

    parser.add_argument("--lambda_stego", type=float, default=1000.0,
                        help="Prioritize pixel fidelity over large message-loss-driven changes")
    parser.add_argument("--lambda_lpips", type=float, default=8.0)
    parser.add_argument("--lambda_secret_reset", type=float, default=10.0)
    parser.add_argument("--lambda_secret_floor", type=float, default=5.0)
    parser.add_argument("--lambda_penalty", type=float, default=1e6)
    parser.add_argument("--v", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.005)

    parser.add_argument("--lambda_z_mean", type=float, default=5e-5)
    parser.add_argument("--lambda_z_std", type=float, default=5e-4)
    parser.add_argument("--lambda_z_skew", type=float, default=1e-4)
    parser.add_argument("--zdist_warmup_epochs", type=int, default=20)
    parser.add_argument("--zdist_warmup_start_scale", type=float, default=0.1)

    parser.add_argument("--p_adv", type=float, default=0.15)
    parser.add_argument("--eps_min", type=float, default=1.0 / 255)
    parser.add_argument("--eps_max", type=float, default=4.0 / 255)
    parser.add_argument("--tau_psnr", type=float, default=38.0,
                        help="Attack distortion constraint; not a cover/stego PSNR target")
    parser.add_argument("--lambda_constraint", type=float, default=5.0)
    parser.add_argument("--atk_lr", type=float, default=1e-5)
    parser.add_argument("--attacker_warmup_epochs", type=int, default=0)
    parser.add_argument("--atk_hidden_channels", type=int, default=64)
    parser.add_argument("--atk_num_blocks", type=int, default=4)

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--logs_path", type=str, default="logs")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")

    check_checkpoint_compat(args)
    train(args)


if __name__ == "__main__":
    main()
