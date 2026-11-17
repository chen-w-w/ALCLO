"""
ALCLO Phase 3 entry point for grayscale 512x512 / 256-bit Phase 2 checkpoints.

This script reuses the Phase 3 score-network implementation in
scripts/train_phase3.py with grayscale handoff defaults. Only ScoreNetwork is
trained; the Phase 2 iIWN and attacker remain frozen. The default Phase 2
checkpoint is scripts/checkpoints/phase2_gray/model_best.pth.

Typical usage:
    python scripts/train_phase3_gray.py \
        --phase2_checkpoint scripts/checkpoints/phase2_gray/model_best.pth \
        --dataset_path /path/to/DIV2K/train \
        --val_dataset_path /path/to/DIV2K/val
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def noise_dict(value):
    """Accept JSON noise settings rather than argparse's unusable dict(str)."""
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Noise settings must be a JSON object") from error
    if not isinstance(result, dict) or not result:
        raise argparse.ArgumentTypeError("Noise settings must be a nonempty JSON object")
    for key, values in result.items():
        values = values if isinstance(values, list) else [values]
        if not values or not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in values):
            raise argparse.ArgumentTypeError(f"Invalid noise intensity for {key}")
    return result


def check_checkpoint_compat(args):
    if not os.path.isfile(args.phase2_checkpoint):
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {args.phase2_checkpoint}")

    try:
        ckpt = torch.load(args.phase2_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.phase2_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or not ckpt.get("model_state_dict"):
        raise ValueError("Phase 2 checkpoint must contain a nonempty model_state_dict")
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
            "Checkpoint parameters do not match grayscale Phase 3 args:\n"
            f"  {joined}\n"
            "Use matching --im_size/--channel_dim/--bit_length/--min_size/--k/--fc values."
        )

    if args.noise_mode == "hybrid" and not ckpt.get("attacker_state_dict"):
        raise ValueError("Hybrid training requires attacker_state_dict from the same Phase 2 checkpoint; "
                         "a Phase 1/pretrained checkpoint cannot supply it. Use --noise_mode fixed only intentionally.")

    # Match the shared resolver, but never fall back to the COLOR 12/255
    # setting when an older grayscale checkpoint does not record epsilon.
    source = "explicit --eps_max"
    if args.eps_max is None:
        saved = ckpt.get("current_eps_max", ckpt.get("eps"))
        if saved is None:
            args.eps_max, source = 4.0 / 255, "grayscale fallback (checkpoint has no epsilon)"
        else:
            args.eps_max = float(saved)
            source = "Phase 2 current_eps_max" if "current_eps_max" in ckpt else "Phase 2 eps"
    if not math.isfinite(args.eps_max) or not args.eps_min <= args.eps_max <= 1:
        raise ValueError("Expected finite 0 <= eps_min <= eps_max <= 1")
    if args.noise_mode == "fixed":
        args.val_restore_noise_mode = "fixed"
    return {
        "phase2_checkpoint": args.phase2_checkpoint,
        "epoch": ckpt.get("epoch", params.get("epoch")),
        "val_psnr": ckpt.get("val_psnr"),
        "val_acc": ckpt.get("val_acc"),
        "p_adv": args.p_adv,
        "eps_max": args.eps_max,
        "eps_source": source,
    }


def find_latest_score_checkpoint(directory):
    """The shared finder matches model_N.pth, but Phase 3 saves score_model_N."""
    candidates = []
    if Path(directory).is_dir():
        for path in Path(directory).iterdir():
            match = re.fullmatch(r"score_model_(\d+)\.pth", path.name)
            if path.is_file() and match:
                candidates.append((int(match.group(1)), str(path)))
    return max(candidates)[1] if candidates else None


def run_training(args):
    if args.continue_train and find_latest_score_checkpoint(Path(args.checkpoint_path) / args.train_name) is None:
        raise FileNotFoundError("--continue_train true requested, but no score_model_<epoch>.pth exists "
                                "in this experiment directory. Use false for a new run.")
    import train_phase3 as training
    # Adapt the shared trainer's resume dependency only for this gray call;
    # do not change the color script or compressor utilities on disk.
    original_finder = training.find_latest_model
    training.find_latest_model = find_latest_score_checkpoint
    try:
        training.train(args)
    finally:
        training.find_latest_model = original_finder


def build_parser():
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
        description="ALCLO Phase 3 score-network training for grayscale checkpoints"
    )

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    # Score training on 512x512 images is memory-heavy; defaults are smaller
    # than the color 256x256 entry point.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_path", type=str, default=str(REPO_ROOT.parent / "DIV2K" / "train"))
    parser.add_argument("--val_dataset_path", type=str, default=str(REPO_ROOT.parent / "DIV2K" / "val"))
    parser.add_argument("--im_size", type=int, default=512)
    parser.add_argument("--channel_dim", type=int, choices=[1], default=1)

    parser.add_argument("--train_noise_dict", type=noise_dict, default=train_noise_dict)
    parser.add_argument("--test_noise_dict", type=noise_dict, default=test_noise_dict)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--noise_mode", choices=["hybrid", "fixed"], default="hybrid")
    parser.add_argument(
        "--val_restore_noise_mode",
        choices=["hybrid", "fixed"],
        default="hybrid",
    )
    parser.add_argument("--p_adv", type=float, default=0.15,
                        help="Match the successful quality-prioritized Phase 2 grayscale run")
    parser.add_argument("--eps_min", type=float, default=1.0 / 255)
    parser.add_argument("--eps_max", type=float, default=None,
                        help="Inherit the selected Phase 2 checkpoint epsilon; fallback 4/255")
    parser.add_argument("--eps_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--atk_hidden_channels", type=int, default=64)
    parser.add_argument("--atk_num_blocks", type=int, default=4)

    parser.add_argument("--bit_length", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=32)
    parser.add_argument("--fc", type=str2bool, default=False)

    parser.add_argument(
        "--phase2_checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "checkpoints" / "phase2_gray" / "model_best.pth"),
        help="The same Phase 2 grayscale model_best.pth used for the satisfactory evaluation.",
    )

    parser.add_argument("--score_base_channels", type=int, default=32)
    parser.add_argument("--score_channel_mults", type=int, nargs="+", default=[1, 2, 4, 4])
    parser.add_argument("--score_embed_dim", type=int, default=256)

    parser.add_argument("--score_lr", type=float, default=1e-4)
    parser.add_argument("--score_min_lr", type=float, default=2e-6)
    parser.add_argument("--score_weight_decay", type=float, default=1e-5)

    parser.add_argument("--z_opt_steps", type=int, default=30)
    parser.add_argument("--z_opt_lr", type=float, default=0.03)
    parser.add_argument("--z_opt_sigma", type=float, default=0.01)
    parser.add_argument("--z_opt_lambda_prior", type=float, default=0.5)
    parser.add_argument("--z_opt_lambda_ent", type=float, default=0.3)
    parser.add_argument("--val_num_images", type=int, default=16)

    parser.add_argument("--train_name", type=str, default="phase3_gray_hybrid")
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--val_save_epoch", type=int, default=1)
    parser.add_argument("--val_z_opt_epoch", type=int, default=4)

    parser.add_argument("--checkpoint_path", type=str, default=str(SCRIPT_DIR / "checkpoints"))
    parser.add_argument("--logs_path", type=str, default=str(SCRIPT_DIR / "logs"))
    parser.add_argument("--continue_train", type=str2bool, default=False)
    parser.add_argument("--check_only", action="store_true",
                        help="Inspect checkpoint metadata/hybrid state and paths, without importing the training stack")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.p_adv) or not 0 <= args.p_adv <= 1 or not math.isfinite(args.eps_min) or not 0 <= args.eps_min <= 1:
        parser.error("p_adv and eps_min must be in [0,1]")
    for name in ("batch_size", "im_size", "bit_length", "min_size", "k", "num_epoch",
                 "val_save_epoch", "val_z_opt_epoch", "val_num_images", "z_opt_steps"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    for name in ("score_lr", "score_min_lr", "z_opt_lr", "z_opt_sigma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in ("score_weight_decay", "z_opt_lambda_prior", "z_opt_lambda_ent"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.score_min_lr > args.score_lr:
        parser.error("score_min_lr cannot exceed score_lr")
    if args.val_num_images < args.batch_size:
        parser.error("val_num_images must be >= batch_size (the shared validation loader drops partial batches)")
    if any(isinstance(value, list) for value in args.test_noise_dict.values()):
        parser.error("test_noise_dict uses one scalar intensity per attack; train_noise_dict may use lists")
    if not 0 <= args.seed < 2 ** 32 or args.gpu_id < 0:
        parser.error("Invalid seed or GPU id")
    for name in ("phase2_checkpoint", "dataset_path", "val_dataset_path", "checkpoint_path", "logs_path"):
        setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    try:
        metadata = check_checkpoint_compat(args)
        for path in (args.dataset_path, args.val_dataset_path):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Dataset directory not found: {path}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Phase 2 checkpoint: {metadata['phase2_checkpoint']}")
    print(f"  epoch={metadata['epoch']} | val_psnr={metadata['val_psnr']} | val_acc={metadata['val_acc']}")
    print(f"  p_adv={args.p_adv} | eps_max={args.eps_max:.8f} ({metadata['eps_source']})")
    print("Phase 3 trains ScoreNetwork only; the Phase 2 iIWN and attacker are frozen.")
    if args.check_only:
        print("Checkpoint metadata, hybrid state and dataset paths checked; no training was started.")
        return

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and (not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count()):
        parser.error("Requested CUDA device is unavailable")
    args.device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    # Help and checkpoint inspection do not need TensorBoard/LPIPS/noise packages.
    # Keep the shared color training implementation unchanged.
    run_training(args)


if __name__ == "__main__":
    main()
