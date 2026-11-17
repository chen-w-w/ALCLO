"""ALCLO Phase 2: joint latent regularization and adversarial training.

Start from Phase 1 weights, use moderate latent distribution penalties,
and warm up the attacker while freezing iIWN. Phase 3 learns corrections
for reconstructions from the resulting latent distribution.

If validation PSNR falls below baseline minus tolerance, reduce eps_max
by a factor of 0.9. Only epochs passing the PSNR check can become the
best checkpoint. Override experiment settings through command-line options;
see README.md for complete commands.
"""

import os
import sys
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crmark.nets import Model
from crmark.compressor.utils import (
    extract_accuracy, overflow_num, compute_psnr,
    quantize_image, find_latest_model
)
from alclo.data import HideImage
from alclo.adversary.attacker import AdversarialAttacker
from alclo.adversary.adv_noise_layer import HybridNoiseLayer, get_eps_schedule
from alclo.losses.z_distribution_loss import ZDistributionLoss
from alclo.losses.adversarial_loss import AdversarialLoss

from prettytable import PrettyTable
from watermarklab.noiselayers.noiselayerloader import DigitalDistortion

warnings.filterwarnings("ignore")


# ==================== Training steps ====================


def train_batch_iiwn(model, hybrid_noise, args, cover, secret, now_epoch,
                     z_dist_loss_fn, eps):
    """Train one iIWN batch with hybrid fixed and adversarial distortions."""
    stego, z = model.forward(cover, secret, args.hard_round, False)
    noised, _ = hybrid_noise(stego, cover, now_epoch, eps)

    rand_z = torch.randn_like(z)
    _, rec_secret = model.forward(noised, rand_z, args.hard_round, True)

    loss_penalty = model.PMSE(stego)
    loss_stego = model.MSE(stego, cover)
    loss_lpips = model.LLoss(stego, cover)
    loss_secret = model.MSE(rec_secret, secret)
    loss_z_dist, z_stats = z_dist_loss_fn(z)

    loss_iiwn = (args.lambda_penalty * loss_penalty
                 + args.lambda_stego * loss_stego
                 + args.lambda_lpips * loss_lpips
                 + args.lambda_secret * loss_secret
                 + loss_z_dist)

    return {
        "loss": loss_iiwn,
        "values": {
            "train_total_loss": loss_iiwn.item(),
            "train_accuracy": extract_accuracy(rec_secret, secret),
            "train_lambda_secret": args.lambda_secret,
            "train_loss_stego": loss_stego.item(),
            "train_loss_lpips": loss_lpips.item(),
            "train_loss_secret": loss_secret.item(),
            "train_loss_penalty": loss_penalty.item(),
            "train_loss_z_dist": loss_z_dist.item(),
            "train_stego_psnr": compute_psnr(stego, cover),
            "train_overflow_0": overflow_num(stego, 0),
            "train_overflow_255": overflow_num(stego, 255),
            "train_eps": eps,
            **{f"train_{k}": v for k, v in z_stats.items()},
        }
    }


def train_batch_attacker(model, attacker, args, cover, secret, adv_loss_fn, eps):
    """Train the attacker on a separate computation graph with iIWN frozen."""
    with torch.no_grad():
        stego, z = model.forward(cover, secret, True, False)
    stego_detached = stego.detach()

    delta = attacker(stego_detached, eps)
    noised_adv = torch.clamp(stego_detached + delta, 0.0, 1.0)

    for p in model.parameters():
        p.requires_grad_(False)

    rand_z = torch.randn(z.shape, device=z.device)
    _, w_hat_adv = model.forward(noised_adv, rand_z, args.hard_round, True)

    for p in model.parameters():
        p.requires_grad_(True)

    loss_atk, atk_stats = adv_loss_fn(w_hat_adv, secret, stego_detached, noised_adv)

    return {
        "loss": loss_atk,
        "values": atk_stats,
    }


def val_noise_batch(model, args, noise_layer, intensity, val_cover, val_secret):
    """Validate one batch under the specified distortion."""
    val_stego, val_z = model.forward(val_cover, val_secret, True, False)
    val_noised = noise_layer.test(quantize_image(val_stego), val_cover, intensity)
    val_noised_quant = quantize_image(val_noised)
    _, val_rec_secret = model.forward(val_noised_quant, torch.randn_like(val_z), True, True)
    return {
        "val_accuracy": extract_accuracy(val_rec_secret, val_secret),
        "val_stego_psnr": compute_psnr(val_stego, val_cover),
    }


def run_validation(model, val_loader, test_noiselayer, args):
    """Measure clean stego PSNR and extraction accuracy under each validation noise."""
    model.eval()
    with torch.no_grad():
        # 1. Measure clean stego PSNR.
        psnr_list = []
        for val_cover, val_secret in val_loader:
            val_cover = val_cover.to(args.device)
            val_secret = val_secret.to(args.device)
            val_stego, _ = model.forward(val_cover, val_secret, True, False)
            psnr_list.append(compute_psnr(val_stego, val_cover))
        avg_psnr = float(np.mean(psnr_list))

        # 2. Measure accuracy under each distortion.
        acc_dict = {}
        for key in test_noiselayer.noise_dict.keys():
            acc_per_noise = []
            for val_cover, val_secret in val_loader:
                val_cover = val_cover.to(args.device)
                val_secret = val_secret.to(args.device)
                vr = val_noise_batch(
                    model, args, test_noiselayer.noise_layers[key],
                    test_noiselayer.noise_dict[key], val_cover, val_secret,
                )
                acc_per_noise.append(vr["val_accuracy"])
            acc_dict[key] = float(np.mean(acc_per_noise))

        avg_acc = float(np.mean(list(acc_dict.values())))
    model.train()
    return avg_psnr, avg_acc, acc_dict


# ==================== Training loop ====================


def train(args):
    torch.manual_seed(args.seed)

    table = PrettyTable(["Argument", "Value"])
    for k, v in vars(args).items():
        table.add_row([k, v])
    print(table)

    log_path = os.path.join(args.logs_path, args.train_name)
    ckpt_dir = os.path.join(args.checkpoint_path, args.train_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_path)

    # Record the configuration as a Markdown table in TensorBoard.
    args_md_lines = ["| Argument | Value |", "| --- | --- |"]
    for k, v in vars(args).items():
        args_md_lines.append(f"| {k} | {v} |")
    writer.add_text("args", "\n".join(args_md_lines), 0)

    # ---- Create iIWN and load Phase 1 weights ----
    model = Model(
        img_size=args.im_size, channel_dim=args.channel_dim,
        bit_length=args.bit_length, k=args.k,
        min_size=args.min_size, fc=args.fc,
    ).to(args.device)

    # Skip Phase 1 loading when resuming; the latest checkpoint restores all state below.
    latest_ckpt_path = find_latest_model(ckpt_dir) if args.continue_train else None
    if latest_ckpt_path is None:
        print(f"\nLoading Phase 1 checkpoint: {args.phase1_checkpoint}")
        upstream_dict = torch.load(args.phase1_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(upstream_dict['model_state_dict'])
        args.lambda_secret = args.lambda_secret_reset
        print(f"Reset lambda_secret = {args.lambda_secret} (Phase 2 initialization)")
    else:
        print(f"\nLatest checkpoint found: {latest_ckpt_path}; skipping Phase 1 loading")

    # ---- Distortion layers ----
    train_noiselayer = DigitalDistortion(noise_dict=args.train_noise_dict, max_step=args.max_step)
    test_noiselayer = DigitalDistortion(noise_dict=args.test_noise_dict, max_step=args.max_step)

    # ---- attacker (v1: 4-ResBlock CNN) ----
    attacker = AdversarialAttacker(
        in_channels=args.channel_dim,
        hidden_channels=args.atk_hidden_channels,
        num_blocks=args.atk_num_blocks,
    ).to(args.device)

    iiwn_params = sum(p.numel() for p in model.parameters())
    atk_params = sum(p.numel() for p in attacker.parameters())
    print(f"iIWN: {iiwn_params / 1e6:.2f}M, attacker: {atk_params / 1e6:.2f}M")

    hybrid_noise = HybridNoiseLayer(
        attacker=attacker,
        fixed_noise_layer=train_noiselayer,
        p_adv=args.p_adv,
    ).to(args.device)
    adv_loss_fn = AdversarialLoss(
        tau_psnr=args.tau_psnr,
        lambda_constraint=args.lambda_constraint,
    )
    optim_atk = Adam(attacker.parameters(), lr=args.atk_lr, betas=(0.9, 0.999))

    # Target weights used to scale the warmup schedule.
    target_lambda_z_mean = args.lambda_z_mean
    target_lambda_z_std = args.lambda_z_std
    target_lambda_z_skew = args.lambda_z_skew
    z_dist_loss_fn = ZDistributionLoss(
        lambda_mean=target_lambda_z_mean,
        lambda_std=target_lambda_z_std,
        lambda_skew=target_lambda_z_skew,
    )

    optim_iiwn = AdamW(model.inn_blocks.parameters(), lr=args.lr, betas=(0.5, 0.999),
                       eps=1e-6, weight_decay=1e-5)

    # ---- Image datasets ----
    train_dataset = HideImage(args.dataset_path, args.im_size, args.bit_length, args.channel_dim)
    val_dataset = HideImage(args.val_dataset_path, args.im_size, args.bit_length, args.channel_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, pin_memory=True)

    # ---- Checkpoint resume ----
    start_epoch = 0
    global_step = 0
    resumed_eps_max = None
    if latest_ckpt_path:
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'attacker_state_dict' in ckpt:
            attacker.load_state_dict(ckpt['attacker_state_dict'])
            if 'optim_atk_state_dict' in ckpt:
                optim_atk.load_state_dict(ckpt['optim_atk_state_dict'])
        optim_iiwn.load_state_dict(ckpt['optim_iiwn_state_dict'])
        start_epoch = ckpt['epoch']
        global_step = ckpt['global_step']
        args.lambda_secret = ckpt['lambda_secret']
        resumed_eps_max = ckpt.get('current_eps_max', None)
        print(f"Resuming from epoch {start_epoch} (lambda_secret={args.lambda_secret:.4f}"
              + (f", current_eps_max={resumed_eps_max:.4f}" if resumed_eps_max is not None else "")
              + ")")

    # ---- Training state ----
    model.train()
    attacker.train()

    average_acc_queue = []
    best_acc_in_safe_zone = -float('inf')
    best_epoch = -1
    current_eps_max = resumed_eps_max if resumed_eps_max is not None else args.eps_max  # The PSNR guardrail can reduce this value dynamically.

    atk_warmup = max(int(args.attacker_warmup_epochs), 0)

    remaining = max(args.num_epoch - start_epoch, 0)
    for epoch in tqdm(range(remaining), position=0, desc="Phase2", ncols=100):
        now_epoch = epoch + start_epoch + 1

        # During attacker warmup, freeze iIWN and train only the attacker.
        in_atk_warmup = (epoch < atk_warmup)

        if in_atk_warmup:
            eps = current_eps_max
        else:
            sched_epoch = max(epoch - atk_warmup, 0)
            sched_total = max(args.num_epoch - atk_warmup, 1)
            eps = get_eps_schedule(sched_epoch, sched_total, args.eps_min, current_eps_max)
        writer.add_scalar("train/in_atk_warmup", float(in_atk_warmup), now_epoch)

        # Linearly ramp lambda_z_* from start_scale to full weight during latent warmup.
        warmup_ep = max(args.zdist_warmup_epochs, 0)
        if warmup_ep > 0:
            progress = min(epoch / warmup_ep, 1.0)
            scale = args.zdist_warmup_start_scale + (1.0 - args.zdist_warmup_start_scale) * progress
        else:
            scale = 1.0
        z_dist_loss_fn.lambda_mean = target_lambda_z_mean * scale
        z_dist_loss_fn.lambda_std = target_lambda_z_std * scale
        z_dist_loss_fn.lambda_skew = target_lambda_z_skew * scale
        writer.add_scalar("train/zdist_warmup_scale", scale, now_epoch)

        # Adaptively decay lambda_secret.
        if len(average_acc_queue) >= args.queue_len:
            if np.mean(average_acc_queue) > 1. - args.delta:
                new_val = args.lambda_secret * args.v
                if new_val >= args.lambda_secret_floor:
                    args.lambda_secret = new_val
                    average_acc_queue.clear()
                    print(f"lambda_secret decayed to: {args.lambda_secret:.4f}")

        loss_list = []
        acc_list = []

        tqdm_iter = tqdm(iter(train_loader), position=1, desc="Iter", ncols=140)
        for cover, secret in tqdm_iter:
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # ---- Update iIWN, except during attacker warmup ----
            if in_atk_warmup:
                with torch.no_grad():
                    iiwn_result = train_batch_iiwn(
                        model, hybrid_noise, args, cover, secret, now_epoch,
                        z_dist_loss_fn, eps,
                    )
                iiwn_result["loss"] = None
            else:
                iiwn_result = train_batch_iiwn(
                    model, hybrid_noise, args, cover, secret, now_epoch,
                    z_dist_loss_fn, eps,
                )
                optim_iiwn.zero_grad()
                iiwn_result["loss"].backward()
                optim_iiwn.step()

            # ---- Update the attacker on every batch ----
            atk_result = train_batch_attacker(
                model, attacker, args, cover, secret, adv_loss_fn, eps,
            )
            optim_atk.zero_grad()
            atk_result["loss"].backward()
            optim_atk.step()

            global_step += 1
            loss_list.append(iiwn_result["values"]["train_total_loss"])
            acc_list.append(iiwn_result["values"]["train_accuracy"])

            desc = (f"E{now_epoch} loss={np.mean(loss_list):.4f} "
                    f"acc={np.mean(acc_list):.4f} eps={eps:.4f} "
                    f"atk_err={atk_result['values']['attacker_watermark_error']:.4f}")
            tqdm_iter.set_description(desc)

            if global_step % 50 == 0:
                for k, v in iiwn_result["values"].items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                for k, v in atk_result["values"].items():
                    writer.add_scalar(f"attacker/{k}", v, global_step)

        average_acc_queue.append(float(np.mean(acc_list)))
        if len(average_acc_queue) > args.queue_len:
            average_acc_queue.pop(0)

        # ---- Validation, checkpointing, and PSNR guardrail ----
        if (now_epoch - 1) % args.val_save_epoch == 0:
            print(f"\nValidating epoch {now_epoch}...")
            avg_val_psnr, avg_val_acc, acc_dict = run_validation(
                model, val_loader, test_noiselayer, args,
            )

            for k, v in acc_dict.items():
                writer.add_scalar(f"val_accuracy/{k}", v, now_epoch)
            writer.add_scalar("val_accuracy/average", avg_val_acc, now_epoch)
            writer.add_scalar("val/psnr", avg_val_psnr, now_epoch)
            writer.add_scalar("val/eps_max_current", current_eps_max, now_epoch)

            psnr_delta = avg_val_psnr - args.baseline_psnr
            acc_delta = avg_val_acc - args.baseline_seen_acc
            psnr_ok = avg_val_psnr >= args.baseline_psnr
            psnr_mark = "[OK]" if psnr_ok else "[LOW]"
            acc_mark = "[OK]" if avg_val_acc >= args.baseline_seen_acc else "[note]"
            print(f"  val_psnr = {avg_val_psnr:.2f}  ({psnr_delta:+.2f} vs baseline {args.baseline_psnr:.2f}) {psnr_mark}")
            print(f"  val_acc  = {avg_val_acc:.4f} ({acc_delta:+.4f} vs baseline {args.baseline_seen_acc:.4f}) {acc_mark}")
            for k, v in acc_dict.items():
                print(f"    {k:<20} {v:.4f}")

            # Reduce eps_max when PSNR drops below the permitted threshold.
            if avg_val_psnr < args.baseline_psnr - args.psnr_tolerance:
                old_eps = current_eps_max
                current_eps_max = max(current_eps_max * 0.9, args.eps_min)
                print(f"  [GUARDRAIL] PSNR below baseline - {args.psnr_tolerance:.2f}, "
                      f"eps_max {old_eps:.4f} -> {current_eps_max:.4f}")

            # Only epochs passing the PSNR check can become the best checkpoint.
            score = avg_val_acc + 0.005 * avg_val_psnr
            writer.add_scalar("val/score", score, now_epoch)
            writer.add_scalar("val/psnr_ok", float(psnr_ok), now_epoch)

            if psnr_ok and avg_val_acc > best_acc_in_safe_zone:
                best_acc_in_safe_zone = avg_val_acc
                best_epoch = now_epoch
                print(f"  [BEST] New best eligible epoch (acc={avg_val_acc:.4f})")

            # Save the checkpoint.
            save_dict = {
                'model_state_dict': model.state_dict(),
                'attacker_state_dict': attacker.state_dict(),
                'optim_iiwn_state_dict': optim_iiwn.state_dict(),
                'optim_atk_state_dict': optim_atk.state_dict(),
                'epoch': now_epoch,
                'global_step': global_step,
                'lambda_secret': args.lambda_secret,
                'eps': eps,
                'current_eps_max': current_eps_max,
                'val_psnr': avg_val_psnr,
                'val_acc': avg_val_acc,
                'acc_per_noise': acc_dict,
                'param_dict': {
                    'bit_length': args.bit_length,
                    'img_size': args.im_size,
                    'channel_dim': args.channel_dim,
                    'min_size': args.min_size,
                    'fc': args.fc,
                    'k': args.k,
                },
            }

            save_path = os.path.join(ckpt_dir, f"model_{now_epoch}.pth")
            torch.save(save_dict, save_path)
            print(f"  Checkpoint saved: {save_path}\n")

            # Also save the best checkpoint when eligible.
            if best_epoch == now_epoch:
                best_path = os.path.join(ckpt_dir, "model_best.pth")
                torch.save(save_dict, best_path)
                writer.add_scalar("val/best_val_acc", best_acc_in_safe_zone, now_epoch)

    writer.close()
    print("\nPhase 2 training complete.")
    if best_epoch > 0:
        print(f"Best eligible epoch: {best_epoch} (acc={best_acc_in_safe_zone:.4f})")
        print(f"Best checkpoint: {os.path.join(ckpt_dir, 'model_best.pth')}")
    else:
        print("Warning: no epoch reached PSNR >= baseline.")
        print("         Inspect TensorBoard train/train_stego_psnr and val/psnr curves.")
        print("         Consider reducing lr, lambda_z_std, or the adversarial strength.")


# ==================== Argument configuration ====================


def main():
    # Use training/validation distortion dictionaries consistent with Phase 1.
    train_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 1.5, "GaussianNoise": 0.15,
        "SaltPepperNoise": 0.15, "MedianFilter": 7, "Dropout": 0.3,
    }
    test_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 3., "GaussianNoise": 0.3,
        "SaltPepperNoise": 0.3, "MedianFilter": 9, "Dropout": 0.5,
    }

    parser = argparse.ArgumentParser(
        description="ALCLO Phase 2: joint latent regularization and adversarial fine-tuning"
    )

    # ---- Input checkpoint ----
    parser.add_argument("--phase1_checkpoint", type=str,
                        default="../pretrain/crmark_color_size_256_bit_64.pth",
                        help="Phase 1 checkpoint used to initialize training")

    # ---- Baseline measurements for the PSNR guardrail ----
    parser.add_argument("--baseline_psnr", type=float, default=40.0,
                        help="Measured Phase 1 validation PSNR on your validation set")
    parser.add_argument("--baseline_seen_acc", type=float, default=0.980,
                        help="Mean Phase 1 seen-attack accuracy, for display only; the guardrail uses PSNR")
    parser.add_argument("--psnr_tolerance", type=float, default=2.0,
                        help="PSNR tolerance: reduce eps_max when val_psnr < baseline - tolerance")

    # ---- Device settings ----
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # ---- Dataset settings ----
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default=r"../../DIV2K/train")
    parser.add_argument("--val_dataset_path", type=str, default=r"../../DIV2K/val")
    parser.add_argument("--im_size", type=int, default=256)
    parser.add_argument("--channel_dim", type=int, default=3)
    parser.add_argument("--train_noise_dict", type=dict, default=train_noise_dict)
    parser.add_argument("--test_noise_dict", type=dict, default=test_noise_dict)

    # ---- iIWN ----
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--fc", type=bool, default=False)
    parser.add_argument("--hard_round", type=bool, default=False)

    # ---- Training loop settings ----
    parser.add_argument("--train_name", type=str, default="phase2_C2_eps_20_255）",
                        help="Checkpoint and TensorBoard experiment directory name")
    parser.add_argument("--num_epoch", type=int, default=70)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--val_save_epoch", type=int, default=1)
    parser.add_argument("--queue_len", type=int, default=3)
    parser.add_argument("--continue_train", type=bool, default=False)

    # ---- iIWN loss weights ----
    parser.add_argument("--lambda_stego", type=float, default=2.0)
    parser.add_argument("--lambda_lpips", type=float, default=8.0)
    parser.add_argument("--lambda_secret_reset", type=float, default=1e2,
                        help="Reset value of lambda_secret at Phase 2 initialization")
    parser.add_argument("--lambda_secret_floor", type=float, default=10.0)
    parser.add_argument("--lambda_penalty", type=float, default=1e6)
    parser.add_argument("--v", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.005)

    # ---- Latent distribution regularization (R0 baseline) ----
    parser.add_argument("--lambda_z_mean", type=float, default=5e-4)
    parser.add_argument("--lambda_z_std", type=float, default=5e-3,
                        help="Sensitivity setting 1 (group B): 1e-3 / 5e-3 / 5e-2")
    parser.add_argument("--lambda_z_skew", type=float, default=1e-3)
    parser.add_argument("--zdist_warmup_epochs", type=int, default=10)
    parser.add_argument("--zdist_warmup_start_scale", type=float, default=0.1)
    # ---- Adversarial training (R0 baseline, residual CNN attacker) ----
    parser.add_argument("--p_adv", type=float, default=0.35,
                        help="Sensitivity setting 2 (group A): 0.15 / 0.35 / 0.55")
    parser.add_argument("--eps_min", type=float, default=2.0 / 255)
    parser.add_argument("--eps_max", type=float, default=20.0 / 255,
                        help="Sensitivity setting 3 (group C): 6/255 / 12/255 / 20/255")
    parser.add_argument("--tau_psnr", type=float, default=32.0)
    parser.add_argument("--lambda_constraint", type=float, default=5.0)
    parser.add_argument("--atk_lr", type=float, default=8e-5,
                        help="Attacker learning rate (8e-5 for the residual CNN configuration)")
    parser.add_argument("--attacker_warmup_epochs", type=int, default=5,
                        help="Freeze iIWN and train only the attacker for the first N epochs")
    parser.add_argument("--atk_hidden_channels", type=int, default=64)
    parser.add_argument("--atk_num_blocks", type=int, default=4)

    # ---- Output paths ----
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--logs_path", type=str, default="logs")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")

    train(args)


if __name__ == "__main__":
    main()
