"""
ALCLO Phase 3 训练脚本: Score Network 训练 (离线)

Phase 2 的 iIWN 参数完全冻结，本阶段只训练 ScoreNetwork。
训练数据在线生成: 冻结的 iIWN 对训练图像做正向+噪声攻击+逆向，得到 I_rec。
ScoreNetwork 学习从 I_rec 预测朝 I_cover 修正的方向。

训练循环 (每个 batch):
    ┌─────────────────────────────────────────────────┐
    │ 1. 加载 cover images + 随机 watermark            │
    │ 2. iIWN 正向: (cover, w) → (stego, z)           │
    │ 3. 固定/混合噪声: stego → noised                 │
    │ 4. iIWN 逆向: (noised, z_rand) → (I_rec, w_hat) │
    │ 5. correction = ScoreNetwork(I_rec, sigma)       │
    │ 6. I_pred = I_rec + correction                   │
    │ 7. 用 cover 监督 I_pred                          │
    └─────────────────────────────────────────────────┘

用法:
    python scripts/train_phase3.py \
        --phase2_checkpoint checkpoints/phase2_R0_baseline/model_63.pth \
        --dataset_path ../../DIV2K/train \
        --val_dataset_path ../../DIV2K/val
"""

import os
import sys
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crmark.nets import Model
from crmark.compressor.utils import (
    extract_accuracy, quantize_image, find_latest_model
)
from alclo.data import HideImage
from alclo.adversary.attacker import AdversarialAttacker
from alclo.adversary.adv_noise_layer import HybridNoiseLayer, get_eps_schedule
from alclo.score.score_network import ScoreNetwork, restoration_loss
from alclo.score.z_optimizer import ZOptimizer

from prettytable import PrettyTable
from watermarklab.noiselayers.noiselayerloader import DigitalDistortion

warnings.filterwarnings("ignore")


# ==================== 工具函数 / 噪声层 ====================


def freeze_module(module):
    """冻结模块参数并切到 eval 模式。"""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def resolve_eps_max(args, phase2_dict):
    """优先复用 Phase 2 守门后保存的 eps；没有则用保守默认值。"""
    if args.eps_max is not None:
        return
    ckpt_eps = phase2_dict.get('current_eps_max', phase2_dict.get('eps', None))
    args.eps_max = float(ckpt_eps) if ckpt_eps is not None else 12.0 / 255.0


def prepare_runtime_args(args, phase2_dict):
    """根据 Phase 2 checkpoint 补齐运行期参数。"""
    resolve_eps_max(args, phase2_dict)
    if args.noise_mode != "hybrid":
        args.val_restore_noise_mode = "fixed"


def current_adv_eps(args, now_epoch):
    """Phase 3 对抗扰动强度；默认使用 Phase 2 最终 eps。"""
    if args.eps_schedule == "constant":
        return args.eps_max
    return get_eps_schedule(max(now_epoch - 1, 0), args.num_epoch,
                            args.eps_min, args.eps_max)


def scalarize_noise_dict(noise_dict):
    """DigitalDistortion 初始化需要标量强度；多强度时取第一个占位。"""
    return {
        key: (value[0] if isinstance(value, (list, tuple)) else value)
        for key, value in noise_dict.items()
    }


class RandomStrengthDigitalDistortion(nn.Module):
    """DigitalDistortion 包装器: 随机选择噪声类型和该类型下的强度。"""

    def __init__(self, noise_dict, max_step=1):
        super().__init__()
        self.noise_dict = noise_dict
        self.base = DigitalDistortion(
            noise_dict=scalarize_noise_dict(noise_dict),
            max_step=max_step,
        )
        self.noise_layers = self.base.noise_layers

    def _sample_key_intensity(self):
        keys = list(self.noise_dict.keys())
        key = keys[torch.randint(len(keys), (1,)).item()]
        values = self.noise_dict[key]
        if isinstance(values, (list, tuple)):
            intensity = values[torch.randint(len(values), (1,)).item()]
        else:
            intensity = values
        return key, intensity

    def forward(self, stego, cover, now_step=None):
        key, intensity = self._sample_key_intensity()
        return self.noise_layers[key].test(stego, cover, intensity)


def build_attacker(args, phase2_dict):
    """从 Phase 2 checkpoint 恢复并冻结 attacker。"""
    if 'attacker_state_dict' not in phase2_dict:
        raise KeyError("Phase 2 checkpoint 缺少 attacker_state_dict，无法启用 hybrid 噪声池。")

    attacker = AdversarialAttacker(
        in_channels=args.channel_dim,
        hidden_channels=args.atk_hidden_channels,
        num_blocks=args.atk_num_blocks,
    ).to(args.device)
    attacker.load_state_dict(phase2_dict['attacker_state_dict'])
    freeze_module(attacker)
    return attacker


def build_noise_layers(args, phase2_dict):
    """构造训练、restore 验证、z-opt 验证三套噪声入口。"""
    train_fixed = RandomStrengthDigitalDistortion(
        noise_dict=args.train_noise_dict, max_step=args.max_step)
    test_fixed = DigitalDistortion(
        noise_dict=args.test_noise_dict, max_step=args.max_step)

    if args.noise_mode != "hybrid":
        print("使用固定 DigitalDistortion 噪声池。")
        return train_fixed, test_fixed, test_fixed

    attacker = build_attacker(args, phase2_dict)
    train_layer = HybridNoiseLayer(
        attacker=attacker,
        fixed_noise_layer=train_fixed,
        p_adv=args.p_adv,
    ).to(args.device)
    val_layer = test_fixed
    if args.val_restore_noise_mode == "hybrid":
        val_layer = HybridNoiseLayer(
            attacker=attacker,
            fixed_noise_layer=test_fixed,
            p_adv=args.p_adv,
        ).to(args.device)

    print(f"Hybrid 噪声池已启用: p_adv={args.p_adv}, "
          f"eps=[{args.eps_min:.6f}, {args.eps_max:.6f}], "
          f"schedule={args.eps_schedule}")
    return train_layer, val_layer, test_fixed


def apply_noise(noise_layer, stego_quant, cover, args, now_epoch):
    """支持固定噪声池和 Phase 2 混合噪声池两种输入分布。"""
    if isinstance(noise_layer, HybridNoiseLayer):
        eps = current_adv_eps(args, now_epoch)
        noised, is_adv = noise_layer(stego_quant, cover, now_epoch, eps)
        stats = {
            'noise_adv_ratio': is_adv.float().mean().item(),
            'noise_eps': eps,
        }
        return noised, stats

    if isinstance(noise_layer, RandomStrengthDigitalDistortion):
        noised = noise_layer(stego_quant, cover, now_epoch)
        return noised, {'noise_adv_ratio': 0.0, 'noise_eps': 0.0}

    noise_keys = list(noise_layer.noise_dict.keys())
    key = noise_keys[torch.randint(len(noise_keys), (1,)).item()]
    intensity = noise_layer.noise_dict[key]
    noised = noise_layer.noise_layers[key].test(stego_quant, cover, intensity)
    return noised, {'noise_adv_ratio': 0.0, 'noise_eps': 0.0}


def generate_reconstruction(model, noise_layer, cover, secret, args, now_epoch):
    """用冻结的 iIWN 在线生成 I_rec，并返回对应 cover 作为监督目标。

    流程: cover+w → iIWN正向 → stego量化 → 噪声池 → noised量化 → iIWN逆向 → I_rec
    """
    with torch.no_grad():
        stego, z = model.forward(cover, secret, True, False)
        stego_quant = quantize_image(stego)

        noised, noise_stats = apply_noise(
            noise_layer, stego_quant, cover, args, now_epoch)
        noised_quant = quantize_image(noised)

        z_rand = torch.randn_like(z)
        I_rec, _ = model.forward(noised_quant, z_rand, True, True)

    return I_rec.detach(), cover.detach(), noise_stats


# ==================== 验证: z 优化效果 ====================


def validate_z_optimization(model, score_net, noise_layer, val_loader, args):
    """验证 z 优化的效果: 对比 random-z 和 optimized-z 的提取精度。"""
    z_optimizer = ZOptimizer(
        iiwn_model=model,
        score_network=score_net,
        bit_length=args.bit_length,
        num_steps=args.z_opt_steps,
        lr=args.z_opt_lr,
        sigma=args.z_opt_sigma,
        lambda_prior=args.z_opt_lambda_prior,
        lambda_ent=args.z_opt_lambda_ent,
        device=args.device,
    )

    results = {}
    noise_keys = list(noise_layer.noise_dict.keys())

    for key in noise_keys:
        acc_baseline_list = []
        acc_optimized_list = []
        intensity = noise_layer.noise_dict[key]

        count = 0
        for val_cover, val_secret in val_loader:
            val_cover = val_cover.to(args.device)
            val_secret = val_secret.to(args.device)

            with torch.no_grad():
                stego, z = model.forward(val_cover, val_secret, True, False)
                stego_quant = quantize_image(stego)
                noised = noise_layer.noise_layers[key].test(
                    stego_quant, val_cover, intensity)
                noised_quant = quantize_image(noised)

            # 逐张处理 (z 优化是单张的)
            B = noised_quant.shape[0]
            for i in range(B):
                noised_i = noised_quant[i:i + 1]
                secret_i = val_secret[i:i + 1]

                # Baseline: random z
                wm_baseline = z_optimizer.extract_baseline(noised_i)
                acc_base = extract_accuracy(wm_baseline, secret_i)
                acc_baseline_list.append(acc_base)

                # Optimized z
                _, wm_opt = z_optimizer.optimize(noised_i)
                acc_opt = extract_accuracy(wm_opt, secret_i)
                acc_optimized_list.append(acc_opt)

            count += B
            if count >= args.val_num_images:
                break

        results[key] = {
            'baseline': np.mean(acc_baseline_list),
            'optimized': np.mean(acc_optimized_list),
            'improvement': np.mean(acc_optimized_list) - np.mean(acc_baseline_list),
        }

    return results


def score_checkpoint_params(args):
    """保存复现实验所需的 ScoreNet、噪声和 z-opt 参数。"""
    keys = (
        'channel_dim', 'noise_mode', 'val_restore_noise_mode',
        'p_adv', 'eps_min', 'eps_max', 'eps_schedule',
        'train_noise_dict', 'test_noise_dict',
        'z_opt_steps', 'z_opt_lr', 'z_opt_sigma',
        'z_opt_lambda_prior', 'z_opt_lambda_ent',
    )
    params = {key: getattr(args, key) for key in keys}
    params.update({
        'in_channels': args.channel_dim,
        'base_channels': args.score_base_channels,
        'channel_mults': args.score_channel_mults,
        'embed_dim': args.score_embed_dim,
        'bit_length': args.bit_length,
        'img_size': args.im_size,
    })
    return params


# ==================== 主训练函数 ====================


def train(args):
    torch.manual_seed(args.seed)

    print(f"\n加载 Phase 2 checkpoint: {args.phase2_checkpoint}")
    phase2_dict = torch.load(args.phase2_checkpoint, map_location="cpu", weights_only=False)
    prepare_runtime_args(args, phase2_dict)

    # 打印参数
    table = PrettyTable(["Argument", "Value"])
    for k, v in vars(args).items():
        table.add_row([k, v])
    print(table)

    # ---- 日志和目录 ----
    log_path = os.path.join(args.logs_path, args.train_name)
    ckpt_dir = os.path.join(args.checkpoint_path, args.train_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_path)

    # ---- 加载 Phase 2 iIWN (冻结) ----
    model = Model(
        img_size=args.im_size, channel_dim=args.channel_dim,
        bit_length=args.bit_length, k=args.k,
        min_size=args.min_size, fc=args.fc
    ).to(args.device)

    model.load_state_dict(phase2_dict['model_state_dict'])
    freeze_module(model)
    print("iIWN 已加载并冻结。")

    # ---- 创建 Score Network ----
    score_net = ScoreNetwork(
        in_channels=args.channel_dim,
        base_channels=args.score_base_channels,
        channel_mults=tuple(args.score_channel_mults),
        embed_dim=args.score_embed_dim,
    ).to(args.device)

    score_params = sum(p.numel() for p in score_net.parameters())
    print(f"Score Network 参数量: {score_params / 1e6:.2f}M")

    # ---- 噪声层 (用于在线生成训练数据) ----
    train_noiselayer, val_restore_noiselayer, zopt_val_noiselayer = build_noise_layers(
        args, phase2_dict)

    # ---- 优化器 ----
    optim = AdamW(score_net.parameters(), lr=args.score_lr, betas=(0.9, 0.999),
                  weight_decay=args.score_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.num_epoch, eta_min=args.score_min_lr)

    # ---- 数据加载 ----
    train_dataset = HideImage(args.dataset_path, args.im_size,
                              args.bit_length, args.channel_dim)
    val_dataset = HideImage(args.val_dataset_path, args.im_size,
                            args.bit_length, args.channel_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=True, pin_memory=True)

    # ---- 断点续训 ----
    start_epoch = 0
    global_step = 0
    if args.continue_train:
        ckpt_path = find_latest_model(ckpt_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            score_net.load_state_dict(ckpt['score_net_state_dict'])
            optim.load_state_dict(ckpt['optim_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch']
            global_step = ckpt['global_step']
            print(f"从 epoch {start_epoch} 继续训练")

    # ---- 训练循环 ----
    score_net.train()

    remaining = max(args.num_epoch - start_epoch, 0)
    for epoch in tqdm(range(remaining), position=0, desc="Phase3 Epoch", ncols=100):
        now_epoch = epoch + start_epoch + 1

        loss_list = []
        adv_ratio_list = []
        tqdm_iter = tqdm(iter(train_loader), position=1, desc="Iteration", ncols=140)

        for cover, secret in tqdm_iter:
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # 在线生成 I_rec，并用 cover 监督恢复方向
            I_rec, cover_target, noise_stats = generate_reconstruction(
                model, train_noiselayer, cover, secret, args, now_epoch)

            # Supervised restoration training
            loss, stats = restoration_loss(
                score_net, I_rec, cover_target,
                sigma=args.z_opt_sigma,
            )

            optim.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)

            optim.step()

            # 记录
            global_step += 1
            loss_list.append(loss.item())
            adv_ratio_list.append(noise_stats['noise_adv_ratio'])

            desc = (f"E{now_epoch} restore_loss={np.mean(loss_list):.4f} "
                    f"adv={np.mean(adv_ratio_list):.2f} eps={noise_stats['noise_eps']:.4f} "
                    f"corr_norm={stats['correction_norm']:.2f}")
            tqdm_iter.set_description(desc)

            if global_step % 50 == 0:
                for k, v in stats.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("train/noise_adv_ratio",
                                  noise_stats['noise_adv_ratio'], global_step)
                writer.add_scalar("train/noise_eps",
                                  noise_stats['noise_eps'], global_step)

        scheduler.step()
        writer.add_scalar("train/epoch_loss", np.mean(loss_list), now_epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], now_epoch)

        # ---- 验证和保存 ----
        if (now_epoch - 1) % args.val_save_epoch == 0:
            print(f"\n验证 epoch {now_epoch}...")
            score_net.eval()

            # 1. Supervised restoration validation loss
            val_losses = []
            for val_cover, val_secret in val_loader:
                val_cover = val_cover.to(args.device)
                val_secret = val_secret.to(args.device)
                I_rec_val, cover_val, _ = generate_reconstruction(
                    model, val_restore_noiselayer, val_cover, val_secret, args, now_epoch)
                with torch.no_grad():
                    val_loss, _ = restoration_loss(
                        score_net, I_rec_val, cover_val,
                        sigma=args.z_opt_sigma,
                    )
                val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("val/restore_loss", avg_val_loss, now_epoch)
            print(f"  验证 restore loss: {avg_val_loss:.4f}")

            # 2. z 优化效果评估 (每 val_z_opt_epoch 做一次，比较耗时)
            if (now_epoch - 1) % args.val_z_opt_epoch == 0:
                print("  评估 z 优化效果...")
                z_opt_results = validate_z_optimization(
                    model, score_net, zopt_val_noiselayer, val_loader, args)

                for noise_type, res in z_opt_results.items():
                    print(f"    {noise_type}: "
                          f"baseline={res['baseline']:.4f} "
                          f"optimized={res['optimized']:.4f} "
                          f"Δ={res['improvement']:+.4f}")
                    writer.add_scalar(
                        f"val_z_opt/baseline_{noise_type}",
                        res['baseline'], now_epoch)
                    writer.add_scalar(
                        f"val_z_opt/optimized_{noise_type}",
                        res['optimized'], now_epoch)
                    writer.add_scalar(
                        f"val_z_opt/improvement_{noise_type}",
                        res['improvement'], now_epoch)

            score_net.train()

            # 保存 checkpoint
            save_dict = {
                'score_net_state_dict': score_net.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': now_epoch,
                'global_step': global_step,
                'param_dict': score_checkpoint_params(args),
                'phase2_checkpoint': args.phase2_checkpoint,
            }
            save_path = os.path.join(ckpt_dir, f"score_model_{now_epoch}.pth")
            torch.save(save_dict, save_path)
            print(f"  checkpoint 已保存: {save_path}\n")

    writer.close()
    print("Phase 3 训练完成。")


# ==================== 参数配置 ====================


def main():
    # 与 Phase 1/2 保持一致的固定训练噪声池。
    train_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 1.5, "GaussianNoise": 0.15,
        "SaltPepperNoise": 0.15, "MedianFilter": 7, "Dropout": 0.3,
    }
    test_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 3., "GaussianNoise": 0.3,
        "SaltPepperNoise": 0.3, "MedianFilter": 9, "Dropout": 0.5
    }

    parser = argparse.ArgumentParser(
        description="ALCLO Phase 3: Score Network 训练")

    # 设备
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # 数据
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_path", type=str, default=r"../../DIV2K/train")
    parser.add_argument("--val_dataset_path", type=str, default=r"../../DIV2K/val")
    parser.add_argument("--im_size", type=int, default=256)
    parser.add_argument("--channel_dim", type=int, default=3)

    # 噪声
    parser.add_argument("--train_noise_dict", type=dict, default=train_noise_dict)
    parser.add_argument("--test_noise_dict", type=dict, default=test_noise_dict)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--noise_mode", choices=["hybrid", "fixed"], default="hybrid",
                        help="Phase 3 训练噪声源: hybrid 复用 Phase 2 attacker + 固定噪声池")
    parser.add_argument("--val_restore_noise_mode", choices=["hybrid", "fixed"],
                        default="hybrid",
                        help="restore loss 验证使用的噪声源; z-opt 验证始终逐项使用固定攻击")
    parser.add_argument("--p_adv", type=float, default=0.35,
                        help="hybrid 噪声池中使用 Phase 2 attacker 的样本比例")
    parser.add_argument("--eps_min", type=float, default=2.0 / 255)
    parser.add_argument("--eps_max", type=float, default=None,
                        help="hybrid attacker 最大扰动; 默认读取 ckpt，否则用 Phase 2 R0 的 12/255")
    parser.add_argument("--eps_schedule", choices=["constant", "linear"],
                        default="constant",
                        help="constant 使用 Phase 2 最终 eps; linear 从 eps_min 线性升到 eps_max")
    parser.add_argument("--atk_hidden_channels", type=int, default=64)
    parser.add_argument("--atk_num_blocks", type=int, default=4)

    # iIWN 模型参数 (必须与 Phase 2 一致)
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--fc", type=bool, default=False)

    # Phase 2 checkpoint
    parser.add_argument("--phase2_checkpoint", type=str, required=True,
                        help="Phase 2 训练好的 checkpoint 路径")

    # Score Network 架构
    parser.add_argument("--score_base_channels", type=int, default=64)
    parser.add_argument("--score_channel_mults", type=int, nargs='+',
                        default=[1, 2, 4, 4],
                        help="Score U-Net 各层通道倍数")
    parser.add_argument("--score_embed_dim", type=int, default=256)

    # ScoreNetwork 训练参数
    parser.add_argument("--score_lr", type=float, default=1e-4)
    parser.add_argument("--score_min_lr", type=float, default=2e-6)
    parser.add_argument("--score_weight_decay", type=float, default=1e-5)

    # z 优化验证参数
    parser.add_argument("--z_opt_steps", type=int, default=30,
                        help="z 优化步数")
    parser.add_argument("--z_opt_lr", type=float, default=0.03,
                        help="z 优化学习率")
    parser.add_argument("--z_opt_sigma", type=float, default=0.01,
                        help="z 优化时 score network 的 σ")
    parser.add_argument("--z_opt_lambda_prior", type=float, default=0.5)
    parser.add_argument("--z_opt_lambda_ent", type=float, default=0.3)
    parser.add_argument("--val_num_images", type=int, default=32,
                        help="z 优化验证时使用的图像数量")

    # 训练参数
    parser.add_argument("--train_name", type=str, default="phase3_hybrid_multinoise")
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--val_save_epoch", type=int, default=1,
                        help="每隔多少 epoch 验证和保存")
    parser.add_argument("--val_z_opt_epoch", type=int, default=4,
                        help="每隔多少 epoch 评估 z 优化效果 (耗时较长)")

    # 路径
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--continue_train", type=bool, default=False)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")

    train(args)


if __name__ == "__main__":
    main()
