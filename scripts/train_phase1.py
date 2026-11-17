"""
ALCLO Phase 1 训练脚本: 基线 iIWN 训练 (复现 DRRW)

基础 iIWN 训练，适配 ALCLO 项目结构:
1. 支持从仓库根目录启动 scripts/ 下的入口
2. checkpoint 格式与 Phase 2 兼容
3. 通过命令行参数配置数据、模型与训练路径

训练完成后的 checkpoint 可直接被 train_phase2.py 加载。

用法:
    python scripts/train_phase1.py \
        --dataset_path ../../COCO/train \
        --val_dataset_path ../../COCO/val
"""

import os
import sys
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crmark.nets import Model
from crmark.compressor.utils import (
    extract_accuracy, overflow_num, compute_psnr,
    quantize_image, quantize_residual_image, find_latest_model
)
from alclo.data import HideImage

from prettytable import PrettyTable
from watermarklab.noiselayers.noiselayerloader import DigitalDistortion

warnings.filterwarnings("ignore")


# ==================== 训练辅助函数 ====================


def train_batch(model, args, noise_layer, cover, secret, now_step):
    """
    单批次训练 (与 DRRW 原始逻辑一致)。

    流程:
    1. 正向: cover + secret → stego + z
    2. 加噪: stego → noised_stego
    3. 逆向: noised_stego + rand_z → rec_cover + rec_secret
    4. 计算五项 loss 的加权和
    """
    # 正向: 嵌入水印
    stego, z = model.forward(cover, secret, args.hard_round, False)

    # 添加噪声
    noised_stego = noise_layer(stego, cover, now_step)

    # 逆向: 恢复 (使用随机 z 模拟有损信道)
    rand_z = torch.randn_like(z)
    rec_cover, rec_secret = model.forward(noised_stego, rand_z, args.hard_round, True)

    # 计算各项损失
    loss_penalty = model.PMSE(stego)
    loss_stego = model.MSE(stego, cover)
    loss_lpips = model.LLoss(stego, cover)
    loss_z = model.MSE(z, torch.zeros_like(z))
    loss_secret = model.MSE(rec_secret, secret)

    # 总损失
    loss = (args.lambda_penalty * loss_penalty
            + args.lambda_stego * loss_stego
            + args.lambda_lpips * loss_lpips
            + args.lambda_secret * loss_secret
            + args.lambda_z * loss_z)

    result = {
        "train_values": {
            "train_total_loss": loss,
            "train_accuracy": extract_accuracy(rec_secret, secret),
            "train_lambda_secret": args.lambda_secret,
            "train_loss_lpips": loss_lpips.item(),
            "train_loss_stego": loss_stego.item(),
            "train_loss_secret": loss_secret.item(),
            "train_loss_penalty": loss_penalty.item(),
            "train_loss_z": loss_z.item(),
            "train_overflow_0": overflow_num(stego, 0),
            "train_overflow_255": overflow_num(stego, 255),
            "train_stego_psnr": compute_psnr(stego, cover),
        }
    }
    return result


def val_batch(model, args, noise_layer, intensity, val_cover, val_secret):
    """
    单批次验证。

    对每种噪声类型分别测试水印提取精度。
    """
    secret_shape = (args.batch_size, 1, int(args.bit_length ** 0.5), int(args.bit_length ** 0.5))

    val_stego, val_z = model.forward(val_cover, val_secret, True, False)
    val_noised = noise_layer.test(quantize_image(val_stego), val_cover, intensity)
    val_noised_quant = quantize_image(val_noised)
    val_rec_cover, val_rec_secret = model.forward(val_noised_quant, torch.randn_like(val_z), True, True)

    result = {
        "val_values": {
            "val_accuracy": extract_accuracy(val_rec_secret, val_secret),
            "val_overflow_0": overflow_num(val_stego, 0),
            "val_overflow_255": overflow_num(val_stego, 255),
            "val_stego_psnr": compute_psnr(val_stego, val_cover),
        },
        "val_images": {
            "val_cover": val_cover,
            "val_noised_stego": quantize_image(val_noised),
            "val_stego": quantize_image(val_stego),
            "val_residual": quantize_residual_image(val_stego, val_cover),
            "val_recon_cover": quantize_image(val_rec_cover),
        }
    }

    # 如果水印比特数是完全平方数，添加可视化
    if int(args.bit_length ** 0.5) ** 2 == args.bit_length:
        result["val_images"]["val_z"] = quantize_image(val_z.view(secret_shape))
        result["val_images"]["val_secret"] = quantize_image(val_secret.view(secret_shape))
        result["val_images"]["val_recon_secret"] = quantize_image(val_rec_secret.view(secret_shape))

    return result


# ==================== 日志保存 ====================


def logs_train_save(writer, result, now_epoch):
    """保存训练/验证日志到 TensorBoard。"""
    for key in result.keys():
        if "values" in key:
            for k, v in result[key].items():
                writer.add_scalar(f"{key}/{k}", v, now_epoch)
        if "accuracy" in key and isinstance(result[key], dict):
            for k, v in result[key].items():
                writer.add_scalar(f"{key}/{k}", v, now_epoch)
        if "images" in key:
            for k, v in result[key].items():
                writer.add_images(f"{key}/{k}", v, now_epoch)


# ==================== 主训练函数 ====================


def train(args):
    torch.manual_seed(args.seed)

    # 打印参数
    table = PrettyTable(["Argument", "Value"])
    for k, v in vars(args).items():
        table.add_row([k, v])
    print(table)

    # 日志和目录
    log_path = os.path.join(args.logs_path, args.train_name)
    ckpt_dir = os.path.join(args.checkpoint_path, args.train_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_path)

    # 将训练参数信息写入 TensorBoard (markdown 表格,便于阅读)
    args_md_lines = ["| Argument | Value |", "| --- | --- |"]
    for k, v in vars(args).items():
        args_md_lines.append(f"| {k} | {v} |")
    writer.add_text("args", "\n".join(args_md_lines), 0)

    # 创建模型
    model = Model(
        img_size=args.im_size, channel_dim=args.channel_dim,
        bit_length=args.bit_length, k=args.k,
        min_size=args.min_size, fc=args.fc,
    ).to(args.device)

    # 噪声层
    train_noiselayer = DigitalDistortion(noise_dict=args.train_noise_dict, max_step=args.max_step)
    test_noiselayer = DigitalDistortion(noise_dict=args.test_noise_dict, max_step=args.max_step)

    # 数据加载
    train_dataset = HideImage(args.dataset_path, args.im_size, args.bit_length, args.channel_dim)
    val_dataset = HideImage(args.val_dataset_path, args.im_size, args.bit_length, args.channel_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, pin_memory=True)

    # 优化器和调度器
    optim_blocks = AdamW(model.inn_blocks.parameters(), lr=args.lr, betas=(0.5, 0.999),
                         eps=1e-6, weight_decay=1e-5)
    scheduler_blocks = torch.optim.lr_scheduler.StepLR(optim_blocks, 200, gamma=0.5)

    # 断点续训
    start_epoch = 0
    global_step = 0
    if args.continue_train:
        model_path = find_latest_model(ckpt_dir)
        if model_path:
            load_params = model.load_model(model_path, optim_blocks, scheduler_blocks)
            if load_params:
                start_epoch, global_step, lambda_secret = load_params
                args.lambda_secret = lambda_secret
                print(f"从 epoch {start_epoch} 继续训练, global_step={global_step}")

    model.train()

    inter_result = []
    average_acc_queue = []
    best_val_acc = -1.0  # 追踪最佳验证精度,用于更新 model_best.pth

    remaining = max(args.num_epoch - start_epoch, 0)
    for epoch in tqdm(range(remaining), position=0, desc="Phase1 Epoch", ncols=100):
        now_epoch = epoch + start_epoch + 1

        # lambda_secret 自适应衰减
        if args.lambda_secret < 5.:
            args.v = 0.95
            args.delta = 0.005

        if len(average_acc_queue) >= args.queue_len:
            if np.mean(average_acc_queue) > 1. - args.delta:
                args.lambda_secret *= args.v
                average_acc_queue.clear()
                print(f"lambda_secret 衰减为: {args.lambda_secret:.4f}")

        loss_list = []
        acc_list = []

        tqdm_iter = tqdm(iter(train_loader), position=1, desc="Iteration", ncols=140)
        for cover, secret in tqdm_iter:
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            result = train_batch(model, args, train_noiselayer, cover, secret, now_epoch)
            total_loss = result["train_values"]["train_total_loss"]
            acc_list.append(result["train_values"]["train_accuracy"])

            optim_blocks.zero_grad()
            total_loss.backward()
            optim_blocks.step()

            inter_result.append([result, global_step + 1])
            global_step += 1
            loss_list.append(total_loss.item())

            tqdm_iter.set_description(
                f"E{now_epoch}/{args.num_epoch} acc={np.mean(acc_list):.5f} "
                f"lam_s={args.lambda_secret:.1f} loss={np.mean(loss_list):.5f}")

        average_acc_queue.append(np.mean(acc_list))
        if len(average_acc_queue) > args.queue_len:
            average_acc_queue.pop(0)

        scheduler_blocks.step()

        # 验证和保存
        if (now_epoch - 1) % args.val_save_epoch == 0:
            print(f"\n验证 epoch {now_epoch}...")

            ave_acc_list = []
            stego_psnr_list = []

            model.eval()
            with torch.no_grad():
                acc_dict = {}
                last_batch_images = {}

                for key in test_noiselayer.noise_dict.keys():
                    accuracy_list = []
                    val_tqdm = tqdm(iter(val_loader), position=1,
                                   desc=f"Val ({key})", ncols=100)

                    for val_cover, val_secret in val_tqdm:
                        val_cover = val_cover.to(args.device)
                        val_secret = val_secret.to(args.device)

                        val_result = val_batch(
                            model, args, test_noiselayer.noise_layers[key],
                            test_noiselayer.noise_dict[key], val_cover, val_secret
                        )

                        accuracy_list.append(val_result["val_values"]["val_accuracy"])
                        ave_acc_list.append(val_result["val_values"]["val_accuracy"])
                        stego_psnr_list.append(val_result["val_values"]["val_stego_psnr"])
                        last_batch_images = val_result["val_images"]

                    acc_dict[key] = np.mean(accuracy_list)

            model.train()

            # 汇总验证结果
            val_summary = {
                "val_values": {
                    "train_loss": np.mean(loss_list),
                    "val_accuracy": np.mean(ave_acc_list),
                    "val_stego_psnr": np.mean(stego_psnr_list),
                },
                "val_images": last_batch_images,
                "val_accuracy": acc_dict,
            }

            # 写入 TensorBoard
            for one_res in inter_result:
                r, step = one_res
                logs_train_save(writer, r, step)
            inter_result.clear()
            logs_train_save(writer, val_summary, now_epoch)

            # 保存 checkpoint (格式与 Phase 2 兼容)
            model.save_model(args, optim_blocks, scheduler_blocks, global_step, now_epoch)

            # 维护 model_best.pth: 若当前验证平均精度超过历史最佳则覆盖
            current_val_acc = float(np.mean(ave_acc_list))
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_path = os.path.join(ckpt_dir, "model_best.pth")
                best_ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim_blocks.state_dict(),
                    "scheduler_state_dict": scheduler_blocks.state_dict(),
                    "epoch": now_epoch,
                    "global_step": global_step,
                    "lambda_secret": args.lambda_secret,
                    "val_accuracy": current_val_acc,
                    "val_acc_dict": acc_dict,
                    "param_dict": {
                        "bit_length": args.bit_length,
                        "img_size": args.im_size,
                        "k": args.k,
                        "channel_dim": args.channel_dim,
                        "min_size": args.min_size,
                        "fc": args.fc,
                    },
                }
                torch.save(best_ckpt, best_path)
                writer.add_scalar("val_values/best_val_accuracy", best_val_acc, now_epoch)
                print(f"  ★ 新的最佳模型已保存 (val_acc={best_val_acc:.4f}) → {best_path}")

            print(f"  验证平均精度: {current_val_acc:.4f} (best: {best_val_acc:.4f})")
            for k, v in acc_dict.items():
                print(f"  {k}: {v:.4f}")
            print(f"  模型已保存 (epoch {now_epoch})\n")

    writer.close()
    print("Phase 1 训练完成。")


# ==================== 参数配置 ====================


def main():
    # 训练噪声 (较弱，与 DRRW 一致)
    train_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 1.5, "GaussianNoise": 0.15,
        "SaltPepperNoise": 0.15, "MedianFilter": 7, "Dropout": 0.3,
    }
    # 测试噪声 (较强)
    test_noise_dict = {
        "Jpeg": 30, "GaussianBlur": 3., "GaussianNoise": 0.3,
        "SaltPepperNoise": 0.3, "MedianFilter": 9, "Dropout": 0.5,
    }

    parser = argparse.ArgumentParser(description="ALCLO Phase 1: 基线 iIWN 训练 (DRRW 复现)")

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

    # 模型
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_size", type=int, default=16)
    parser.add_argument("--fc", type=bool, default=False)
    parser.add_argument("--hard_round", type=bool, default=False)

    # 训练
    parser.add_argument("--train_name", type=str, default="phase1_baseline_lambda_z1e-3")
    parser.add_argument("--num_epoch", type=int, default=650)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--val_save_epoch", type=int, default=1)
    parser.add_argument("--queue_len", type=int, default=3)

    # Loss 权重
    parser.add_argument("--lambda_stego", type=float, default=1.)
    parser.add_argument("--lambda_lpips", type=float, default=5.)
    parser.add_argument("--lambda_secret", type=float, default=1e4)
    parser.add_argument("--lambda_z", type=float, default=1e-3)
    parser.add_argument("--lambda_penalty", type=float, default=1e6)

    # lambda_secret 自适应衰减
    parser.add_argument("--v", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.01)

    # 路径
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--continue_train", type=bool, default=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu_id}")
    else:
        args.device = torch.device("cpu")

    train(args)


if __name__ == "__main__":
    main()
