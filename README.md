# ALCLO

ALCLO is a research project for robust reversible image watermarking. It combines an integer invertible watermark network, adversarial training, latent distribution regularization, and score-guided latent optimization during watermark extraction.

The training pipeline has three stages:

1. **Phase 1:** Train the base integer invertible watermark network (iIWN).
2. **Phase 2:** Jointly apply adversarial training and latent distribution regularization.
3. **Phase 3:** Freeze the Phase 2 watermark network and train a Score Network. At evaluation time, use the Score Network to optimize the latent variable `z`.

The default configurations support color images at 256 x 256 with 64 watermark bits and grayscale images at 512 x 512 with 256 watermark bits. Phase 3 requires the exact Phase 2 weights that were frozen during Score Network training, together with the corresponding Score Network checkpoint.

## Main directories

```text
ALCLO/
|-- alclo/
|   |-- adversary/       # Adversarial attacker and hybrid noise layers
|   |-- data/            # Image dataset loader
|   |-- losses/          # Adversarial and latent distribution losses
|   `-- score/           # Score Network and latent optimizer
|-- scripts/
|   |-- train_phase1.py
|   |-- train_phase2.py
|   |-- train_phase2_gray.py
|   |-- train_phase3.py
|   |-- train_phase3_gray.py
|   |-- evaluate.py
|   `-- evaluate_gray.py
|-- requirements.txt
`-- LICENSE
```

## Installation and data

Create a Python environment and install a compatible PyTorch and torchvision pair for your CUDA environment. Then run:

```bash
git clone https://github.com/chen-w-w/ALCLO.git
cd ALCLO
pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies; it is not a fully pinned reproduction environment. The LPIPS training loss may download perceptual network weights on first use. The grayscale evaluation entry point only requires `torch`, `torchvision`, `numpy`, `Pillow`, and `tqdm`. The remaining dependencies support training, color evaluation, and reversible compression utilities.

Image directories are searched recursively for PNG, JPEG, TIFF, BMP, GIF, and WebP files. Set `--dataset_path` and `--val_dataset_path` to your training and validation directories. Images are cropped and resized to the configured dimensions, and random watermark bits are generated online.

Run all examples from the repository root. Replace `data/train`, `data/val`, and `data/test` with your actual paths.

This repository contains source code only. Datasets, pretrained weights, and training artifacts are not included. Run Phase 1 to obtain base network weights, then proceed to Phases 2 and 3. If you already have compatible base weights, pass their location through `--phase1_checkpoint`. Only load checkpoints from trusted sources.

## Color training and evaluation

Train the base network from scratch. Skip this step if you already have compatible weights:

```bash
python scripts/train_phase1.py --dataset_path data/train --val_dataset_path data/val --train_name phase1_color
```

Start Phase 2 from the base network checkpoint. Replace the example `--baseline_psnr` value with a measurement on your validation set:

```bash
python scripts/train_phase2.py --phase1_checkpoint checkpoints/phase1_color/model_best.pth --dataset_path data/train --val_dataset_path data/val --train_name phase2_color --baseline_psnr 40.0
```

Train the Score Network using your Phase 2 checkpoint:

```bash
python scripts/train_phase3.py --phase2_checkpoint checkpoints/phase2_color/model_best.pth --dataset_path data/train --val_dataset_path data/val --train_name phase3_color
```

Evaluate with an explicitly selected checkpoint pair. `score_model_4.pth` below is an example filename; replace it with the checkpoint selected on your validation set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/phase2_color/model_best.pth --score_checkpoint checkpoints/phase3_color/score_model_4.pth --baseline checkpoints/phase1_color/model_best.pth --dataset_path data/test --batch_size 1
```

Add `--skip_phase3` to evaluate Phase 2 only. Pass `--baseline ""` to omit the base network comparison. The color evaluator prints result tables and drops the final incomplete batch; use `--batch_size 1` to include every selected image.

## Grayscale training and evaluation

Train a grayscale base network, or skip this step if you have compatible weights:

```bash
python scripts/train_phase1.py --dataset_path data/train --val_dataset_path data/val --im_size 512 --channel_dim 1 --bit_length 256 --min_size 32 --batch_size 2 --train_name phase1_gray
```

Train Phase 2:

```bash
python scripts/train_phase2_gray.py --phase1_checkpoint checkpoints/phase1_gray/model_best.pth --dataset_path data/train --val_dataset_path data/val --train_name phase2_gray
```

Train Phase 3:

```bash
python scripts/train_phase3_gray.py --phase2_checkpoint checkpoints/phase2_gray/model_best.pth --dataset_path data/train --val_dataset_path data/val --checkpoint_path checkpoints --logs_path logs --train_name phase3_gray_hybrid
```

Evaluate the paired Phase 2 and Phase 3 checkpoints:

```bash
python scripts/evaluate_gray.py --checkpoint checkpoints/phase2_gray/model_best.pth --score_checkpoint checkpoints/phase3_gray_hybrid/score_model_4.pth --dataset_path data/test --no_baseline --num_images 0 --output outputs/evaluate_gray.json
```

Replace `score_model_4.pth` with the actual validation-selected checkpoint trained with the specified Phase 2 network. Always pass your own checkpoint paths instead of relying on local experiment defaults in the scripts. To include a base network comparison, remove `--no_baseline` and add `--baseline checkpoints/phase1_gray/model_best.pth`.

Evaluate Phase 2 only:

```bash
python scripts/evaluate_gray.py --checkpoint checkpoints/phase2_gray/model_best.pth --dataset_path data/test --no_baseline --skip_phase3 --num_images 0 --output outputs/evaluate_gray_phase2.json
```

The grayscale evaluator writes JSON reports containing PSNR, SSIM, bit accuracy, BER, and per-attack results. Phase 2 and Phase 3 share the same watermarked and attacked images; their extraction methods differ. Phase 3 requires gradients with respect to `z` and takes longer to evaluate. Reduce `--batch_size` if GPU memory is limited.

Inspect all available options:

```bash
python scripts/train_phase3_gray.py --help
python scripts/evaluate_gray.py --help
```

## Publication scope

`.gitignore` uses an allowlist for the model source, required base network and reversible compression implementation, the seven training and evaluation entry points listed above, and the README, dependency list, license, and Git configuration files.

Baselines, tests, demonstrations, plotting utilities, `research/`, local documents, images, datasets, pretrained weights, checkpoints, TensorBoard logs, evaluation outputs, temporary files, and IDE settings stay local. The required dataset loader is included in `alclo/data/`, so training does not depend on the excluded `research/` directory.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
