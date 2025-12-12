import argparse
import os
import time
import math
import io
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image

from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser


# -------------------------
# Config & model utilities
# -------------------------

def build_hidden_config():
    """
    Build HiDDeN config consistent with your training setup.
    Adjust here ONLY if you changed the architecture during training.
    """
    config = HiDDenConfiguration(
        H=128,
        W=128,
        message_length=30,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=0.001,
        enable_fp16=False,
    )
    return config


def load_hidden_model(checkpoint_path: str, device: torch.device):
    """
    Load HiDDeN model for evaluation.

    This version assumes checkpoints saved by train.py that contain:
      - 'enc-dec-model': state_dict of EncoderDecoder
      - 'discrim-model': state_dict of Discriminator
    """
    from options import HiDDenConfiguration  # 用项目里原来的配置类

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"Checkpoint top-level keys: {checkpoint.keys()}")

    # === 1. 手动构造与训练时一致的配置 ===
    config = HiDDenConfiguration(
        H=128,
        W=128,
        message_length=30,
        encoder_blocks=4,
        encoder_channels=64,
        decoder_blocks=7,
        decoder_channels=64,
        use_discriminator=True,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        decoder_loss=1.0,
        encoder_loss=0.7,
        adversarial_loss=0.001,
        enable_fp16=False,
    )

    # === 2. 创建 Noiser 和 Hidden 模型 ===
    noiser = Noiser([], device)  # 评估时先不加噪声
    model = Hidden(config, device, noiser, tb_logger=None)

    # === 3. 从 checkpoint 中加载参数 ===
    if "enc-dec-model" in checkpoint and "discrim-model" in checkpoint:
        model.encoder_decoder.load_state_dict(checkpoint["enc-dec-model"])
        model.discriminator.load_state_dict(checkpoint["discrim-model"])
        print("Loaded encoder-decoder and discriminator weights from 'enc-dec-model' / 'discrim-model'.")
    elif "enc_dec_state_dict" in checkpoint and "discrim_state_dict" in checkpoint:
        # 兼容另一种保存格式（如果以后用到）
        model.encoder_decoder.load_state_dict(checkpoint["enc_dec_state_dict"])
        model.discriminator.load_state_dict(checkpoint["discrim_state_dict"])
        print("Loaded encoder-decoder and discriminator weights from 'enc_dec_state_dict' / 'discrim_state_dict'.")
    else:
        raise RuntimeError(
            "Could not find known model keys in checkpoint. "
            "Expected 'enc-dec-model'/'discrim-model' or 'enc_dec_state_dict'/'discrim_state_dict'."
        )

    # 切到 eval 模式
    model.encoder_decoder.eval()
    model.discriminator.eval()

    return model, config

# -------------------------
# Dataset & dataloader
# -------------------------

def build_dataloader(data_dir, image_size, batch_size):
    """
    Build a DataLoader from a folder with ImageFolder structure:
    data_dir/
        some_class_name/
            img1.jpg
            img2.jpg
            ...
    If your images are directly in data_dir without subfolders,
    you can create a dummy subfolder and move all images inside.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataset, loader


# -------------------------
# Noise helpers
# -------------------------

def apply_gaussian_blur(batch_tensor, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to a batch of images (N, C, H, W).
    Returns a tensor on the same device.
    """
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    imgs_cpu = batch_tensor.detach().cpu()
    out = []
    for img in imgs_cpu:
        pil_img = TF.to_pil_image(img)
        pil_blur = blur(pil_img)
        out.append(TF.to_tensor(pil_blur))
    return torch.stack(out, dim=0).to(batch_tensor.device)


def apply_jpeg_compression(batch_tensor, quality=50):
    """
    Apply JPEG compression (with given quality) to a batch of images.
    """
    imgs_cpu = batch_tensor.detach().cpu()
    out = []
    for img in imgs_cpu:
        pil_img = TF.to_pil_image(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_jpeg = Image.open(buf).convert("RGB")
        out.append(TF.to_tensor(pil_jpeg))
    return torch.stack(out, dim=0).to(batch_tensor.device)


def apply_rotation(batch_tensor, degrees=5.0):
    """
    Apply a fixed small rotation to a batch of images.
    """
    imgs_cpu = batch_tensor.detach().cpu()
    out = []
    for img in imgs_cpu:
        pil_img = TF.to_pil_image(img)
        pil_rot = TF.rotate(pil_img, angle=degrees)
        out.append(TF.to_tensor(pil_rot))
    return torch.stack(out, dim=0).to(batch_tensor.device)


# -------------------------
# Metric helpers
# -------------------------

def batch_psnr(x, y):
    """
    Compute PSNR per image for two batches of images in [0, 1].
    x, y: shape (N, C, H, W)
    Returns: tensor of shape (N,)
    """
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr


# -------------------------
# Main evaluation loop
# -------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model, config = load_hidden_model(args.checkpoint, device)

    dataset, loader = build_dataloader(
        data_dir=args.data_dir,
        image_size=config.H,
        batch_size=args.batch_size
    )

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    # Global accumulators
    total_bits = 0
    total_bit_errors_clean = 0
    total_bit_errors_blur = 0
    total_bit_errors_jpeg = 0
    total_bit_errors_rot = 0

    total_psnr_clean = 0.0
    total_psnr_blur = 0.0
    total_psnr_jpeg = 0.0
    total_psnr_rot = 0.0

    num_images = 0

    # Open CSV writer
    with open(args.csv_out, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "image_path",
            "bit_error_clean",
            "psnr_clean",
            "bit_error_blur",
            "psnr_blur",
            "bit_error_jpeg",
            "psnr_jpeg",
            "bit_error_rot",
            "psnr_rot",
        ])

        start_time = time.time()
        global_index = 0  # track which sample in dataset

        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Generate random messages (0/1) as float
            messages = torch.randint(
                low=0, high=2,
                size=(batch_size, config.message_length),
                device=device,
                dtype=torch.float32
            )

            with torch.no_grad():
                # ----- Clean encode + decode -----
                encoded_images, noised_images, decoded_messages = model.encoder_decoder(images, messages)
                # decoded_messages: (N, L)
                decoded_bin = decoded_messages.detach().round().clamp(0, 1)

                # Bit error per sample
                bit_errors = (decoded_bin != messages).sum(dim=1)  # (N,)
                bit_error_rate = bit_errors.float() / config.message_length  # (N,)

                # PSNR between cover and encoded (how much visual distortion)
                psnr_clean_batch = batch_psnr(images, encoded_images)  # (N,)

                # ----- Add synthetic noises on encoded images -----
                # 1) Gaussian blur
                encoded_blur = apply_gaussian_blur(encoded_images)
                decoded_blur = model.encoder_decoder.decoder(encoded_blur)
                decoded_blur_bin = decoded_blur.detach().round().clamp(0, 1)
                bit_errors_blur = (decoded_blur_bin != messages).sum(dim=1)
                bit_error_rate_blur = bit_errors_blur.float() / config.message_length
                psnr_blur_batch = batch_psnr(images, encoded_blur)

                # 2) JPEG compression
                encoded_jpeg = apply_jpeg_compression(encoded_images, quality=50)
                decoded_jpeg = model.encoder_decoder.decoder(encoded_jpeg)
                decoded_jpeg_bin = decoded_jpeg.detach().round().clamp(0, 1)
                bit_errors_jpeg = (decoded_jpeg_bin != messages).sum(dim=1)
                bit_error_rate_jpeg = bit_errors_jpeg.float() / config.message_length
                psnr_jpeg_batch = batch_psnr(images, encoded_jpeg)

                # 3) Small rotation
                encoded_rot = apply_rotation(encoded_images, degrees=5.0)
                decoded_rot = model.encoder_decoder.decoder(encoded_rot)
                decoded_rot_bin = decoded_rot.detach().round().clamp(0, 1)
                bit_errors_rot = (decoded_rot_bin != messages).sum(dim=1)
                bit_error_rate_rot = bit_errors_rot.float() / config.message_length
                psnr_rot_batch = batch_psnr(images, encoded_rot)

            # Accumulate global stats
            total_bits += batch_size * config.message_length
            total_bit_errors_clean += int(bit_errors.sum().item())
            total_bit_errors_blur += int(bit_errors_blur.sum().item())
            total_bit_errors_jpeg += int(bit_errors_jpeg.sum().item())
            total_bit_errors_rot += int(bit_errors_rot.sum().item())

            total_psnr_clean += float(psnr_clean_batch.sum().item())
            total_psnr_blur += float(psnr_blur_batch.sum().item())
            total_psnr_jpeg += float(psnr_jpeg_batch.sum().item())
            total_psnr_rot += float(psnr_rot_batch.sum().item())

            num_images += batch_size

            # Write per-image rows to CSV
            for i in range(batch_size):
                # dataset.samples[global_index] = (path, class_index)
                img_path = dataset.samples[global_index][0]
                writer.writerow([
                    img_path,
                    float(bit_error_rate[i].item()),
                    float(psnr_clean_batch[i].item()),
                    float(bit_error_rate_blur[i].item()),
                    float(psnr_blur_batch[i].item()),
                    float(bit_error_rate_jpeg[i].item()),
                    float(psnr_jpeg_batch[i].item()),
                    float(bit_error_rate_rot[i].item()),
                    float(psnr_rot_batch[i].item()),
                ])
                global_index += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"[Batch {batch_idx + 1}/{len(loader)}] processed")

        end_time = time.time()

    # Final aggregated metrics
    ber_clean = total_bit_errors_clean / total_bits
    ber_blur = total_bit_errors_blur / total_bits
    ber_jpeg = total_bit_errors_jpeg / total_bits
    ber_rot = total_bit_errors_rot / total_bits

    avg_psnr_clean = total_psnr_clean / num_images
    avg_psnr_blur = total_psnr_blur / num_images
    avg_psnr_jpeg = total_psnr_jpeg / num_images
    avg_psnr_rot = total_psnr_rot / num_images

    print("====================================")
    print("Evaluation finished.")
    print(f"Total images: {num_images}")
    print(f"Total bits: {total_bits}")
    print(f"Time cost: {end_time - start_time:.2f} s")
    print("----- Bit error rate (lower is better) -----")
    print(f"Clean:   {ber_clean:.6f}")
    print(f"Blur:    {ber_blur:.6f}")
    print(f"JPEG:    {ber_jpeg:.6f}")
    print(f"Rotate:  {ber_rot:.6f}")
    print("----- Average PSNR (dB, higher is better) -----")
    print(f"Clean:   {avg_psnr_clean:.2f}")
    print(f"Blur:    {avg_psnr_blur:.2f}")
    print(f"JPEG:    {avg_psnr_jpeg:.2f}")
    print(f"Rotate:  {avg_psnr_rot:.2f}")
    print("Per-image results saved to:", args.csv_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiDDeN on a test image folder.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to test data root folder (ImageFolder-style: contains subfolders with images).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the HiDDeN checkpoint (.pyt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="results_hidden_e20.csv",
        help="CSV file path to save per-image metrics.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
