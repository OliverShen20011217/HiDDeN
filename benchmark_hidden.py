import argparse
import csv
import os
import io
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter

from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import time


# =========================
# Dataset
# =========================

class PathDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int):
        self.paths = paths
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), p


def load_paths_from_sample_list(sample_list: str, data_dir: Optional[str]) -> List[str]:
    paths = []
    with open(sample_list, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            paths.append(s if os.path.isabs(s) else os.path.join(data_dir, s))
    return paths


# =========================
# Model
# =========================

def build_config(image_size: int, message_length: int):
    return HiDDenConfiguration(
        H=image_size,
        W=image_size,
        message_length=message_length,
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


def load_hidden_from_checkpoint(ckpt_path, device, config):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Hidden(config, device, Noiser([], device), tb_logger=None)
    model.encoder_decoder.load_state_dict(ckpt["enc-dec-model"])
    model.discriminator.load_state_dict(ckpt["discrim-model"])
    model.encoder_decoder.eval()
    return model


# =========================
# Metrics
# =========================

def psnr_batch(x, y):
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def ber_k(decoded, target, k):
    d = decoded[:, :k].round()
    t = target[:, :k]
    return (d != t).float().sum(dim=1) / k


def make_messages(bs, L, k, device):
    msg = torch.zeros((bs, L), device=device)
    msg[:, :k] = torch.randint(0, 2, (bs, k), device=device).float()
    return msg


# =========================
# Attacks
# =========================

def rotate_cpu(batch, deg):
    out = []
    for img in batch.cpu():
        out.append(TF.to_tensor(TF.rotate(TF.to_pil_image(img), deg)))
    return torch.stack(out).to(batch.device)


def gaussian_noise(batch, sigma):
    return (batch + torch.randn_like(batch) * sigma).clamp(0, 1)


def gaussian_blur(batch, sigma):
    out = []
    for img in batch.cpu():
        pil = TF.to_pil_image(img)
        out.append(TF.to_tensor(pil.filter(ImageFilter.GaussianBlur(radius=sigma))))
    return torch.stack(out).to(batch.device)


def brightness(batch, factor):
    return (batch * factor).clamp(0, 1)


def contrast(batch, factor):
    mean = batch.mean(dim=(2, 3), keepdim=True)
    return ((batch - mean) * factor + mean).clamp(0, 1)


def jpeg(batch, quality):
    out = []
    for img in batch.cpu():
        pil = TF.to_pil_image(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out).to(batch.device)

@torch.no_grad()
def eval_payload(model, loader, L, payload_list, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "payload_k",
            "num_images",
            "mean_psnr",
            "mean_BER",
            "recover_rate",
            "embed_time_ms",
            "decode_time_ms",
        ])

        device = next(model.encoder_decoder.parameters()).device

        for k in payload_list:
            psnr_sum, ber_sum, ok_sum = 0.0, 0.0, 0
            embed_time, decode_time = 0.0, 0.0
            n = 0

            for images, _ in loader:
                images = images.to(device)
                bsz = images.size(0)
                messages = make_messages(bsz, L, k, device)

                # ---- embedding time ----
                torch.cuda.synchronize()
                t0 = time.time()
                encoded, _, _ = model.encoder_decoder(images, messages)
                torch.cuda.synchronize()
                t1 = time.time()
                embed_time += (t1 - t0)

                # ---- decoding time ----
                torch.cuda.synchronize()
                t2 = time.time()
                decoded = model.encoder_decoder.decoder(encoded)
                torch.cuda.synchronize()
                t3 = time.time()
                decode_time += (t3 - t2)

                psnr = psnr_batch(images, encoded)
                ber = ber_k(decoded, messages, k)

                psnr_sum += psnr.sum().item()
                ber_sum += ber.sum().item()
                ok_sum += (ber == 0).sum().item()
                n += bsz

            w.writerow([
                k,
                n,
                psnr_sum / n,
                ber_sum / n,
                ok_sum / n,
                1000 * embed_time / n,   # ms / image
                1000 * decode_time / n,  # ms / image
            ])

            print(
                f"[payload] k={k} | "
                f"PSNR={psnr_sum/n:.2f} | "
                f"BER={ber_sum/n:.4f} | "
                f"recover={ok_sum/n:.3f} | "
                f"embed={1000*embed_time/n:.2f}ms | "
                f"decode={1000*decode_time/n:.2f}ms"
            )




# =========================
# Generic eval helper
# =========================

@torch.no_grad()
def eval_attack(model, loader, L, k, values, attack_fn, csv_header, csv_out):
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)

        for v in values:
            ber_sum, ok, n = 0.0, 0, 0
            for imgs, _ in loader:
                imgs = imgs.to(next(model.encoder_decoder.parameters()).device)
                msgs = make_messages(imgs.size(0), L, k, imgs.device)
                enc, _, _ = model.encoder_decoder(imgs, msgs)
                attacked = attack_fn(enc, v)
                dec = model.encoder_decoder.decoder(attacked)
                ber = ber_k(dec, msgs, k)
                ber_sum += ber.sum().item()
                ok += (ber == 0).sum().item()
                n += imgs.size(0)

            w.writerow([v, n, ber_sum / n, ok / n])
            print(f"[attack] val={v} BER={ber_sum/n:.4f} recover={ok/n:.3f}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample-list", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["payload", "rotate", "gauss", "blur",
                             "brightness", "contrast", "jpeg"])
    ap.add_argument("--csv-out", required=True)

    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--message-length", type=int, default=30)
    ap.add_argument("--payload", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=100)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_paths_from_sample_list(args.sample_list, args.data_dir)

    ds = PathDataset(paths, args.image_size)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)),
                        shuffle=False, num_workers=0)

    model = load_hidden_from_checkpoint(
        args.checkpoint,
        device,
        build_config(args.image_size, args.message_length)
    )

    if args.mode == "payload":
        ks = [5, 10, 15, 20, 25, 30]
        eval_payload(
            model,
            loader,
            L=args.message_length,
            payload_list=ks,
            out_csv=args.csv_out
        )


    elif args.mode == "rotate":
        eval_attack(model, loader, args.message_length, args.payload,
                    list(range(1, 11)),
                    rotate_cpu,
                    ["angle_deg", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)

    elif args.mode == "gauss":
        eval_attack(model, loader, args.message_length, args.payload,
                    [i * 0.05 for i in range(0, 11)],
                    gaussian_noise,
                    ["sigma", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)

    elif args.mode == "blur":
        eval_attack(model, loader, args.message_length, args.payload,
                    [0, 1, 2, 3, 4, 5],
                    gaussian_blur,
                    ["sigma", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)

    elif args.mode == "brightness":
        eval_attack(model, loader, args.message_length, args.payload,
                    [0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4],
                    brightness,
                    ["factor", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)

    elif args.mode == "contrast":
        eval_attack(model, loader, args.message_length, args.payload,
                    [0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4],
                    contrast,
                    ["factor", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)

    elif args.mode == "jpeg":
        eval_attack(model, loader, args.message_length, args.payload,
                    [10,20,30,40,50,60,70,80,90],
                    jpeg,
                    ["quality", "num_images", "mean_BER", "recover_rate"],
                    args.csv_out)


if __name__ == "__main__":
    main()
