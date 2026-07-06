"""
PopNet Section 3.1 — Source-free Depth Inference
=================================================
Generates source-free depth maps Dsf for a folder of RGB images using:
  1. DPT-Large [50] with FROZEN weights for base depth prediction
  2. Boosting Monocular Depth [45] for high-resolution local detail enhancement

Usage:
    python popnet_depth_infer.py \
        --input_dir /path/to/rgb_images \
        --output_dir /path/to/output_depths \
        --dpt_weights /path/to/dpt_large-midas-2f21e586.pt \
        --boost          # add this flag to apply boosting (slower but higher quality)
        --save_vis       # also save colored visualization maps

References:
    [50] Ranftl et al., "Vision Transformers for Dense Prediction," ICCV 2021
    [45] Miangoleh et al., "Boosting Monocular Depth Estimation Models to
         High-Resolution via Content-Adaptive Multi-Resolution Merging," CVPR 2021
    [51] Ranftl et al., "Towards Robust Monocular Depth Estimation," TPAMI 2022
"""

import sys
import types
import argparse
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

# ── resolve repo paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DPT_ROOT   = SCRIPT_DIR / "DPT"
BOOST_ROOT = SCRIPT_DIR / "BoostingMonocularDepth"

for p in [DPT_ROOT, BOOST_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# =============================================================================
# DPT loader
# =============================================================================
def load_dpt(weights_path: str, device: torch.device):
    """
    Load DPT-Large with FROZEN weights, as described in PopNet Sec. 3.1.
    The model choice is justified by its generalization capability [51].
    """
    from dpt.models import DPTDepthModel
    from dpt.transforms import Resize, NormalizeImage, PrepareForNet

    model = DPTDepthModel(
        path=weights_path,
        backbone="vitl16_384",
        non_negative=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)

    transform = Compose([
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])
    return model, transform


def dpt_infer(model, transform, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run DPT on a single BGR image. Returns depth map normalized to [0,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    sample  = transform({"image": img_rgb})["image"]
    tensor  = torch.from_numpy(sample).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(tensor)

    depth = depth.squeeze().cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    return depth.astype(np.float32)


# =============================================================================
# Boosting wrapper
# =============================================================================
def load_boost_model(device: torch.device):
    """
    Load the Boosting pix2pix merge network [45] without using TestOptions,
    to avoid argparse conflicts when called from inside another script.
    """
    from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

    opt = types.SimpleNamespace()
    # core
    opt.isTrain         = False
    opt.checkpoints_dir = str(BOOST_ROOT / "pix2pix" / "checkpoints")
    opt.name            = "mergemodel"
    opt.gpu_ids         = [0] if device.type == "cuda" else []
    opt.verbose         = False
    opt.preprocess      = "none"
    # network architecture
    opt.model           = "pix2pix4depth"
    opt.input_nc        = 2
    opt.output_nc       = 1
    opt.ngf             = 64
    opt.netG            = "unet_1024"
    opt.netD            = "basic"
    opt.norm            = "none"
    opt.no_dropout      = True
    opt.init_type       = "normal"
    opt.init_gain       = 0.02
    # loss / training (unused at inference but accessed during init)
    opt.gan_mode        = "vanilla"
    opt.lambda_L1       = 100.0
    opt.beta1           = 0.5
    opt.lr              = 0.0002
    opt.lr_policy       = "linear"
    opt.lr_decay_iters  = 50
    opt.n_epochs        = 100
    opt.n_epochs_decay  = 100
    opt.epoch           = "latest"
    opt.continue_train  = False

    merge_model = Pix2Pix4DepthModel(opt)
    merge_model.save_dir = str(BOOST_ROOT / "pix2pix" / "checkpoints" / "mergemodel")
    merge_model.load_networks("latest")
    merge_model.eval()
    return merge_model


def boost_infer(dpt_model, dpt_transform, merge_model,
                img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run Boosting pipeline from [45]:
      1. Low-res base depth via DPT
      2. Grid patches of high-res depth via DPT
      3. Each patch merged with base via pix2pix merge network
      4. Patches stitched back with Gaussian blending
    Returns depth map normalized to [0, 1], same resolution as input.
    """
    from utils import ImageandPatchs, applyGridpatch, impatch

    def generatemask(size):
        # Edge-to-edge Gaussian weight — no blank border, so seams blend fully.
        # Replaces the utils.generatemask which reserves a 15% zero-weight border.
        h, w = size
        cy, cx = h / 2.0, w / 2.0
        sigma_y, sigma_x = h / 2.5, w / 2.5
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        mask = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        return mask.astype(np.float32)

    h, w = img_bgr.shape[:2]

    # --- 1. base (low-res) depth ---
    base_depth = dpt_infer(dpt_model, dpt_transform, img_bgr, device)
    base_depth = cv2.resize(base_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    MERGE_SIZE = 1024
    # patch_size capped to image dimensions; stride = half for 50% overlap
    patch_size = min(h, w, 512)
    stride     = patch_size // 2

    # --- 2. build patch grid guaranteeing full-image coverage ---
    # We build rects manually instead of applyGridpatch so edge pixels are
    # always covered — the last patch in each row/col is anchored to the edge.
    patches = []
    y0 = 0
    while True:
        x0 = 0
        while True:
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)
            xa = x1 - patch_size  # anchor: shift left so patch fits inside
            ya = y1 - patch_size
            patches.append((max(0, xa), max(0, ya), x1 - max(0, xa), y1 - max(0, ya)))
            if x1 >= w:
                break
            x0 += stride
        if y1 >= h:
            break
        y0 += stride
    patches = list(dict.fromkeys(patches))  # deduplicate

    # seed accumulators with base depth so pixels missed by patches are valid
    depth_accum  = base_depth.copy() * 1e-6
    weight_accum = np.full((h, w), 1e-6, dtype=np.float32)

    to_tensor = transforms.ToTensor()

    # --- 3. process each patch ---
    for (px, py, pw, ph) in patches:
        patch_bgr  = img_bgr   [py:py+ph, px:px+pw]
        patch_base = base_depth[py:py+ph, px:px+pw]

        patch_depth = dpt_infer(dpt_model, dpt_transform, patch_bgr, device)
        patch_depth = cv2.resize(patch_depth, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # upscale to 1024 for merge network
        base_1024  = cv2.resize(patch_base,  (MERGE_SIZE, MERGE_SIZE), interpolation=cv2.INTER_LINEAR)
        depth_1024 = cv2.resize(patch_depth, (MERGE_SIZE, MERGE_SIZE), interpolation=cv2.INTER_LINEAR)

        t_base  = to_tensor(base_1024 ).unsqueeze(0).float().to(device)
        t_patch = to_tensor(depth_1024).unsqueeze(0).float().to(device)

        with torch.no_grad():
            merged = merge_model.netG(torch.cat([t_base, t_patch], dim=1))

        merged_np = merged.squeeze().cpu().numpy()
        merged_np = cv2.resize(merged_np, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # Gaussian mask: high weight in centre, tapers to zero at edges
        # — this is what eliminates visible seams at patch boundaries
        gauss = generatemask((ph, pw))

        depth_accum [py:py+ph, px:px+pw] += merged_np * gauss
        weight_accum[py:py+ph, px:px+pw] += gauss

    # --- 4. weighted blend — every pixel is covered so no hard edges ---
    depth_out = depth_accum / weight_accum

    # normalize to [0, 1]
    d_min, d_max = depth_out.min(), depth_out.max()
    if d_max - d_min > 1e-6:
        depth_out = (depth_out - d_min) / (d_max - d_min)
    return depth_out.astype(np.float32)



# =============================================================================
# I/O utilities
# =============================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def collect_images(input_dir: Path):
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def save_depth(depth: np.ndarray, out_path: Path, save_vis: bool = False):
    """Save depth as uint16 PNG (lossless) + optional colorized visualization."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_uint16 = (depth * 65535).astype(np.uint16)

    ok = cv2.imwrite(str(out_path), depth_uint16)
    if not ok:
        # cv2.imwrite fails silently on Windows with unicode/long paths.
        # Fall back to writing via numpy so we get an actual error message.
        try:
            import imageio
            imageio.imwrite(str(out_path), depth_uint16)
        except Exception as e:
            print(f"  [ERROR] Failed to save {out_path}: {e}")
            return

    if save_vis:
        vis_path = out_path.parent / "vis" / (out_path.stem + "_vis.png")
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        colored = cv2.applyColorMap(
            (depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        ok_vis = cv2.imwrite(str(vis_path), colored)
        if not ok_vis:
            print(f"  [ERROR] Failed to save vis {vis_path}")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="PopNet Sec 3.1 source-free depth generator")
    p.add_argument("--input_dir",   default="./Datasets/ACOD-12K/Test/Imgs",
                   help="Directory of RGB images")
    p.add_argument("--output_dir",  default="./Datasets/ACOD-12K/Test/PopNet_Depth",
                   help="Directory to write depth maps")
    p.add_argument("--dpt_weights", default="./Backbones/dpt_large-midas-2f21e586.pt",
                   help="Path to dpt_large-midas-2f21e586.pt")
    p.add_argument("--boost", default=True, action="store_true",
                   help="Apply Boosting [45] for high-res depth (slower, higher quality)")
    p.add_argument("--save_vis",    action="store_true",
                   help="Save colorized depth visualizations alongside depth maps")
    p.add_argument("--device",      default="cuda",
                   help="'cuda' or 'cpu' (default: cuda)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load DPT (frozen)
    print("Loading DPT-Large [50] with frozen weights ...")
    dpt_model, dpt_transform = load_dpt(args.dpt_weights, device)
    print(f"  Trainable DPT params: "
          f"{sum(p.numel() for p in dpt_model.parameters() if p.requires_grad):,} "
          f"(should be 0 — frozen)")

    # optionally load boost merge network
    merge_model = None
    if args.boost:
        print("Loading Boosting merge network [45] ...")
        merge_model = load_boost_model(device)

    # inference loop
    images = collect_images(input_dir)
    print(f"Found {len(images)} images in {input_dir}")

    for i, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [skip] Cannot read: {img_path.name}")
            continue

        if args.boost and merge_model is not None:
            depth  = boost_infer(dpt_model, dpt_transform, merge_model, img_bgr, device)
            suffix = ""
        else:
            h, w   = img_bgr.shape[:2]
            depth  = dpt_infer(dpt_model, dpt_transform, img_bgr, device)
            depth  = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            suffix = ""

        out_name = img_path.stem + suffix + ".png"
        out_path = output_dir / out_name
        save_depth(depth, out_path, save_vis=args.save_vis)

        print(f"  [{i:04d}/{len(images):04d}] {img_path.name} -> {out_name}  "
              f"depth range [{depth.min():.3f}, {depth.max():.3f}]")

    print(f"\nDone. Depth maps saved to: {output_dir}")


if __name__ == "__main__":
    main()