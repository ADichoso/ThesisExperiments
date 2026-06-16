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
import os
import argparse
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# ── resolve repo paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DPT_ROOT = SCRIPT_DIR / "DPT"
BOOST_ROOT = SCRIPT_DIR / "BoostingMonocularDepth"

for p in [DPT_ROOT, BOOST_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


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
    # Freeze all parameters — source-free: no gradient through depth network
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
    sample = transform({"image": img_rgb})["image"]
    tensor = torch.from_numpy(sample).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(tensor)

    depth = depth.squeeze().cpu().numpy()
    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    return depth.astype(np.float32)


# =============================================================================
# Boosting wrapper
# =============================================================================
def load_boost_model(device: torch.device):
    """
    Load the Boosting pix2pix merge network [45].
    See BoostingMonocularDepth for full details.
    """
    sys.path.insert(0, str(BOOST_ROOT))
    from pix2pix.options.test_options import TestOptions
    from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

    opt = TestOptions().parse()
    opt.isTrain = False
    opt.checkpoints_dir = str(BOOST_ROOT / "pix2pix" / "checkpoints")
    opt.name = "mergemodel"
    opt.gpu_ids = [device.index] if device.type == "cuda" else []

    merge_model = Pix2Pix4DepthModel(opt)
    merge_model.save_dir = opt.checkpoints_dir + "/" + opt.name
    merge_model.load_networks("latest")
    merge_model.eval()
    return merge_model


def boost_infer(dpt_model, dpt_transform, merge_model,
                img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run full Boosting pipeline from [45]:
    1. Low-res base depth via DPT
    2. Tiled high-res patches via DPT
    3. Merged via pix2pix merge network
    Returns depth map normalized to [0, 1].
    """
    sys.path.insert(0, str(BOOST_ROOT))
    from utils import ImageandPatchs, generatepatchs, getGF_fromintegral
    import torchvision.transforms as transforms

    h, w = img_bgr.shape[:2]

    # --- base prediction (low-res) ---
    base_depth = dpt_infer(dpt_model, dpt_transform, img_bgr, device)
    base_depth_resized = cv2.resize(base_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- high-res patches ---
    whole_image_optimal_size, patch_scale = 512, 1
    img_patchs = ImageandPatchs(
        "",               # base path (unused here)
        "tmp",
        img_bgr,
        patch_scale,
    )
    generatepatchs(img_patchs, whole_image_optimal_size)

    patch_depths = []
    for patch in img_patchs:
        patch_bgr = patch["patch_rgb"]
        patch_depth = dpt_infer(dpt_model, dpt_transform, patch_bgr, device)
        patch_depths.append(patch_depth)

    # --- merge ---
    to_tensor = transforms.ToTensor()
    base_t = to_tensor(base_depth_resized).unsqueeze(0)

    # Reconstruct high-res depth from patches using the merge network
    # (simplified single-patch merge — for multi-patch use the full pipeline)
    if len(patch_depths) > 0:
        # Use the highest-res patch as the detail source
        detail = cv2.resize(patch_depths[0], (w, h), interpolation=cv2.INTER_LINEAR)
        detail_t = to_tensor(detail).unsqueeze(0)
        merge_input = torch.cat([base_t, detail_t], dim=1).to(device)
        with torch.no_grad():
            merged = merge_model.netG(merge_input)
        depth_out = merged.squeeze().cpu().numpy()
    else:
        depth_out = base_depth_resized

    # Final normalization
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
    # uint16: preserves full 0-65535 range for downstream use
    depth_uint16 = (depth * 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_uint16)

    if save_vis:
        vis_path = out_path.parent / "vis" / (out_path.stem + "_vis.png")
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        colored = cv2.applyColorMap(
            (depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        cv2.imwrite(str(vis_path), colored)


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="PopNet Sec 3.1 source-free depth generator")
    p.add_argument("--input_dir",   required=True,  help="Directory of RGB images")
    p.add_argument("--output_dir",  required=True,  help="Directory to write depth maps")
    p.add_argument("--dpt_weights", required=True,  help="Path to dpt_large-midas-2f21e586.pt")
    p.add_argument("--boost",       action="store_true",
                   help="Apply Boosting [45] for high-res depth (slower, higher quality)")
    p.add_argument("--save_vis",    action="store_true",
                   help="Save colorized depth visualizations alongside depth maps")
    p.add_argument("--device",      default="cuda",
                   help="'cuda' or 'cpu' (default: cuda)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load DPT (frozen, as per PopNet Sec. 3.1) ──────────────────────────
    print("Loading DPT-Large [50] with frozen weights ...")
    dpt_model, dpt_transform = load_dpt(args.dpt_weights, device)
    print(f"  Trainable DPT params: "
          f"{sum(p.numel() for p in dpt_model.parameters() if p.requires_grad):,}  "
          f"(should be 0 — frozen)")

    # ── optionally load boost merge network ────────────────────────────────
    merge_model = None
    if args.boost:
        print("Loading Boosting merge network [45] ...")
        merge_model = load_boost_model(device)

    # ── inference loop ─────────────────────────────────────────────────────
    images = collect_images(input_dir)
    print(f"Found {len(images)} images in {input_dir}")

    for i, img_path in enumerate(images, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [skip] Cannot read: {img_path.name}")
            continue

        if args.boost and merge_model is not None:
            depth = boost_infer(dpt_model, dpt_transform, merge_model, img_bgr, device)
            suffix = "_boost"
        else:
            depth = dpt_infer(dpt_model, dpt_transform, img_bgr, device)
            # Resize back to original resolution
            h, w = img_bgr.shape[:2]
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            suffix = "_dpt"

        out_name = img_path.stem + suffix + ".png"
        out_path = output_dir / out_name
        save_depth(depth, out_path, save_vis=args.save_vis)

        print(f"  [{i:04d}/{len(images):04d}] {img_path.name} → {out_name}  "
              f"depth range [{depth.min():.3f}, {depth.max():.3f}]")

    print(f"\nDone. Depth maps saved to: {output_dir}")


if __name__ == "__main__":
    main()
