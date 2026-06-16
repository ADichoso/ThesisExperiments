"""
Quick sanity-check for PopNet Section 3.1 depth pipeline.
Run this BEFORE processing your full dataset to verify everything works.

Usage:
    python popnet_depth_test.py \
        --image /path/to/any/test.jpg \
        --dpt_weights ~/popnet_depth/DPT/weights/dpt_large-midas-2f21e586.pt
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "DPT"))


def test_dpt(image_path: str, weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM free   : {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # --- load DPT ---
    from dpt.models import DPTDepthModel
    from dpt.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose

    print(f"\n  Loading DPT-Large from: {weights_path}")
    model = DPTDepthModel(path=weights_path, backbone="vitl16_384", non_negative=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    print(f"  Params (frozen): {sum(p.numel() for p in model.parameters()):,}")

    transform = Compose([
        Resize(384, 384, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method="minimal",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet(),
    ])

    # --- load image ---
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Cannot read image: {image_path}"
    h, w = img_bgr.shape[:2]
    print(f"\n  Image size  : {w}×{h}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    sample  = transform({"image": img_rgb})["image"]
    tensor  = torch.from_numpy(sample).unsqueeze(0).to(device)

    # --- infer ---
    with torch.no_grad():
        depth = model(tensor)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    print(f"  Depth range : [{depth.min():.4f}, {depth.max():.4f}]  (normalized)")
    print(f"  Depth shape : {depth.shape}")

    # --- save outputs ---
    out_uint16 = (depth * 65535).astype("uint16")
    out_vis    = cv2.applyColorMap((depth * 255).astype("uint8"), cv2.COLORMAP_INFERNO)

    cv2.imwrite("test_depth_raw.png", out_uint16)
    cv2.imwrite("test_depth_vis.png", out_vis)
    print("\n  Saved: test_depth_raw.png  (uint16, for training)")
    print("  Saved: test_depth_vis.png  (colorized, for inspection)")
    print("\n  ✓ DPT pipeline OK — ready for full dataset inference")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",       required=True)
    p.add_argument("--dpt_weights", required=True)
    args = p.parse_args()
    test_dpt(args.image, args.dpt_weights)
