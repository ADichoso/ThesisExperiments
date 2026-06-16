#!/bin/bash
# =============================================================================
# PopNet Section 3.1 — Source-free Depth Network Setup
# Pipeline: DPT (frozen) + Boosting for high-res depth maps
# Refs: [50] Ranftl et al., ICCV 2021 | [45] Miangoleh et al., CVPR 2021
# =============================================================================

set -e
WORK_DIR="$HOME/popnet_depth"
mkdir -p "$WORK_DIR" && cd "$WORK_DIR"

echo "===> [1/4] Cloning DPT (Vision Transformers for Dense Prediction)"
if [ ! -d "DPT" ]; then
    git clone https://github.com/isl-org/DPT.git
fi

echo "===> [2/4] Cloning Boosting Monocular Depth"
if [ ! -d "BoostingMonocularDepth" ]; then
    git clone https://github.com/compphoto/BoostingMonocularDepth.git
fi

echo "===> [3/4] Installing dependencies"
pip install timm==0.6.13 --break-system-packages
pip install opencv-python-headless matplotlib --break-system-packages

echo "===> [4/4] Downloading DPT-Large pretrained weights"
mkdir -p DPT/weights
# DPT-Large trained on MIX6 — highest generalization (ref [51])
WEIGHT_URL="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt"
if [ ! -f "DPT/weights/dpt_large-midas-2f21e586.pt" ]; then
    wget -q --show-progress -O DPT/weights/dpt_large-midas-2f21e586.pt "$WEIGHT_URL"
    echo "DPT weights downloaded."
else
    echo "DPT weights already present, skipping."
fi

echo ""
echo "Setup complete. Directory: $WORK_DIR"
echo "Next: run popnet_depth_infer.py"
