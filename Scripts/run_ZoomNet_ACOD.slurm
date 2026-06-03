#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="ZoomNet-Apple-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output="./output/ACOD/ZoomNet.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END

cd /home/aaron.dichoso/scratch1
ls
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi


module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda -V

conda activate ZoomNet

srun python Networks/ZoomNet/main.py --config=Networks/ZoomNet/configs/zoomnet/apples.py --datasets-info=Networks/ZoomNet/configs/_base_/dataset/dataset_configs.json

conda deactivate
