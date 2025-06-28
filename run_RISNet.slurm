#!/bin/bash
#SBATCH --partition="gpu"
#SBATCH --qos="gpu-p40_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="RISNet-APPL-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --output="RISNet_test.out"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END


echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda/3-2024
conda init
conda activate RISNet

srun python RISNet/Train.py --train_path="./datasets/detection/detection/train" --no_depth

conda deactivate
