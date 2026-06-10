#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="PopNet-APPLE"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output="./output/Apple/PopNet.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END


echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi
nvcc --version

cd ..

module load anaconda
conda init
conda activate PopNet

srun python Networks/PopNet/train.py --dataset="PApple_RGB-D-Size"

conda deactivate
