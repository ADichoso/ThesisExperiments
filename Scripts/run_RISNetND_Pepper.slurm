#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="RISNetND-PEPPER-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output="./output/Pepper/RISNetND.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END


echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda/3-2024
conda init
conda activate RISNet

cd ..

srun python Networks/RISNet_NoDepth/Train.py --dataset="Sweet_Pepper"

conda deactivate
