#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="ZoomNet-ACOD-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --output="Logs/Train/ZoomNet_ACOD_1.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END

cd /home/aaron.dichoso/scratch1
ls
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi


module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda -V

conda activate TrainingApex

srun python Networks/ZoomNet/main.py 

conda deactivate
