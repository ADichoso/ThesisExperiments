#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="OCENet-ACOD-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --output="OCENet_ACOD_1.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END


echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda
conda init
conda activate Python36

#srun torchrun --nproc_per_node=1 --master_port=3000 OCENet/train.py
#srun python -m torch.distributed.launch --nproc_per_node=1 --master_port=3000 OCENet/train.py
srun python OCENet/test.py

conda deactivate
