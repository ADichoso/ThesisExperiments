#!/bin/bash
#SBATCH --partition="gpu"
#SBATCH --qos="gpu-p40_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="TEST-SINetV2-ACOD-1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --output="SINetV2_ACOD_1_test.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END

cd /home/aaron.dichoso/scratch1
ls
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

export CUDA_HOME=$HOME/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda -V

conda activate TrainingApex

srun python SINetV2/MyTesting.py

conda deactivate
