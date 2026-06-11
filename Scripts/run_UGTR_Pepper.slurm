#!/bin/bash
#SBATCH --partition="gpu"
#SBATCH --qos="gpu-p40_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="UGTR-PEPPER"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output="./output/PEPPER/UGTR.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END

cd /home/aaron.dichoso/scratch1
ls
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi


module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda -V

conda activate base

srun /scratch1/aaron.dichoso/Environments/MGL/bin/python Networks/UGTR/trainnew.py --dataset="Sweet_Pepper"

conda deactivate

