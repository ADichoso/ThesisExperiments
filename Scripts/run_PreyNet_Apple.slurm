#!/bin/bash
#SBATCH --partition="gpu_a100"
#SBATCH --qos="gpu-a100_default"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name="PreyNet-APPLE"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output="./output/Apple/PreyNet.txt"
#SBATCH --mail-user=aaron_gabrielle_dichoso@dlsu.edu.ph
#SBATCH --mail-type=END

cd ..

export HOME=/scratch1/$USER
export CONDA_CONFIG_FILE=/dev/null

ls
#echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

#export CUDA_HOME=$HOME/cuda-11.3
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


#module load anaconda
#source $(conda info --base)/etc/profile.d/conda.sh
#conda -V

#conda activate base

#srun python Networks/PreyNet/trainnew.py

srun Environments/HitNet/bin/python Networks/PreyNet/trainnew.py --dataset="PApple_RGB-D-Size"

#conda deactivate
