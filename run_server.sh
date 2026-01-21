#!/bin/bash

#SBATCH --job-name=yolo
#SBATCH --partition=paula
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:a30
#SBATCH --time=47:59:00
#SBATCH --mem=40G

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0
source venv/bin/activate
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0
pip install ultralytics

python src/optuna_tune.py

