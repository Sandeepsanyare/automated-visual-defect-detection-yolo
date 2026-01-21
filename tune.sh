#!/bin/bash

#SBATCH --job-name=yolo-optuna
#SBATCH --partition=paula
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:a30
#SBATCH --time=30:00:00
#SBATCH --mem=40G

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0
source venv/bin/activate

pip install --upgrade pip
pip install ultralytics optuna

python src/optuna_tune.py --trials 20
