#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --error=log_10.err
#SBATCH --output=log_10.out
#SBATCH --job-name=SHAP_10 
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.biswas.0060@gmail.com

# Display assigned nodes
echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"

# Load required modules
module load anaconda cuda/cuda-12.1

# Activate Conda environment
conda activate ML_EXP_AI

# Run the Python script
python SHAP_main_3.py --job_id 10 --start_idx 2000 --n_seq 1000 &

# Wait for background processes to finish
wait
