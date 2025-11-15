#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="sqlite-raw"
#SBATCH --time=03-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=4  # 2 CPU cores
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_sqlite_raw4.txt
#SBATCH --error=slurm_outputs/output_sqlite_raw4.err
#SBATCH --mail-user=danielnh@stud.ntnu.no,jmberget@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1

pip install uv --user

# Optionally set WANDB and AWS creds here if not already configured:
# export WANDB_API_KEY=...
# export AWS_ACCESS_KEY_ID=...
# export AWS_SECRET_ACCESS_KEY=...
# export AWS_DEFAULT_REGION=eu-north-1

# Run the raw JPEG -> SQLite builder (uses backend/s3bucket.py:main)
uv run python -m backend.s3bucket


