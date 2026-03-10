#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="proto-test"
#SBATCH --time=03-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=4  # 4 CPU cores
#SBATCH --mem=200GB
#SBATCH --nodes=1
# Use Slurm placeholders: %j=jobid, %x=jobname, %A=array jobid, %a=array index
#SBATCH --output=slurm_outputs/proto_%j.txt
#SBATCH --error=slurm_outputs/proto_%j.err
#SBATCH --mail-user=danielnh@stud.ntnu.no
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

# Run the main.py script
uv run main.py 