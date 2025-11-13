#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="sqlite-duplicate"
#SBATCH --time=00-01:00:00
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=16  # 2 CPU cores
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_sqlite_duplicate.txt
#SBATCH --error=slurm_outputs/output_sqlite_duplicate.err
#SBATCH --mail-user=jmberget@ntnu.no
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

uv run python -m training.duplicate_sqlites.py


