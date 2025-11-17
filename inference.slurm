#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="geoguessr-ai"
#SBATCH --time=05-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu80g)"
#SBATCH --output=slurm_outputs/output_geo-guessr-ai-tinyvit-panorama-3.txt
#SBATCH --error=slurm_outputs/output_geo-guessr-ai-tinyvit-panorama-3.err
#SBATCH --mail-user=magnuvbr@stud.ntnu.no,danielnh@stud.ntnu.no,jens.m.n.berget@stud.ntnu.no
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

# Run the Python script
uv run python inference.py
