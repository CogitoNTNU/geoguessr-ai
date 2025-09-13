#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="geoguessr-ai"
#SBATCH --time=03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --constraint="(p100|a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/output_geo-guessr-ai.txt
#SBATCH --error=slurm_outputs/output_geo-guessr-ai.err
#SBATCH --mail-user=danielnh@stud.ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$(whoami)/geoguessr-ai/cv-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.12

# Activate the environment
conda activate ${ENV_PATH}
pip install -r requirements.txt

echo "Downloaded dependencies:"
pip freeze


# Run the Python script
python main.py

# Deactivate the environment
conda deactivate

echo "Job finished"

# "sbatch ./train.slurm"  to run the job