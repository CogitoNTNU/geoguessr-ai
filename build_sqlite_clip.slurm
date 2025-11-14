#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="sqlite-clip"
#SBATCH --time=03-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/output_sqlite_clip.txt
#SBATCH --error=slurm_outputs/output_sqlite_clip.err
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

uv run python - <<'PY'
import json
from backend.s3bucket import create_and_upload_sqlite_clip_embeddings_from_latest_snapshot as run

res = run(
    device='cuda',
    num_workers=256,
    embed_batch_size=512,
    commit_interval=20000,
    max_rows=None,
)
print(json.dumps(res, indent=2))
PY


