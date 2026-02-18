#!/bin/bash
#SBATCH --account=def-scoulombe
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --job-name=nerf-train
#SBATCH --output=nerf-%j.out

###### %SBATCH --gpus=a100_4g.20gb:1
echo "job run at $(date +"%T")"

# Load modules BEFORE creating/activating venv
module load cuda/12.6 python/3.11 opencv/4.12

# Create and activate venv
virtualenv -q --no-download $SLURM_TMPDIR/env_sogs
source $SLURM_TMPDIR/env_sogs/bin/activate
pip install -q --no-index --upgrade pip

# Install build & deps
pip install -q --no-index build -r requirements.txt


## Rebuild and install local modules
#mkdir -p ./dist
#python -m build --no-isolation submodules/diff-gaussian-rasterization -w --outdir ./dist
#python -m build --no-isolation submodules/simple-knn -w --outdir ./dist
#python -m build --no-isolation submodules/PLAS -w --outdir  ./dist


# Install your wheels
pip install -q --no-index ./dist/*.whl

export WANDB_MODE=offline

### Run training (Hydra overrides MUST be quoted)
##python train.py \
##  --config-name ours_q_sh_local_test \
##  'hydra.run.dir=./output/${now:%Y-%m-%d}/${now:%H-%M-%S}-${run.name}' \
##  dataset.source_path=/home/erikmam/projects/def-scoulomb/erikmam/mipnerf360/garden \
##  run.no_progress_bar=false \
##  run.name=vs-code-debug


python -m SOGS.playground.main