#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jk3417
#SBATCH --output=/vol/bitbucket/jk3417/rl-starter-files/slurm_outputs/minigrid_%j.out
export PATH=/vol/bitbucket/jk3417/venvs/minigridrl/bin/:$PATH
source activate

. /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/11.3.1-cudnn8.2.1
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Using algorithm: $1."
echo "Running task: $2."
# Param 1: config (e.g.  dmlab, atari, crafter, dmc_vision, dmc_proprio, pinpad, loconav)
# Param 2: task (e.g. dmc_walker_walk)

cd /vol/bitbucket/jk3417/rl-starter-files/ && srun python -m scripts.train --algo $1 --env $2 --save-interval 1000 --frames 1000000


