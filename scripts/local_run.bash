#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --job-name=basic
#SBATCH --time=96:00:00
#SBATCH --partition=example
#SBATCH --begin=now
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=slurm@example.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=saved/slurm_logs/%j_%n_%x.txt
# format like node+jobname+user+jobid

# check out examples https://github.com/accre/SLURM
# or this one https://github.com/statgen/SLURM-examples
# https://github.com/cdt-data-science/cluster-scripts

if [[ -e ".env" ]]; then
    echo "sourcing platform-specific environment file"
    source .env
fi

command_exists () {
    # https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
    type "$1" &> /dev/null ;
}

# https://github.com/StanfordVL/MinkowskiEngine/issues/121
export OMP_NUM_THREADS=12

if command_exists poetry ; then
    echo "Using poetry"
    srun poetry run train $@
else
    echo "Poetry not installed, use on your own risk!"
    srun python -m mix3d.train $@
fi
