#!/usr/bin/env bash
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --job-name=basic
#SBATCH --time=4:00:00
#SBATCH --begin=now
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=slurm@example.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=saved/preprocessing_logs/%j_%n_%x.txt
# format like node+jobname+user+jobid

# check out examples https://github.com/accre/SLURM
# or this one https://github.com/statgen/SLURM-examples
# https://github.com/cdt-data-science/cluster-scripts

DATA_DIR="data/raw/matterport/v1"
SAVE_DIR="data/processed/matterport"
GIT_REPO="data/raw/matterport/Matterport"
LABEL_DB="data/processed/scannet/label_database.yaml"

if [ ! -d "$GIT_REPO" ]; then
    git clone https://github.com/niessner/Matterport.git $GIT_REPO
fi

# stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
command_exists () {
    type "$1" &> /dev/null ;
}

if [[ -e ".env" ]]; then
    echo "sourcing platform-specific environment file"
    source .env
fi

function preprocess() {
    poetry run \
        python mix3d/datasets/preprocessing/matterport_preprocessing.py preprocess_sequential \
            --data_dir="$DATA_DIR" \
            --save_dir="$SAVE_DIR" \
            --git_repo="$GIT_REPO" \
            --label_db="$LABEL_DB"
}

function make_instance_database() {
    poetry run \
        python mix3d/datasets/preprocessing/matterport_preprocessing.py make_instance_database_sequential \
            --train_database_path="$SAVE_DIR/train_database.yaml" \
            --git_repo="$GIT_REPO" \
            --data_dir="$DATA_DIR" \
            --save_dir=$SAVE_DIR \
            --label_db="$LABEL_DB"
}

function preprocess_dvc () {
    python -m dvc run \
        -d $LABEL_DB \
        -d mix3d/datasets/preprocessing/matterport_preprocessing.py \
        -o "${SAVE_DIR}/train" \
        -o "${SAVE_DIR}/validation" \
        -o "${SAVE_DIR}/test" \
        -o "${SAVE_DIR}/train_database.yaml" \
        -o "${SAVE_DIR}/test_database.yaml" \
        -o "${SAVE_DIR}/validation_database.yaml" \
        -n "matterport" \
        \
        poetry run \
            python mix3d/datasets/preprocessing/matterport_preprocessing.py preprocess \
            --data_dir="$DATA_DIR" \
            --save_dir="$SAVE_DIR" \
            --git_repo="$GIT_REPO" \
            --label_db="$LABEL_DB"
        # -d $DATA_DIR \
        # -d $GIT_REPO \
}

function make_instance_database_dvc () {
    python -m dvc run \
        -d "${SAVE_DIR}/train_database.yaml" \
        -d mix3d/datasets/preprocessing/matterport_preprocessing.py \
        -o "${SAVE_DIR}/instance_database.yaml" \
        -o "${SAVE_DIR}/instances" \
        -d $LABEL_DB \
        -n "matterport_instance" \
        \
        poetry run \
            python mix3d/datasets/preprocessing/matterport_preprocessing.py make_instance_database \
            --train_database_path="$SAVE_DIR/train_database.yaml" \
            --git_repo="$GIT_REPO" \
            --data_dir="$DATA_DIR" \
            --save_dir="$SAVE_DIR" \
            --label_db="$LABEL_DB"
}

function reproduce () {
    python -m dvc repro $1
}

while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do case $1 in
    -h | --help )
    echo "
    -h(--help): to see this message;
    -rd(--run-dvc): run matterport preprocess and track to dvc;
    -pa(--reproduce-all): to reproduce dvc pipeline;
    -r(--run) run preprocess in sequential manner;

    ---
    Variables:
    DATA_DIR="data/raw/matterport/v1"
    SAVE_DIR="data/processed/matterport"
    GIT_REPO="data/raw/matterport/Matterport"
    LABEL_DB="data/processed/scannet/label_database.yaml"
    "
    ;;
    -rd | --run-dvc )
    echo "Run full matterport preparation using dvc"
    preprocess_dvc
    ;;
    -pa | --reproduce-all )
    echo "Reproducing dvc pipeline"
    reproduce $2
    ;;
    -r | --run )
    echo "run preprocess of matterport in sequential manner"
    preprocess
    ;;
esac; shift; done
if [[ "$1" == '--' ]]; then shift; fi
