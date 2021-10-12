#!/usr/bin/env bash

DVC_CACHE_DIR="/work/$USERNAME/data/dvc_cache"
SCANNET_RAW_DIR="/work/$USERNAME/data/scannet"

# install python evironment
pyenv install 3.7.7
pyenv virtualenv 3.7.7 mix3d
pyenv local mix3d
pyenv activate

# install all dependencies
python -m pip install poetry
python -m poetry install

mkdir .dvc
echo "[cache]\n\tdir = $DVC_CACHE_DIR\n\ttype = 'symlink'" > ./.dvc/config.local
ln -s $SCANNET_RAW_DIR ./data/raw/scannet

# for github
pre-commit install
dvc checkout

echo "for visualizations also install orca 1.2.0 \
    https://github.com/plotly/orca/releases/tag/v1.2.0"

# for running the code
echo "NEPTUNE_API_TOKEN='$NEPTUNE_API_TOKEN'" > .env
