<div align="center">
# Mix3D: Out-of-Context Data Augmentation for 3D Scenes (3DV 2021)

Alexey Nekrasov*, Jonas Schult*, Or Litany, Bastian Leibe, Francis Engelmann

Mix3D is a data augmentation technique for 3D segmentation methods that improves generalization.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mix3d-out-of-context-data-augmentation-for-3d/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=mix3d-out-of-context-data-augmentation-for-3d)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>

![teaser](./docs/github_teaser.jpg)

</div>
<br><br>

## News

* **12. October 2021**: Code released.
* **6. October 2021**: Mix3D accepted for oral presentation at [3DV](https://3dv2021.surrey.ac.uk/) 2021. Paper on [[arXiv]](https://arxiv.org/abs/2110.02210).
* **30. July 2021**: Mix3D ranks 1st on the [ScanNet](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_label_3d) semantic labeling benchmark.

![Learderboard](./docs/scannet_benchmark_screenshot.png)


## Running the code
This repository contains the code for analysis experiments for the section `4.2. Motivation and Analysis Experiments` from the paper.
For the ScanNet benchmark and validation training we use the original <a href="https://github.com/chrischoy/SpatioTemporalSegmentation-Scannet"><SpatioTemporalSegmentation-Scannet></a> code, to add Mix3D we provide `SpatioTemporalSegmentation.patch` patch file. Check the supplementary for more details.


### Code structure

```
├── mix3d
│   ├── __init__.py
│   ├── __main__.py     <- the main file
│   ├── conf            <- hydra configuration files
│   ├── datasets
│   │   ├── outdoor_semseg.py       <- outdoor dataset
│   │   ├── preprocessing       <- folder with preprocessing scripts
│   │   ├── semseg.py       <- indoor dataset
│   │   └── utils.py        <- code for mixing point clouds
│   ├── logger
│   ├── models      <- MinkowskiNet models
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py      <- train loop
│   └── utils
├── data
│   ├── processed       <- folder for preprocessed datasets
│   └── raw     <- folder for raw datasets
├── scripts
│   ├── experiments
│   │   └── 1000_scene_merging.bash
│   ├── init.bash
│   ├── local_run.bash
│   ├── preprocess_matterport.bash
│   ├── preprocess_rio.bash
│   ├── preprocess_scannet.bash
│   └── preprocess_semantic_kitti.bash
├── docs
├── dvc.lock
├── dvc.yaml        <- dvc file to reproduce the data
├── poetry.lock
├── pyproject.toml      <- project dependencies
├── README.md
├── saved       <- folder that stores models and logs
└── SpatioTemporalSegmentation-ScanNet.patch        <- patch file for original repo
```

### Dependencies
The main dependencies of the project are the following:
```yaml
python: 3.7
cuda: 10.1
```
for others the project uses [poetry](https://github.com/python-poetry/poetry) dependency management package. And everything could be installed with a command:
```yaml
poetry install
```
check `scripts/init.bash` for more details.

### Data preprocessing
After dependencies are installed it is important to run preprocessing scripts, they would bring `scannet, matterport, rio, semantic_kitti` datasets to a single format. By default, scripts expect to find datsets in `data/raw/` folder. Check `scripts/preprocess_*.bash` for more details.
```yaml
dvc repro scannet # matterport, rio, semantic_kitti
```
This command will run the preprocessing for scannet and will save the result using `dvc` data versioning system.

### Training and testing
Train network on scannet dataset without Mix3D with a voxel size of 5cm:
```yaml
poetry run train
```
Train network on scannet dataset with Mix3D with a voxel size of 5cm:
```yaml
poetry run train data/collation_functions=voxelize_collate_merge
```

## BibTeX
```
@inproceedings{Nekrasov213DV,
  title     = {{Mix3D: Out-of-Context Data Augmentation for 3D Scenes}},
  author    = {Nekrasov, Alexey and Schult, Jonas and Litany, Or and Leibe, Bastian and Engelmann, Francis},
  booktitle = {{International Conference on 3D Vision (3DV)}},
  year      = {2021}
}
```
