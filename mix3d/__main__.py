import logging
import os
from datetime import datetime
from hashlib import md5
from uuid import uuid4

import hydra
import torch
from dotenv import load_dotenv
from git import Repo
from omegaconf import DictConfig, OmegaConf
from mix3d import __version__
from mix3d.trainer.trainer import SemanticSegmentation
from mix3d.utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    # making shure reproducable
    assert not Repo("./").is_dirty(), "repo is dirty, commit first"
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    cfg.general.experiment_id = str(Repo("./").commit())[:8]
    params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    unique_id = "_" + str(uuid4())[:4]
    cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    for log in cfg.logging:
        if "NeptuneLogger" in log._target_:
            loggers.append(
                hydra.utils.instantiate(
                    log, api_key=os.environ.get("NEPTUNE_API_TOKEN"), params=params,
                )
            )
            if "offline" not in loggers[-1].version:
                cfg.general.version = loggers[-1].version
        else:
            loggers.append(hydra.utils.instantiate(log))
            loggers[-1].log_hyperparams(
                flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            )

    model = SemanticSegmentation(cfg)
    if cfg.general.checkpoint is not None:
        if cfg.general.checkpoint[-3:] == "pth":
            # loading model weights, if it has .pth in the end, it will work with it
            # as if it work with original Minkowski weights
            cfg, model = load_baseline_model(cfg, SemanticSegmentation)
        else:
            cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))
    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(config_path="conf", config_name="config.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.test(model)
