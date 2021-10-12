import json
import io
import os
import argparse
from typing import Any, Dict, Optional, Union


from pytorch_lightning.loggers.base import LightningLoggerBase


class JsonLogger(LightningLoggerBase):
    @property
    def experiment(self) -> Any:
        pass

    def log_hyperparams(self, params: argparse.Namespace):
        with open(self.json_path, "a") as f:
            json.dump(params, f)

    @property
    def version(self) -> Union[int, str]:
        pass

    def __init__(self, json_path: str = "json_log.json"):
        super().__init__()
        self.json_path = json_path

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        metrics.update({"step": step})
        with open(self.json_path, "a") as f:
            json.dump(metrics, f)

    @property
    def name(self):
        return "JsonLogger"
