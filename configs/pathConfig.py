from dataclasses import dataclass
import os
import sys
import time
from utils.exceptions import NoExperimentFound, NoCheckpointFound

@dataclass
class PathConfig:
    root_path: str
    data_dir: str = "data"
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    expr_dir: str = "experiments"

    # Below is experiment-wise path
    train_dir: str = "runs"
    configs_dir: str = "configs"
    logging_dir: str = "logging"
    logger_filename = "train.log"
    tb_dir: str = "tensorboard"
    models_dir: str = "checkpoints"
    result_dir: str = "results"
    metrics_dir: str = "metrics"

    def __post_init__(self):
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Root path {self.root_path} not found")
        self.data_dir = os.path.join(self.root_path, self.data_dir)
        self.expr_dir = os.path.join(self.root_path, self.expr_dir)
        self.ensure_dir(self.data_dir)
        self.raw_dir = os.path.join(self.data_dir, self.raw_dir)
        self.processed_dir = os.path.join(self.data_dir, self.processed_dir)
        self.ensure_dir(self.raw_dir)
        self.ensure_dir(self.processed_dir)
        self.train_raw_path = os.path.join(self.raw_dir, "train.parquet")
        self.test_raw_path = os.path.join(self.raw_dir, "test.parquet")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.train_processed_path = os.path.join(self.processed_dir, "train.parquet")
        self.test_processed_path = os.path.join(self.processed_dir, "test.parquet")
        self.ensure_dir(self.expr_dir)

    def ensure_dir(self, path: dir):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def create_expr(self, expr_name: str = None):
        if expr_name is None:
            expr_name = time.strftime("%Y%m%d-%H%M%S")

        if not hasattr(self, "expr_name"):
            self.expr_name = expr_name
        self.expr_specific_dir = os.path.join(self.expr_dir, self.expr_name)
        self.train_dir = os.path.join(self.expr_specific_dir, self.train_dir)

        self.configs_dir = os.path.join(self.train_dir, self.configs_dir)
        self.logging_dir = os.path.join(self.train_dir, self.logging_dir)
        self.logging_path = os.path.join(self.logging_dir, self.logger_filename)
        self.tb_dir = os.path.join(self.train_dir, self.tb_dir)
        self.models_dir = os.path.join(self.train_dir, self.models_dir)
        self.results_dir = os.path.join(self.expr_specific_dir, self.result_dir)
        self.metrics_dir = os.path.join(self.results_dir, self.metrics_dir)

        self.ensure_dir(self.train_dir)
        self.ensure_dir(self.expr_dir)
        self.ensure_dir(self.configs_dir)
        self.ensure_dir(self.logging_dir)
        self.ensure_dir(self.tb_dir)
        self.ensure_dir(self.models_dir)
        self.ensure_dir(self.results_dir)
        self.ensure_dir(self.metrics_dir)

    def find_latest_expr(self):
        expr_dirs = os.listdir(self.expr_dir)
        expr_dirs = [d for d in expr_dirs if os.path.isdir(os.path.join(self.expr_dir, d))]
        expr_dirs = sorted(expr_dirs, reverse=True)
        if len(expr_dirs) == 0:
            raise NoExperimentFound()
        latest_expr = expr_dirs[0]
        self.create_expr(latest_expr)
        return latest_expr


    def find_latest_epoch(self, expr_name: str = None):
        if expr_name is None:
            expr_name = self.find_latest_expr()
        ckpt_files = os.listdir(self.models_dir)
        ckpt_files = [f for f in ckpt_files if f.startswith("ckpt_") and f.endswith(".pth")]
        ckpt_files = [f.split(".")[0] for f in ckpt_files]
        ckpt_files = [int(f.split("_")[1]) for f in ckpt_files]
        if len(ckpt_files) == 0:
            raise NoCheckpointFound()
        return max(ckpt_files)
