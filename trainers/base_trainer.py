from abc import ABC, abstractmethod
import logging
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
from utils.train_val_split import train_val_split
from utils.dualfeature_dataLoader import DualFeature_Dataset, DualFeature_DataLoader
from utils.exceptions import NoExperimentFound, NoCheckpointFound

class BaseTrainer(ABC):
    def __init__(self, model_config, train_config, data_config, path_config):
        self.model_config = model_config
        self.train_config = train_config
        self.data_config = data_config
        self.path_config = path_config
        self._parse_params()

    # TODO: Check if there is a more brilliant way to organize the parse and init
    def _parse_params(self):
        self.feature_engineering = self.data_config.feature_engineering
        self.train_data = self.path_config.train_processed_path if self.feature_engineering else self.path_config.train_raw_path
        self.test_data = self.path_config.test_processed_path if self.feature_engineering else self.path_config.test_raw_path
        self.val_ratio = self.data_config.val_size
        self.num_epochs = self.train_config.num_epochs
        self.save_freq = self.train_config.save_freq
        self.continue_learning = self.train_config.continue_learning
        self.early_stop_patience = self.train_config.early_stop_patience
        self.device = self.train_config.device

    def _init(self):
        # TODO: could specify the optimizer and scheduler params
        self.build_model()
        if not hasattr(self, "model") or self.model is None:
            raise NotImplementedError("Model not built")
        self.optimizer = self.train_config.get_optimizer(self.model)
        self.scheduler = self.train_config.get_shceduler(self.optimizer)
        self.loss_fn = self.train_config.get_loss_fn()
        self.epoch = 0
        self.early_stopped = False
        self.early_stop_count = 0
        self.best_epoch = 0
        self.best_loss = float("inf")
        self.loss = 0
        self.path_config.create_expr()
        if not hasattr(self, "logger"):
            self.init_logging()
        if not hasattr(self, "writer"):
            self.init_tensorboard()

    def init_logging(self):
        logging.basicConfig(
            filename=self.path_config.logging_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger()

    def init_tensorboard(self):
        self.writer = SummaryWriter(self.path_config.tb_dir)

    def save_config(self):
        self.data_config.to_json(self.path_config.configs_dir)
        self.train_config.to_json(self.path_config.configs_dir)
        self.model_config.to_json(self.path_config.configs_dir)

    def save_model(self, best: bool = False):
        ckpt = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "early_stopped": self.early_stopped,
            "early_stop_count": self.early_stop_count,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
        }
        # TODO: If best is ckpt, need to solve mismatch
        if best:
            ckpt_name = "ckpt_best.pth"
        else:
            ckpt_name = f"ckpt_{self.epoch}.pth"
        ckpt_path = os.path.join(self.path_config.models_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_model(self, expr: str = None, epoch: int = None, best: bool = False):
        # TODO: Set default to continue learning at the latest checkpoint
        if expr is None:
            expr = self.path_config.find_latest_expr()
        self.path_config.create_expr(expr)
        if best:
            model_path = os.path.join(
                self.path_config.models_dir, "ckpt_best.pth")
        else:
            if epoch is None:
                epoch = self.path_config.find_latest_epoch()
            model_path = os.path.join(
                self.path_config.models_dir, f"ckpt_{epoch}.pth")

        if not os.path.exists(model_path):
            raise NoCheckpointFound()
        ckpt = torch.load(model_path)
        self._init()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.early_stopped = ckpt["early_stopped"]
        self.early_stop_count = ckpt["early_stop_count"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_loss = ckpt["best_loss"]

    def handle_early_stopped(self):
        if self.early_stopped:
            self.logger.warning(
                f"Latest Experiment {self.path_config.expr_name} has early stopped")
            new_expr = input(
                f"Latest Experiment {self.path_config.expr_name} has early stopped, start a new experiment? (y/n) ")
            if new_expr.lower() == "y":
                self._init()
                self.logger.info("Start training from scratch")
                self.logger.info(
                    f"New experiment created: {self.path_config.expr_name}")
            else:
                self.logger.info("Refuse to start a new experiment, exit.")
                sys.exit(1)

    def prepare_learning(self):
        if self.continue_learning:
            if self.train_config.continue_expr is None:
                try:
                   latest_expr = self.path_config.find_latest_expr()
                except NoExperimentFound:
                    latest_expr = None
                if latest_expr is None:
                    self._init()
                    self.logger.info("No experiment found, Start training from scratch")
                    self.logger.info(f"New experiment created: {self.path_config.expr_name}")
                else:
                    self.load_model(expr=latest_expr, best=False)
                    self.handle_early_stopped()
                    self.logger.info(
                        f"Model loaded, Continue learning from {latest_expr}, Epoch: {self.epoch}")
            else:
                self.load_model(
                    expr=self.train_config.continue_expr, best=False)
                self.handle_early_stopped()

                self.logger.info(
                    f"Model loaded, Continue learning from {self.train_config.continue_expr}, Epoch: {self.epoch}")
        else:
            self._init()
            self.logger.info("Continue learning is False, Start training from scratch")
            self.logger.info(
                f"New experiment created: {self.path_config.expr_name}")
        self.save_config()
        self.logger.info("Config saved")

    def load_train_data(self):
        train, val = train_val_split(self.train_data, self.val_ratio)
        train_dataset = DualFeature_Dataset(
            train, self.data_config.seq_len, self.data_config.stride)
        val_dataset = DualFeature_Dataset(
            val, self.data_config.seq_len, self.data_config.stride)
        train_loader = DualFeature_DataLoader(
            train_dataset, self.train_config.batch_size, shuffle=True, drop_last=True)
        val_loader = DualFeature_DataLoader(
            val_dataset, self.train_config.batch_size, shuffle=True, drop_last=True)
        if not hasattr(self, "train_loader"):
            self.train_loader = train_loader
        if not hasattr(self, "val_loader"):
            self.val_loader = val_loader

    def load_test_data(self):
        test = pd.read_parquet(self.test_data)
        test_dataset = DualFeature_Dataset(
            test, self.data_config.seq_len, self.data_config.stride)
        test_loader = DualFeature_DataLoader(
            test_dataset, self.train_config.batch_size, shuffle=True, drop_last=True)
        if not hasattr(self, "test_loader"):
            self.test_loader = test_loader

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def build_model(self):
        pass
