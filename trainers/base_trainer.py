from abc import ABC, abstractmethod
import logging
import os
import sys
import json
import torch
import torch.nn as nn
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
        self.use_early_stop = self.train_config.use_early_stop
        self.early_stop_patience = self.train_config.early_stop_patience if self.use_early_stop else None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _init(self):
        self.build_model()
        if not hasattr(self, "model") or self.model is None:
            raise NotImplementedError("Model not built")
        self.optimizer = self.train_config.get_optimizer(self.model)
        self.scheduler = self.train_config.get_shceduler(self.optimizer)
        self.loss_fn = self.train_config.get_loss_fn()
        self.epoch = 0
        if self.use_early_stop:
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
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
        }
        if self.use_early_stop:
            ckpt["early_stopped"] = self.early_stopped
            ckpt["early_stop_count"] = self.early_stop_count

        if best:
            ckpt_name = "ckpt_best.pth"
        else:
            ckpt_name = f"ckpt_{self.epoch}.pth"
        ckpt_path = os.path.join(self.path_config.models_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_model(self, expr: str = None, epoch: int = None, best: bool = False):
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
        if self.use_early_stop:
            self.early_stopped = ckpt["early_stopped"]
            self.early_stop_count = ckpt["early_stop_count"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_loss = ckpt["best_loss"]

    def handle_early_stopped(self):
        if not self.use_early_stop:
            return
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

    def train_epoch(self):
        self.model.to(self.device)
        self.model.train()
        ttl_loss = 0
        for k_features, a_features, y in self.train_loader:
            k_features, a_features, y = k_features.to(
                self.device), a_features.to(self.device), y.to(self.device)
            X = torch.cat((k_features, a_features), dim=2)
            output = self.model(X)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            if self.train_config.grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_norm)
            self.optimizer.step()
            ttl_loss += loss.item()
        return ttl_loss / len(self.train_loader)

    def train(self):
        self.prepare_learning()
        self.logger.info("Device Selected: " + str(self.device))
        self.load_train_data()
        self.logger.info(
            "Training dataset and validation dataset loaded successfully, start training...")
        pbar = tqdm(range(self.epoch, self.num_epochs))
        for _ in pbar:
            current_epoch = self.epoch
            pbar.set_description(f"Epoch {current_epoch}")

            train_loss = self.train_epoch()
            val_loss = self.eval()
            pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})

            self.scheduler.step(val_loss)
            self.writer.add_scalar("Loss/train", train_loss, current_epoch)
            self.writer.add_scalar("Loss/val", val_loss, current_epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], current_epoch)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = current_epoch
                self.save_model(best=True)
                self.logger.info(
                    "Best model saved at epoch: " + str(current_epoch))
                self.early_stop_count = 0
            else:
                if self.use_early_stop:
                    self.early_stop_count += 1
                    if self.early_stop_count >= self.early_stop_patience:
                        self.early_stopped = True
                        self.logger.warning(
                            "Early stopped at epoch: " + str(current_epoch))
                        break

            if self.epoch % self.save_freq == 0:
                self.save_model()
                self.logger.info(
                    "Checkpoint saved at epoch: " + str(current_epoch))

            self.epoch += 1

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            ttl_loss = 0
            for k_features, a_features, y in self.val_loader:
                k_features, a_features, y = k_features.to(
                    self.device), a_features.to(self.device), y.to(self.device)
                X = torch.cat((k_features, a_features), dim=2)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                ttl_loss += loss.item()
            return ttl_loss / len(self.val_loader)

    def test(self, expr: str = None):
        self.model = self.load_model(expr=expr, best=True)
        self.model.to(self.device)
        self.model.eval()
        self.load_test_data()
        if hasattr(self, "logger"):
            self.logger.info("Best Model loaded, Start testing...")
        with torch.no_grad():
            ttl_loss = 0
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                ttl_loss += loss.item()
            avg_loss = ttl_loss / len(self.test_loader)
        metrics = {"loss": avg_loss}
        metrics_filename = os.path.join(
            self.path_config.metrics_dir, "test_metrics.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4, separators=(",", ": "))
        if hasattr(self, "logger"):
            self.logger.info("Test metrics saved")
            self.logger.info(f"Test loss: {avg_loss}")

    @abstractmethod
    def build_model(self):
        pass
