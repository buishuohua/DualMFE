from abc import ABC, abstractmethod
import logging
import json
import os
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

    @abstractmethod
    def build_model(self):
        pass
    # TODO: Check if there is a more brilliant way to organize the parse and init
    def _parse_params(self):
        self.num_epochs = self.train_config.num_epochs
        self.save_freq = self.train_config.save_freq
        self.continue_learning = self.train_config.continue_learning
        self.early_stop_patience = self.train_config.early_stop_patience
        self.device = self.train_config.device

    def _init(self):
        # TODO: could specify the optimizer and scheduler params
        self.build_model()
        if not hasattr(self, "model"):
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

    def init_loggings(self):
        # TODO: May can be simplified in the logging_path
        self.logger = logging.basicConfig(
            filename=self.path_config.logging_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def init_tensorboard(self):
        self.writer = SummaryWriter(self.path_config.tb_dir)

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
        torch.save(ckpt, self.path_config.models_dir, ckpt_name)

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
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.early_stopped = ckpt["early_stopped"]
        self.early_stop_count = ckpt["early_stop_count"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_loss = ckpt["best_loss"]

    def prepare_learning(self):
        if self.continue_learning:
            if self.train_config.continue_expr is None:
                try:
                   latest_expr = self.path_config.find_latest_expr()
                except NoExperimentFound:
                    latest_expr = None
                if latest_expr is None:
                    self._init()
                    self.init_loggings()
                    self.init_tensorboard()
                    self.logger.info("No experiment found, Start training from scratch")
                    self.logger.info(f"New experiment created: {self.path_config.expr_name}")
                else:
                    self.load_model(expr=latest_expr, best=False)
            else:
                self.load_model(
                    expr=self.train_config.continue_expr, best=False)
            if not hasattr(self, "logger"):
                self.init_loggings()
            if not hasattr(self, "writer"):
                self.init_tensorboard()
            self.logger.info(f"Model loaded, Continue learning from {latest_expr}")
        else:
            self._init()
            self.init_loggings()
            self.init_tensorboard()
            self.logger.info("Continue learning is False, Start training from scratch")

    def load_train_data(self):
        train_data_path = self.path_config.train_file_path
        val_ratio = self.data_config.val_size
        train, val = train_val_split(train_data_path, val_ratio)
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
        test_data_path = self.path_config.test_file_path
        test = pd.read_parquet(test_data_path)
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
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ttl_loss += loss.item()
        return ttl_loss / len(self.train_loader)

    def train(self):
        self.prepare_learning()
        self.load_train_data()
        self.logger.info(
            "Training dataset and validation dataset loaded successfully, start training...")
        pbar = tqdm(range(self.epoch, self.num_epochs + 1))
        for epoch in pbar:
            train_loss = self.train_epoch()
            val_loss = self.eval()
            self.scheduler.step(val_loss)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.save_model(best=True)
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= self.early_stop_patience:
                    self.early_stopped = True
                    break

            if self.epoch % self.save_freq == 0:
                self.save_model()

            pbar.set_description(f"Epoch {epoch}")
            pbar.update(1)
            # TODO: maybe mismatch
            self.epoch += 1

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            ttl_loss = 0
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                ttl_loss += loss.item()
            return ttl_loss / len(self.val_loader)

    def test(self, expr: str = None):
        self.model = self.load_model(expr=expr, best=True)
        self.model.to(self.device)
        self.model.eval()
        self.load_test_data()
        with torch.no_grad():
            ttl_loss = 0
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                ttl_loss += loss.item()
            return ttl_loss / len(self.test_loader)
