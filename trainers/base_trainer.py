from abc import ABC, abstractmethod
import logging
import os
import sys
import json
import torch
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
from configs.pathConfig import PathConfig
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

    def _create_fresh_experiment(self):
        root_path = self.path_config.root_path
        self.path_config = PathConfig(root_path)
        new_expr_name = time.strftime("%Y%m%d-%H%M%S")
        self.path_config.create_expr(new_expr_name)
        self.init_logging()
        self.init_tensorboard()
        self._prepare_net()

        self.epoch = 0
        if self.use_early_stop:
            self.early_stopped = False
            self.early_stop_count = 0
        self.best_epoch = 0
        self.loss = 0
        self.best_loss = float("inf")

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

    def _check_early_stopped(self, expr_name: str):
        if not self.use_early_stop:
            return False
        logging_path = os.path.join(
            self.path_config.expr_dir, expr_name, "runs", "logging", "train.log")
        if not os.path.exists(logging_path):
            raise NoExperimentFound(f"{expr_name} not found")
        with open(logging_path, "r") as f:
            log_content = f.read()
            if "Early stopped at epoch" in log_content:
                return True

    def _check_model_compatibility(self, expr_name: str):
        model_config_path = os.path.join(
            self.path_config.expr_dir, expr_name, "runs", "configs", "model_config.json")
        if not os.path.exists(model_config_path):
            raise NoExperimentFound(f"{expr_name} not found")

        with open(model_config_path, "r") as f:
            loaded_config = json.load(f)

        selected_model_name = self.model_config.modelname
        loaded_model_name = loaded_config.get("modelname")

        if selected_model_name != loaded_model_name:
            return False
        else:
            return True

    def _prepare_net(self):
        self.build_model()
        if not hasattr(self, "optimizer"):
            self.optimizer = self.train_config.get_optimizer(self.model)
        if not hasattr(self, "scheduler"):
            self.scheduler = self.train_config.get_shceduler(self.optimizer)
        if not hasattr(self, "loss_fn"):
            self.loss_fn = self.train_config.get_loss_fn()

    def load_model(self, expr: str, epoch: int = None, best: bool = False):
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
        self._prepare_net()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_loss = ckpt["best_loss"]
        if self.use_early_stop:
            self.early_stopped = ckpt["early_stopped"]
            self.early_stop_count = ckpt["early_stop_count"]

    def prepare_learning(self):
        if self.continue_learning:
            expr_name = self.train_config.continue_expr
            if expr_name is None:
                try:
                    expr_name = self.path_config.find_latest_expr()
                except NoExperimentFound:
                    self._create_fresh_experiment()
                    self.logger.info(
                        "Continue learning is False, Start training from scratch")
                    self.logger.info(
                        "New experiment created: " + self.path_config.expr_name)
                    return
            else:
                if not os.path.exists(os.path.join(self.path_config.expr_dir, expr_name)):
                    raise NoExperimentFound(f"{expr_name} not found")
            self.path_config.create_expr(expr_name)

            model_compatible = self._check_model_compatibility(expr_name)
            if not model_compatible:
                response = input(
                    f"Model mismatch, start a new experiment with selected model:{self.model_config.modelname}? (y/n) ")
                if response.lower() == "y":
                    self._create_fresh_experiment()
                    self.logger.warning(
                        f"Selected Model:{self.model_config.modelname} mismatch with previous experiment {expr_name}, start a new experiment from scratch")
                    self.logger.info(
                        "New experiment created: " + self.path_config.expr_name)
                else:
                    print("Refuse to start a new experiment with selected model, exit.")
                    sys.exit(1)

            prev_early_stopped = self._check_early_stopped(expr_name)
            if prev_early_stopped:
                response = input(
                    f"Experiment {expr_name} has early stopped, start a new experiment? (y/n) ")
                if response.lower() == "y":
                    self._create_fresh_experiment()
                    self.logger.warning(
                        f"Experiment {expr_name} has early stopped, start a new experiment from scratch")
                    self.logger.info(
                        "New experiment created: " + self.path_config.expr_name)
                    return
                else:
                    print("Refuse to start a new experiment, exit.")
                    sys.exit(1)

            try:
                self.load_model(expr_name, best=False)
                self.init_logging()
                self.init_tensorboard()

                self.logger.info(
                    f"Model loaded, Continue learning from {expr_name}, Epoch: {self.epoch}")
            except NoCheckpointFound:
                self._create_fresh_experiment()
                self.logger.warning(
                    f"No checkpoint found for latest experiment {expr_name}, Start training from scratch")
                self.logger.info(
                    "New experiment created: " + self.path_config.expr_name)
        else:
            self._create_fresh_experiment()
            self.logger.info("Continue learning is False, Start training from scratch")
            self.logger.info(
                "New experiment created: " + self.path_config.expr_name)
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
            test_dataset, self.train_config.batch_size, shuffle=False, drop_last=False)
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
        if expr is None:
            try:
                expr = self.path_config.find_latest_expr()
            except NoExperimentFound:
                raise NoExperimentFound("Cannot find any experiment")
        else:
            if not os.path.exists(os.path.join(self.path_config.expr_dir, expr)):
                raise NoExperimentFound(f"{expr} not found")
        self.path_config.create_expr(expr)
        self.init_logging()

        model_compatible = self._check_model_compatibility(expr)
        if not model_compatible:
            print(
                f"Model mismatch, change the model selection in terminal to match with {expr}")
            sys.exit(1)

        self.load_model(expr=expr, best=True)

        self.model.to(self.device)
        self.model.eval()
        self.load_test_data()
        if hasattr(self, "logger"):
            self.logger.info("Best Model loaded, Start testing...")
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for k_features, a_features, y in self.test_loader:
                k_features, a_features, y = k_features.to(
                    self.device), a_features.to(self.device), y.to(self.device)
                X = torch.cat((k_features, a_features), dim=2)
                output = self.model(X)
                all_preds.append(output.cpu())
                all_targets.append(y.cpu())
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            mse = self.loss_fn(all_preds, all_targets).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            mae = nn.L1Loss()(all_preds, all_targets).item()
        metrics = {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }
        metrics_filename = os.path.join(
            self.path_config.metrics_dir, "test_metrics.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4, separators=(",", ": "))
        if hasattr(self, "logger"):
            self.logger.info("Test metrics saved")
            self.logger.info(
                f"TEST MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    @abstractmethod
    def build_model(self):
        pass
