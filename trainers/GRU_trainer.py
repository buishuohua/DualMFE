from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from models.AFF import demoGRU

class GRUTrainer(BaseTrainer):
    def __init__(self, model_config, train_config, data_config, path_config):
        super().__init__(model_config, train_config, data_config, path_config)

    def build_model(self):
        if not hasattr(self, "model"):
            self.model = demoGRU(self.model_config.to_dict())

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
            self.optimizer.step()
            ttl_loss += loss.item()
        return ttl_loss / len(self.train_loader)

    def train(self):
        # TODO: Beautify the pbar
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
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= self.early_stop_patience:
                    self.early_stopped = True
                    self.logger.Warning(
                        "Early stopped at epoch: " + str(current_epoch))
                    break

            if self.epoch % self.save_freq == 0:
                self.save_model()
                self.logger.info(
                    "Model saved at epoch: " + str(current_epoch))

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
        with torch.no_grad():
            ttl_loss = 0
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                ttl_loss += loss.item()
            return ttl_loss / len(self.test_loader)
