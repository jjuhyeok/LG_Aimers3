import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from src.config.config import CFG
from src.train.utils import set_logger, save_model_state_dict


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, logger=True):
        self.model = model.to(CFG.DEVICE)
        self.criterion = criterion.to(CFG.DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 9999999
        self.best_model = None
        if logger:
            set_logger('train', self.model, self.criterion, self.optimizer, self.scheduler)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0

        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(CFG.DEVICE), label.to(CFG.DEVICE)

            self.optimizer.zero_grad()

            prediction = self.model(data)
            loss = self.criterion(prediction, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() / len(train_loader)

        return train_loss

    def fit(self, train_loader):
        for epoch in range(CFG.EPOCHS):
            avg_train_loss = self.train(train_loader)

            log_msg = (
                f"Epoch [{epoch + 1}/{CFG.EPOCHS}] "
                f"Training Loss: {avg_train_loss:.4f} "
            )

            logging.info(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.best_loss > avg_train_loss:
                time = datetime.now().strftime('%Y.%m.%d')
                save_model_name = f'{time}{self.model.__class__.__name__}.pth'
                self.best_loss = avg_train_loss
                best_model = self.model
                save_model_state_dict(best_model, save_model_name)

        return best_model
