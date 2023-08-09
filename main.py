from src.config.config import CFG
from src.data.utils import load_pickle
from src.data.load_dataset import CustomDataLoader
from src.model.LSTM import BaseModel

import torch
import torch.nn as nn
from src.train.train import Trainer


def main():
    train_input = load_pickle(r"input_feature32.pickle")
    target = load_pickle(r"target32.pickle")

    DL = CustomDataLoader(train_input, target, False)
    train_loader = DL.fit

    model = BaseModel(num_layer=2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG.LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    trainer = Trainer(model, criterion, optimizer, scheduler, logger=True)
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()
