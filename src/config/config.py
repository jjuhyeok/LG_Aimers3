import os
import random

import numpy as np
import torch


class Config:
    TRAIN_WINDOW_SIZE = 90
    PREDICT_SIZE = 21

    EPOCHS = 30
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 4096
    SEED = 1103

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    SAVE_MODEL_PATH = r"C:\MB_Project\project\product-sales\models"
    LOG_DIR = 'logs'


CFG = Config()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
