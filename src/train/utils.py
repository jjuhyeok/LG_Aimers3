import logging
import os
import sys
from datetime import datetime

import torch

from src.config.config import CFG


def set_logger(mode='train', *args):
    if not os.path.exists(CFG.LOG_DIR):
        os.makedirs(CFG.LOG_DIR)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f'{mode}_{current_time}.log'
    log_file_path = os.path.join(CFG.LOG_DIR, log_file_name)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()])

    if mode == "train":
        # Environment Info
        logger = logging.getLogger(__name__)
        logger.info('Environment Info:')
        logger.info('------------------------------------------------------------')
        logger.info('Author: Myungbin')
        logger.info(f'sys.platform: {sys.platform}')
        logger.info(f'Python: {sys.version}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logger.info(f'GPU 0: {torch.cuda.get_device_name(0)}')
            logger.info(f'CUDA_HOME: {torch.utils.cmake_prefix_path}')
            logger.info(f'NVCC: {torch.version.cuda}')
        logger.info(f'PyTorch: {torch.__version__}')
        logger.info(f'PyTorch compiling details: {torch.__config__.show()}')

        # Parameters Info
        logger.info('Parameters Info:')
        logger.info('------------------------------------------------------------')
        logger.info(f'Batch Size: {CFG.BATCH_SIZE}')
        logger.info(f'Epoch: {CFG.EPOCHS}')
        logger.info(f'Learning Rate: {CFG.LEARNING_RATE}')
        for arg in args:
            logger.info(f"{arg.__class__.__name__}: {arg}")


def save_model_state_dict(model, filename):
    if not os.path.exists(CFG.SAVE_MODEL_PATH):
        os.makedirs(CFG.SAVE_MODEL_PATH)
    torch.save(model.state_dict(), os.path.join(CFG.SAVE_MODEL_PATH, filename))
