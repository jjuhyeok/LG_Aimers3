import torch
from torch.utils.data import Dataset, DataLoader
from src.config.config import CFG


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)


class CustomDataLoader:
    def __init__(self, x_train, y_train, validation=False):
        self.x_train = x_train
        self.y_train = y_train
        self.validation = validation

    @staticmethod
    def train_test_split(x_train, y_train):
        data_len = len(x_train)
        train_input = x_train[:-int(data_len * 0.2)]
        train_target = y_train[:-int(data_len * 0.2)]
        val_input = x_train[-int(data_len * 0.2):]
        val_target = y_train[-int(data_len * 0.2):]
        return train_input, train_target, val_input, val_target

    def init_dataset(self):
        if self.validation:
            train_input, train_target, val_input, val_target = self.train_test_split(self.x_train, self.y_train)
            train_dataset = CustomDataset(train_input, train_target)
            val_dataset = CustomDataset(val_input, val_target)
            return train_dataset, val_dataset
        else:
            train_dataset = CustomDataset(self.x_train, self.y_train)
            return train_dataset

    @property
    def fit(self):
        if self.validation:
            train_dataset, val_dataset = self.init_dataset()
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
            return train_loader, val_loader
        else:
            train_dataset = self.init_dataset()
            train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
            return train_loader


if __name__ == "__main__":
    from src.data.utils import load_pickle

    train_input = load_pickle(r"input_feature32.pickle")
    target = load_pickle(r"target32.pickle")
    test_input = load_pickle(r"test_input32.pickle")

    DL = CustomDataLoader(train_input, target, False)
    train_loader = DL.fit

    for i in train_loader:
        print(i.shape)
