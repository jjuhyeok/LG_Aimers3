import torch.nn as nn
import torch
from src.config.config import CFG


class BaseModel(nn.Module):
    def __init__(self, input_size=14, hidden_size=512, output_size=CFG.PREDICT_SIZE, num_layer=1):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layer)
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layer)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.num_layer = num_layer

        self.actv = nn.ReLU()

    def forward(self, x):
        # x shape: (B, TRAIN_WINDOW_SIZE, 5)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.device)

        # LSTM layer
        lstm_out, hidden = self.lstm(x, hidden)

        # Only use the last output sequence
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.actv(self.fc(last_output))

        return output.squeeze(1)

    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state
        return (torch.zeros(self.num_layer, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layer, batch_size, self.hidden_size, device=device))
