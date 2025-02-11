import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchinfo import summary

# Model that uses LSTM network
class AudioClassifierLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=60,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.4
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, hidden=None):
        # x: [batch, seq_len, 20] (train) or [1, 1, 20] (realtime)
        outputs, hidden = self.lstm(x, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

class AudioClassifierGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=60,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.gru(x, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

class AudioClassifierRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=60,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            nonlinearity='tanh',
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.rnn(x, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden


class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Get the sequence consisting of MFCCs and labels
        sequence = self.data[idx]

        # Extract MFCCs and labels from the sequence
        mfcc_sequence = [item[0] for item in sequence]
        labels = [item[1] for item in sequence]

        # Convert the lists to PyTorch tensors
        mfcc_sequence = torch.tensor(np.array(mfcc_sequence), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Return lists of MFCC coefficients and labels
        return mfcc_sequence, labels

if __name__ == '__main__':
    print("LSTM summary")
    model = AudioClassifierLSTM()
    summary(model, input_size=(1, 1, 20))

    print("RNN summary")
    model = AudioClassifierRNN()
    summary(model, input_size=(1, 1, 20))

    print("GRU summary")
    model = AudioClassifierGRU()
    summary(model, input_size=(1, 1, 20))
