import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Model that uses LSTM network
class AudioClassifierLSTM(nn.Module):
    def __init__(self):

        # Call the parent class constructor
        super(AudioClassifierLSTM, self).__init__()

        # input_size - number of features in the input (20 MFCC coefficients)
        self.lstm = nn.LSTM(input_size=20, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)

        # Couple of fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    # x - input tensor of shape [batch_size, max_length, num_features]
    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden


class AudioClassifierGRU(nn.Module):
    def __init__(self):
        super(AudioClassifierGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=20,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x, hidden=None):
        x, hidden = self.gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden


class AudioClassifierRNN(nn.Module):
    def __init__(self):
        super(AudioClassifierRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=20,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden


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
        mfcc_sequence = torch.tensor(mfcc_sequence, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Return lists of MFCC coefficients and labels
        return mfcc_sequence, labels
