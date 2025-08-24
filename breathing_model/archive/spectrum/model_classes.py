import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torchinfo import summary


class AudioClassifierLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x, hidden=None):
        outputs, hidden = self.lstm(x, hidden)
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
        mfcc_sequence = torch.tensor(
            np.array(mfcc_sequence), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Return lists of MFCC coefficients and labels
        return mfcc_sequence, labels


if __name__ == '__main__':
    print("LSTM summary")
    model = AudioClassifierLSTM()
    summary(model, input_size=(1, 1, 20))
