# python
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

from breathing_model.model.transformer.utils import load_yaml


# Klasa Dataset dla audio (wyodrębnianie MFCC)
class AudioDataset(Dataset):
    def __init__(self, file_paths, n_mfcc=13, max_len=100):
        self.file_paths = file_paths
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_paths[idx], sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = mfcc.T
        if len(mfcc) > self.max_len:
            mfcc = mfcc[:self.max_len]
        elif len(mfcc) < self.max_len:
            pad_width = self.max_len - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        mu = mfcc.mean()
        sigma = mfcc.std() + 1e-8
        mfcc = (mfcc - mu) / sigma
        mfcc = mfcc.flatten()
        return torch.tensor(mfcc, dtype=torch.float32)


# Klasa Autoenkodera
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Trening z early stopping i zapisem najlepszego modelu
def train_autoencoder(model,
                      train_loader,
                      val_loader,
                      epochs=50,
                      lr=0.001,
                      patience=7,
                      min_delta=1e-4,
                      ckpt_path=None,
                      device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / max(1, len(train_loader))

        # Val
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss_sum += loss.item()
        val_loss = val_loss_sum / max(1, len(val_loader))

        print(f'Epoch {epoch}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}')

        # Early stopping + checkpoint
        if val_loss + min_delta < best_val:
            best_val = val_loss
            patience_ctr = 0
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(ckpt_path))
                print(f'Zapisano najlepszy model do: {ckpt_path}')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print('Early stopping.')
                break

    # Wczytaj najlepszy checkpoint po treningu
    if ckpt_path is not None and ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        print(f'Wczytano najlepszy model z: {ckpt_path}')

    return model


# Detekcja anomalii
def detect_anomaly(model, input_data, threshold):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        input_data = input_data.to(device)
        reconstructed = model(input_data)
        loss = nn.MSELoss(reduction='mean')(reconstructed, input_data)
        return 1 if loss.item() > threshold else 0


# Przykład użycia
if __name__ == "__main__":
    # Wczytaj config
    config_path = (Path(__file__).resolve().parent.parent / 'transformer' / 'config.yaml')
    config = load_yaml(str(config_path))

    # Dane
    cfg_root = config_path.parent
    data_dir = (cfg_root / config['data']['data_dir']).resolve()
    normal_files = [str(p) for p in data_dir.rglob('*.wav')]
    train_files, val_files = train_test_split(normal_files, test_size=0.2, random_state=42)

    # Parametry
    n_mfcc = 13
    max_len = 100
    input_dim = n_mfcc * max_len
    batch_size = 32

    # Loadery
    train_dataset = AudioDataset(train_files, n_mfcc, max_len)
    val_dataset = AudioDataset(val_files, n_mfcc, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Ścieżka zapisu modelu
    ckpt_dir = Path(__file__).resolve().parent / 'checkpoints'
    ckpt_path = ckpt_dir / 'autoencoder_best.pt'

    # Trening z early stopping + zapis modelu
    model = Autoencoder(input_dim)
    model = train_autoencoder(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        patience=7,
        min_delta=1e-4,
        ckpt_path=ckpt_path
    )

    # Oblicz próg na walidacji (na najlepszych wagach)
    device = next(model.parameters()).device
    errors = []
    model.eval()
    with torch.no_grad():
        for data in DataLoader(val_dataset, batch_size=1, shuffle=False):
            data = data.to(device)
            reconstructed = model(data)
            error = nn.MSELoss()(reconstructed, data).item()
            errors.append(error)
    threshold = np.mean(errors) + 3 * np.std(errors)
    print(f'Ustalony próg: {threshold:.6f}')

    # Test na nowym pliku
    test_file = 'test_audio.wav'
    test_data = AudioDataset([test_file], n_mfcc, max_len)[0].unsqueeze(0)
    result = detect_anomaly(model, test_data, threshold)
    print(f'Wynik: {result} (1 - anomalia, 0 - normalne)')