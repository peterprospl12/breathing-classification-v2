import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import json


class SimplerBreathingAutoencoder(nn.Module):
    """
    Uproszczona architektura autoenkodera dla lepszego wykrywania anomalii.
    Mniejsza liczba warstw i parametrów powinna pomóc w lepszym uogólnianiu.
    """

    def __init__(self, n_mels=40, latent_dim=8):
        super(SimplerBreathingAutoencoder, self).__init__()

        # Uproszczony enkoder - mniej warstw, mniejsza kompresja
        self.encoder = nn.Sequential(
            # Layer 1: (1, n_mels, time) -> (16, n_mels, time/2)
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 2: (16, n_mels, time/2) -> (32, n_mels/2, time/4)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 3: (32, n_mels/2, time/4) -> (latent_dim, n_mels/4, time/8)
            nn.Conv2d(32, latent_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
        )

        # Uproszczony dekoder
        self.decoder = nn.Sequential(
            # Layer 1: (latent_dim, n_mels/4, time/8) -> (32, n_mels/2, time/4)
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 2: (32, n_mels/2, time/4) -> (16, n_mels, time/2)
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.LeakyReLU(0.2),

            # Layer 3: (16, n_mels, time/2) -> (1, n_mels, time)
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1),
                               output_padding=(0, 1)),
        )

    def forward(self, x):
        # Normalizacja wejścia z zachowaniem skali
        x_scale = torch.mean(torch.abs(x)) + 1e-8
        x_normalized = x / x_scale

        # Enkodowanie i dekodowanie
        latent = self.encoder(x_normalized)
        reconstructed = self.decoder(latent)

        # Przywrócenie oryginalnej skali
        reconstructed = reconstructed * x_scale

        # Dopasowanie rozmiaru wyjścia do wejścia
        if reconstructed.size() != x.size():
            reconstructed = F.interpolate(reconstructed, size=(x.size(2), x.size(3)),
                                          mode='bilinear', align_corners=False)
        return reconstructed, latent

    def encode(self, x):
        x_scale = torch.mean(torch.abs(x)) + 1e-8
        x_normalized = x / x_scale
        return self.encoder(x_normalized)


class EnhancedReconstructionLoss(nn.Module):
    """
    Ulepszona funkcja straty łącząca kilka metryk:
    - MSE dla ogólnej rekonstrukcji
    - Logarytmiczna MSE dla lepszego wychwycenia wzorców o niskiej energii
    - Korelacja Pearsona dla zachowania wzorca spektrogramu
    """

    def __init__(self, mse_weight=0.3, log_mse_weight=0.4, corr_weight=0.3):
        super(EnhancedReconstructionLoss, self).__init__()
        self.mse_weight = mse_weight
        self.log_mse_weight = log_mse_weight
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        # Standardowe MSE
        mse_loss = self.mse(x, y)

        # Logarytmiczna MSE - lepsza dla wzorców o niskiej energii
        # Dodajemy małą wartość dla stabilności logarytmu
        eps = 1e-8
        log_x = torch.log(torch.abs(x) + eps)
        log_y = torch.log(torch.abs(y) + eps)
        log_mse_loss = self.mse(log_x, log_y)

        # Korelacja Pearsona dla zachowania wzorca
        x_flat = x.reshape(x.size(0), -1)
        y_flat = y.reshape(y.size(0), -1)

        # Centrowanie danych
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)

        # Obliczenie korelacji
        x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=1) + eps)
        y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=1) + eps)
        corr = torch.sum(x_centered * y_centered, dim=1) / (x_std * y_std)
        corr_loss = 1.0 - corr.mean()  # Chcemy maksymalizować korelację

        # Łączna strata
        total_loss = (self.mse_weight * mse_loss +
                      self.log_mse_weight * log_mse_loss +
                      self.corr_weight * corr_loss)

        return total_loss


class AnomalyDetector:
    """
    Klasa do wykrywania anomalii na podstawie autoenkodera.
    Uwzględnia kilka metryk rekonstrukcji dla lepszej detekcji.
    """

    def __init__(self, autoencoder, device, contamination=0.05):
        self.autoencoder = autoencoder
        self.device = device
        self.contamination = contamination
        self.threshold = None
        self.criterion = EnhancedReconstructionLoss()

    def fit(self, dataloader):
        """Wyucz progi anomalii na podstawie danych treningowych"""
        self.autoencoder.eval()
        reconstruction_errors = []
        latent_features = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs, latent = self.autoencoder(inputs)

                # Oblicz błąd rekonstrukcji dla każdej próbki
                batch_errors = []
                for i in range(inputs.size(0)):
                    error = self.criterion(outputs[i:i + 1], inputs[i:i + 1]).item()
                    batch_errors.append(error)

                reconstruction_errors.extend(batch_errors)

                # Zapisz cechy z warstwy ukrytej do dalszej analizy
                latent_features.append(latent.cpu().numpy())

        # Oblicz próg anomalii na podstawie percentyla
        self.threshold = np.percentile(reconstruction_errors, 100 * (1 - self.contamination))

        # Zapisz rozkład błędów do wizualizacji
        plt.figure(figsize=(10, 5))
        plt.hist(reconstruction_errors, bins=50)
        plt.axvline(self.threshold, color='r', linestyle='dashed',
                    label=f'Threshold: {self.threshold:.4f}')
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig('anomaly_threshold.png')
        plt.close()

        return self.threshold

    def detect(self, mel_spec):
        """Wykryj czy próbka jest anomalią"""
        self.autoencoder.eval()

        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)  # Dodaj wymiar batcha
        mel_spec = mel_spec.to(self.device)

        with torch.no_grad():
            reconstruction, _ = self.autoencoder(mel_spec)
            error = self.criterion(reconstruction, mel_spec).item()

        is_anomaly = error > self.threshold

        return is_anomaly, error


def train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=30,
                      learning_rate=1e-3, scheduler_factor=0.5, scheduler_patience=3):
    """
    Ulepszona funkcja treningowa z regularyzacją i planowaniem współczynnika uczenia.
    """
    autoencoder.to(device)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = EnhancedReconstructionLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Faza treningowa
        autoencoder.train()
        train_loss = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs, _ = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            # Backward i optymalizacja
            optimizer.zero_grad()
            loss.backward()

            # Przycinanie gradientów dla stabilności
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Faza walidacyjna
        autoencoder.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs, _ = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Aktualizacja schedulera
        scheduler.step(val_loss)

        # Zapisz najlepszy model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), 'best_breathing_anomaly_detector.pth')
            print(f"Epoch {epoch + 1}/{num_epochs} - Saved best model")

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Co kilka epok, generuj przykładowe rekonstrukcje dla wizualizacji
        if (epoch + 1) % 5 == 0 and epoch > 0:
            visualize_reconstructions(autoencoder, val_loader, device, epoch + 1)

    # Wizualizacja krzywych uczenia
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig('enhanced_autoencoder_loss.png')
    plt.close()

    return train_losses, val_losses


def visualize_reconstructions(autoencoder, dataloader, device, epoch):
    """Generuj wizualizacje rekonstrukcji dla oceny jakości autoenkodera"""
    autoencoder.eval()

    # Pobierz kilka próbek z dataloaderów
    dataiter = iter(dataloader)
    images, _ = next(dataiter)

    with torch.no_grad():
        images = images.to(device)
        reconstructions, _ = autoencoder(images)

        # Wybierz kilka obrazów do wizualizacji
        num_images = min(4, images.size(0))

        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            # Oryginalne obrazy
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i, 0].cpu().numpy(), aspect='auto', cmap='viridis')
            plt.title(f"Original {i + 1}")
            plt.colorbar()

            # Rekonstrukcje
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructions[i, 0].cpu().numpy(), aspect='auto', cmap='viridis')
            plt.title(f"Reconstructed {i + 1}")
            plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'reconstruction_epoch_{epoch}.png')
        plt.close()


def extract_latent_features(autoencoder, dataloader, device):
    """Wyodrębnij cechy z warstwy ukrytej dla dalszej analizy"""
    autoencoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            latent = autoencoder.encode(inputs)

            # Spłaszcz cechy latentne dla każdej próbki
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # Średnia przestrzenna cech latentnych
                feat = latent[i].mean(dim=(1, 2)).cpu().numpy()
                features.append(feat)
                labels.append(label[i].item() if isinstance(label[i], torch.Tensor) else label[i])

    return np.array(features), labels


def apply_spectral_normalization(model):
    """
    Dodaj normalizację spektralną do warstw konwolucyjnych dla lepszej stabilności.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            setattr(model, name, nn.utils.spectral_norm(module))
    return model


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Użyj tego samego datasetu co w modelu transformer
    from model.transformer_model.transformer_model import BreathSeqDataset

    # Ścieżka do folderu z danymi
    data_dir = "../../deprecated/data-sequences"

    # Utwórz dataset
    full_dataset = BreathSeqDataset(data_dir, sample_rate=44100, n_mels=40, n_fft=1024, hop_length=512)

    print(f"Znaleziono {len(full_dataset)} plików audio w {data_dir}")

    # Podział zbioru danych na treningowy i walidacyjny
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)  # Tasowanie dla losowego podziału
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Utwórz data loadery
    batch_size = 16  # Zwiększenie rozmiaru batcha dla lepszej stabilności
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Inicjalizacja ulepszonego autoenkodera
    autoencoder = SimplerBreathingAutoencoder(n_mels=40, latent_dim=8)

    # Opcjonalnie zastosuj normalizację spektralną
    # autoencoder = apply_spectral_normalization(autoencoder)

    # Trenuj autoenkoder z nową funkcją straty
    train_losses, val_losses = train_autoencoder(
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=30,
        learning_rate=1e-3
    )

    # Utwórz i dopasuj detektor anomalii
    anomaly_detector = AnomalyDetector(autoencoder, device, contamination=0.05)
    threshold = anomaly_detector.fit(val_loader)
    print(f"Obliczony próg anomalii: {threshold:.6f}")

    # Zapisz próg do pliku dla późniejszego użycia
    with open('anomaly_threshold.json', 'w') as f:
        json.dump({"threshold": float(threshold)}, f)

    # Opcjonalnie: analiza cech latentnych
    features, labels = extract_latent_features(autoencoder, val_loader, device)

    # Zapisz model do użycia w produkcji
    torch.save({
        'autoencoder_state_dict': autoencoder.state_dict(),
        'anomaly_threshold': threshold,
        'silence_threshold': silence_threshold
    }, 'breathing_anomaly_detector_complete.pth')

    print("Trening i konfiguracja detektora anomalii zakończone.")