import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class BreathingAutoencoder(nn.Module):
    def __init__(self, n_mels=40, latent_dim=16):
        super(BreathingAutoencoder, self).__init__()

        # Encoder (compress time dimension while preserving mel dimension initially)
        self.encoder = nn.Sequential(
            # Layer 1: (1, n_mels, time) -> (16, n_mels, time/2)
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2: (16, n_mels, time/2) -> (32, n_mels/2, time/4)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3: (32, n_mels/2, time/4) -> (64, n_mels/4, time/8)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 4: (64, n_mels/4, time/8) -> (latent_dim, n_mels/8, time/16)
            nn.Conv2d(64, latent_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: (latent_dim, n_mels/8, time/16) -> (64, n_mels/4, time/8)
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 2: (64, n_mels/4, time/8) -> (32, n_mels/2, time/4)
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3: (32, n_mels/2, time/4) -> (16, n_mels, time/2)
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 4: (16, n_mels, time/2) -> (1, n_mels, time)
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 1)),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # Ensure output has same size as input in time dimension
        if reconstructed.size() != x.size():
            # Resize output to match input size
            reconstructed = F.interpolate(reconstructed, size=(x.size(2), x.size(3)), mode='bilinear',
                                          align_corners=False)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=50, learning_rate=1e-3, patience=10):
    """
    Train the autoencoder with early stopping based on validation loss.

    Args:
        autoencoder: The autoencoder model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate
        patience: Number of epochs to wait for improvement before stopping
    """
    autoencoder.to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        # Training
        autoencoder.train()
        train_loss = 0
        for inputs, _ in train_loader:  # We only need the spectrograms, not the labels
            inputs = inputs.to(device)

            # Forward pass
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(autoencoder.state_dict(), 'best_breathing_autoencoder.pth')
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                early_stop = True
                break

    if not early_stop:
        print("Completed all epochs without early stopping")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig('autoencoder_loss.png')
    plt.close()

    return train_losses, val_losses
def calculate_reconstruction_error(autoencoder, dataloader, device):
    """
    Calculate reconstruction error for each sample in the dataset.

    Args:
        autoencoder: Trained autoencoder model
        dataloader: DataLoader containing the dataset
        device: Device to run inference on

    Returns:
        numpy.ndarray: Array of reconstruction errors for each sample
    """
    autoencoder.eval()
    criterion = nn.MSELoss(reduction='none')
    errors = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)

            # Calculate error (MSE) for each sample
            error = criterion(outputs, inputs)
            # Average over all dimensions except batch
            error = error.mean(dim=(1, 2, 3)).cpu().numpy()
            errors.append(error)

    return np.concatenate(errors)


def find_threshold(reconstruction_errors, contamination=0.05):
    """
    Find threshold for anomaly detection based on the contamination rate.

    Args:
        reconstruction_errors: Array of reconstruction errors
        contamination: Expected proportion of outliers in the data

    Returns:
        float: Threshold value
    """
    threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
    return threshold


def is_anomaly(autoencoder, mel_spec, threshold, device):
    """
    Determine if a new sample is an anomaly.

    Args:
        autoencoder: Trained autoencoder model
        mel_spec: Input mel spectrogram (torch.Tensor of shape [1, n_mels, time])
        threshold: Reconstruction error threshold
        device: Device to run inference on

    Returns:
        bool: True if the sample is an anomaly, False otherwise
        float: Reconstruction error
    """
    autoencoder.eval()
    criterion = nn.MSELoss(reduction='none')

    # Ensure proper shape and device
    if mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
    mel_spec = mel_spec.to(device)

    with torch.no_grad():
        reconstruction = autoencoder(mel_spec)
        error = criterion(reconstruction, mel_spec)
        error = error.mean().item()  # Average over all dimensions

    return error > threshold, error


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same dataset as in the transformer model
    from breathing_model.model.transformer_model.transformer_model import BreathSeqDataset

    # Path to the data folder
    data_dir = "../../data-sequences"

    # Create dataset
    full_dataset = BreathSeqDataset(data_dir, sample_rate=44100, n_mels=40, n_fft=1024, hop_length=512)

    # Split dataset into train and validation sets
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    from torch.utils.data import Subset

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize autoencoder
    autoencoder = BreathingAutoencoder(n_mels=40, latent_dim=16)

    # Train autoencoder
    train_losses, val_losses = train_autoencoder(
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50,
        learning_rate=1e-3,
        patience=5
    )

    # Load the best model
    autoencoder.load_state_dict(torch.load('best_breathing_autoencoder.pth'))

    # Calculate reconstruction errors on the validation set
    recon_errors = calculate_reconstruction_error(autoencoder, val_loader, device)

    # Find threshold for anomaly detection (assuming 5% of data are outliers)
    threshold = find_threshold(recon_errors, contamination=0.05)
    print(f"Threshold for anomaly detection: {threshold:.6f}")

    # Plot histogram of reconstruction errors
    plt.figure(figsize=(10, 5))
    plt.hist(recon_errors, bins=50, alpha=0.75)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig('reconstruction_error_histogram.png')
    plt.close()

    # Save threshold for later use
    import json

    with open('anomaly_threshold.json', 'w') as f:
        json.dump({"threshold": float(threshold)}, f)

    # Example usage for anomaly detection
    print("\nExample anomaly detection:")
    for i in range(5):  # Check first 5 samples from validation set
        mel_spec, _ = full_dataset[val_indices[i]]
        is_anomalous, error = is_anomaly(autoencoder, mel_spec, threshold, device)
        print(f"Sample {i}: {'Anomaly' if is_anomalous else 'Normal'} (Error: {error:.6f})")