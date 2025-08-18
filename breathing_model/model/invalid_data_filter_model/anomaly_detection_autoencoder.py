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
    Simplified autoencoder architecture for better anomaly detection.
    Fewer layers and parameters should help with better generalization.
    """

    def __init__(self, n_mels=40, latent_dim=8):
        super(SimplerBreathingAutoencoder, self).__init__()

        # Simplified encoder - fewer layers, less compression
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

        # Simplified decoder
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
        # Input normalization with scale preservation
        x_scale = torch.mean(torch.abs(x)) + 1e-8
        x_normalized = x / x_scale

        # Encoding and decoding
        latent = self.encoder(x_normalized)
        reconstructed = self.decoder(latent)

        # Restore original scale
        reconstructed = reconstructed * x_scale

        # Adjust output size to match input
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
    Enhanced loss function combining multiple metrics:
    - MSE for general reconstruction
    - Logarithmic MSE for better detection of low-energy patterns
    - Pearson correlation for preserving spectrogram patterns
    """

    def __init__(self, mse_weight=0.3, log_mse_weight=0.4, corr_weight=0.3):
        super(EnhancedReconstructionLoss, self).__init__()
        self.mse_weight = mse_weight
        self.log_mse_weight = log_mse_weight
        self.corr_weight = corr_weight
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, x, y):
        # Standard MSE
        mse_loss = self.mse(x, y)

        # Logarithmic MSE - better for low-energy patterns
        # Add small value for logarithm stability
        eps = 1e-8
        log_x = torch.log(torch.abs(x) + eps)
        log_y = torch.log(torch.abs(y) + eps)
        log_mse_loss = self.mse(log_x, log_y)

        # Pearson correlation for pattern preservation
        x_flat = x.reshape(x.size(0), -1)
        y_flat = y.reshape(y.size(0), -1)

        # Center the data
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)

        # Calculate correlation
        x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=1) + eps)
        y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=1) + eps)
        corr = torch.sum(x_centered * y_centered, dim=1) / (x_std * y_std)
        corr_loss = 1.0 - corr.mean()  # We want to maximize correlation

        # Total loss
        total_loss = (self.mse_weight * mse_loss +
                      self.log_mse_weight * log_mse_loss +
                      self.corr_weight * corr_loss)

        return total_loss


class AnomalyDetector:
    """
    Class for anomaly detection based on autoencoder.
    Considers multiple reconstruction metrics for better detection.
    """

    def __init__(self, autoencoder, device, contamination=0.05):
        self.autoencoder = autoencoder
        self.device = device
        self.contamination = contamination
        self.threshold = None
        self.criterion = EnhancedReconstructionLoss()

    def fit(self, dataloader):
        """Learn anomaly thresholds based on training data"""
        self.autoencoder.eval()
        reconstruction_errors = []
        latent_features = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs, latent = self.autoencoder(inputs)

                # Calculate reconstruction error for each sample
                batch_errors = []
                for i in range(inputs.size(0)):
                    error = self.criterion(outputs[i:i + 1], inputs[i:i + 1]).item()
                    batch_errors.append(error)

                reconstruction_errors.extend(batch_errors)

                # Save latent features for further analysis
                latent_features.append(latent.cpu().numpy())

        # Calculate anomaly threshold based on percentile
        self.threshold = np.percentile(reconstruction_errors, 100 * (1 - self.contamination))

        # Save error distribution for visualization
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
        """Detect if sample is an anomaly"""
        self.autoencoder.eval()

        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        mel_spec = mel_spec.to(self.device)

        with torch.no_grad():
            reconstruction, _ = self.autoencoder(mel_spec)
            error = self.criterion(reconstruction, mel_spec).item()

        is_anomaly = error > self.threshold

        return is_anomaly, error


def train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=30,
                      learning_rate=1e-3, scheduler_factor=0.5, scheduler_patience=3):
    """
    Enhanced training function with regularization and learning rate scheduling.
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
        # Training phase
        autoencoder.train()
        train_loss = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs, _ = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            # Backward and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
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

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), 'best_breathing_anomaly_detector.pth')
            print(f"Epoch {epoch + 1}/{num_epochs} - Saved best model")

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Every few epochs, generate sample reconstructions for visualization
        if (epoch + 1) % 5 == 0 and epoch > 0:
            visualize_reconstructions(autoencoder, val_loader, device, epoch + 1)

    # Visualize learning curves
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
    """Generate reconstruction visualizations for autoencoder quality assessment"""
    autoencoder.eval()

    # Get few samples from dataloader
    dataiter = iter(dataloader)
    images, _ = next(dataiter)

    with torch.no_grad():
        images = images.to(device)
        reconstructions, _ = autoencoder(images)

        # Select few images for visualization
        num_images = min(4, images.size(0))

        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            # Original images
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i, 0].cpu().numpy(), aspect='auto', cmap='viridis')
            plt.title(f"Original {i + 1}")
            plt.colorbar()

            # Reconstructions
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructions[i, 0].cpu().numpy(), aspect='auto', cmap='viridis')
            plt.title(f"Reconstructed {i + 1}")
            plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'reconstruction_epoch_{epoch}.png')
        plt.close()


def extract_latent_features(autoencoder, dataloader, device):
    """Extract features from latent layer for further analysis"""
    autoencoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            latent = autoencoder.encode(inputs)

            # Flatten latent features for each sample
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # Spatial average of latent features
                feat = latent[i].mean(dim=(1, 2)).cpu().numpy()
                features.append(feat)
                labels.append(label[i].item() if isinstance(label[i], torch.Tensor) else label[i])

    return np.array(features), labels


def apply_spectral_normalization(model):
    """
    Add spectral normalization to convolutional layers for better stability.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            setattr(model, name, nn.utils.spectral_norm(module))
    return model


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same dataset as in transformer model
    from model.transformer_model.transformer_model import BreathSeqDataset

    # Path to data folder
    data_dir = "../../deprecated/data-sequences"

    # Create dataset
    full_dataset = BreathSeqDataset(data_dir, sample_rate=44100, n_mels=40, n_fft=1024, hop_length=512)

    print(f"Found {len(full_dataset)} audio files in {data_dir}")

    # Split dataset into training and validation
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)  # Shuffle for random split
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create data loaders
    batch_size = 16  # Increase batch size for better stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize enhanced autoencoder
    autoencoder = SimplerBreathingAutoencoder(n_mels=40, latent_dim=8)

    # Optionally apply spectral normalization
    # autoencoder = apply_spectral_normalization(autoencoder)

    # Train autoencoder with new loss function
    train_losses, val_losses = train_autoencoder(
        autoencoder=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=30,
        learning_rate=1e-3
    )

    # Create and fit anomaly detector
    anomaly_detector = AnomalyDetector(autoencoder, device, contamination=0.05)
    threshold = anomaly_detector.fit(val_loader)
    print(f"Calculated anomaly threshold: {threshold:.6f}")

    # Save threshold to file for later use
    with open('anomaly_threshold.json', 'w') as f:
        json.dump({"threshold": float(threshold)}, f)

    # Optional: latent features analysis
    features, labels = extract_latent_features(autoencoder, val_loader, device)

    # Save model for production use
    torch.save({
        'autoencoder_state_dict': autoencoder.state_dict(),
        'anomaly_threshold': threshold,
    }, 'breathing_anomaly_detector_complete.pth')

    print("Training and anomaly detector configuration completed.")