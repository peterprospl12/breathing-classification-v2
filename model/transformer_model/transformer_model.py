import os
import glob
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

#########################################
# 1. Dataset definition for sequential recordings
#########################################

class BreathSeqDataset(Dataset):
    """
    Dataset loading audio recordings and their corresponding labels stored in CSV files.
    In the `data_dir` folder, we expect pairs of files: .wav and .csv with the same name (e.g., recording1.wav and recording1.csv).
    The CSV file should contain columns: phase_code, start_sample, end_sample.
    Each phase is marked with its code:
      0: exhale
      1: inhale
      2: silence
    """
    def __init__(self, data_dir, sample_rate=44100, n_mels=40, n_fft=1024, hop_length=512, transform=None):
        """
        Args:
            data_dir (str): Path to the data folder (e.g., "../../scripts/data-seq")
            sample_rate (int): Target sampling rate (set to 44100 Hz here)
            n_mels (int): Number of mel filters
            n_fft (int): FFT length
            hop_length (int): Hop length for spectrogram calculation
            transform (callable, optional): Additional transformation for the spectrogram
        """
        self.data_dir = data_dir
        # Search for all WAV files in the folder
        self.audio_files = glob.glob(os.path.join(data_dir, "*.wav"))
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Get the path to the audio file
        audio_path = self.audio_files[idx]
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        csv_path = os.path.join(self.data_dir, base_name + ".csv")

        # Load the audio (waveform has shape (channels, num_samples))
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if the recording has more than one channel
        if waveform.shape[0] > 1:
            print("Warning: stereo audio converted to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to the target sample_rate (44100 Hz)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Compute the mel spectrogram and apply log transformation (prevents numerical issues)
        mel_spec = self.mel_transform(waveform)  # kształt: (channels, n_mels, time_steps)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Load labels from the CSV file
        df = pd.read_csv(csv_path)
        time_steps = mel_spec.shape[-1]
        # Initialize the label sequence – for each spectrogram step (assuming labels cover the entire signal)
        label_seq = np.zeros(time_steps, dtype=np.int64)
        # For each row in the CSV, determine which spectrogram frames (based on hop_length) will receive the given label
        for _, row in df.iterrows():
            phase_code = str(row['class'])
            if phase_code == 'exhale':
                phase_code = 0
            elif phase_code == 'inhale':
                phase_code = 1
            elif phase_code == 'silence':
                phase_code = 2
            start_sample = int(row['start_sample'])
            end_sample = int(row['end_sample'])
            # Calculate frame numbers – assuming a frame corresponds to a sample: i * hop_length
            start_frame = int(np.floor(start_sample / self.hop_length))
            end_frame = int(np.ceil(end_sample / self.hop_length))
            end_frame = min(end_frame, time_steps)  # zabezpieczenie, gdyby wykraczało poza liczbę klatek
            label_seq[start_frame:end_frame] = phase_code

        label_seq = torch.tensor(label_seq, dtype=torch.long)  # kształt: (time_steps,)

        if self.transform:
            mel_spec = self.transform(mel_spec)
        # Ensure mel_spec has shape (1, n_mels, time_steps)
        return mel_spec, label_seq

#########################################
# 2. Positional encoding for Transformer
#########################################

class PositionalEncoding(nn.Module):
    """
    Implementation of positional encoding according to "Attention is All You Need".
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with added positional encoding.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#########################################
# 3. Model: CNN + Transformer with sequential prediction
#########################################

class BreathPhaseTransformerSeq(nn.Module):
    def __init__(self, n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2):
        """
        Args:
            n_mels (int): Number of mel coefficients
            num_classes (int): Number of classes (0: exhale, 1: inhale, 2: silence)
            d_model (int): Dimension of the vector space in the Transformer
            nhead (int): Number of heads in the attention mechanism
            num_transformer_layers (int): Number of Transformer layers
        """
        super(BreathPhaseTransformerSeq, self).__init__()
        # CNN part – pooling is performed only in the frequency axis to preserve temporal resolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool_freq1 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool_freq2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool_freq3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        # After 3 times pooling in the frequency axis: out_freq = n_mels // 2 // 2 // 2
        self.out_freq = n_mels // 8  
        cnn_feature_dim = 128 * self.out_freq  # cecha dla jednej klatki
        
        # Projection to d_model space
        self.fc_proj = nn.Linear(cnn_feature_dim, d_model)
        
        # Transformer – using batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
        
        self.dropout = nn.Dropout(0.3)
        # Output head – prediction of the label for each frame (sequentially)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 1, n_mels, time_steps)
        Returns:
            Logits of shape (batch, time_steps, num_classes)
        """
        # CNN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_freq1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool_freq2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool_freq3(x)
        # x: (batch, 128, out_freq, time_steps)
        # Permute: we want the time axis to be the second axis – (batch, time_steps, channels, out_freq)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, freq = x.size()
        # Flatten features for each frame
        x = x.contiguous().view(batch_size, time_steps, channels * freq)  # (batch, time_steps, 128*out_freq)
        
        # Projection to d_model dimension
        x = self.fc_proj(x)  # (batch, time_steps, d_model)
        x = self.pos_encoder(x)
        # Transformer – process the sequence
        x = self.transformer(x)  # (batch, time_steps, d_model)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (batch, time_steps, num_classes)
        return logits

#########################################
# 4. Training function
#########################################

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_frames = 0
        correct_frames = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)   # shape: (B, 1, n_mels, time_steps)
            labels = labels.to(device)   # shape: (B, time_steps)
            optimizer.zero_grad()
            outputs = model(inputs)      # shape: (B, time_steps, num_classes)
            
            # Calculate loss: reshape predictions and labels to shape (B*time_steps, num_classes)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate frame accuracy
            _, predicted = torch.max(outputs, dim=-1)  # (B, time_steps)
            total_frames += labels.numel()
            correct_frames += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_frames / total_frames
        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.4f} | Frame Acc: {epoch_acc:.4f}")
        
        # Walidacja
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, dim=-1)
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {val_loss:.4f} | Frame Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_breath_seq_transformer_model_CURR_BEST.pth")
            print("Saved the best model.")
    
    print("Training completed.")

#########################################
# 5. Inference function
#########################################

def infer(model, input_tensor, device):
    """
    Args:
        model (nn.Module): Trained model.
        input_tensor (torch.Tensor): Single spectrogram with dimensions (1, n_mels, time_steps)
        device: Device ("cuda" or "cpu")
    Returns:
        predicted_seq (numpy.ndarray): Predicted labels for each frame (shape: (time_steps,))
    """
    model.eval()
    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # (1, 1, n_mels, time_steps)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)  # (1, time_steps, num_classes)
        _, predicted = torch.max(outputs, dim=-1)  # (1, time_steps)
    return predicted.squeeze(0).cpu().numpy()

#########################################
# 6. Example usage
#########################################

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 4   # Value chosen as an example – depends on the amount of data and GPU/CPU power
    learning_rate = 1e-3

    # Path to the folder with sequential data
    data_dir = "../../sequences"
    
    # Create dataset and DataLoader (example split into training and validation)
    full_dataset = BreathSeqDataset(data_dir, sample_rate=44100, n_mels=40, n_fft=1024, hop_length=512)
    
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(0.8 * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = BreathPhaseTransformerSeq(n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
    
    # Example inference: load one file and display predictions for each frame
    example_audio = full_dataset.audio_files[0]
    mel_spec, true_labels = full_dataset[0]
    predicted_seq = infer(model, mel_spec, device)
    
    # Updated label dictionary according to the new map
    label_names = {0: "exhale", 1: "inhale", 2: "silence"}
    print("Predictions for each frame:")
    print(predicted_seq)
    print("Predictions (text):", [label_names[label] for label in predicted_seq])