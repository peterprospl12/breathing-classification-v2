from typing import Tuple

import torch
import os

from torch import nn, optim
from torch.utils.data import DataLoader

from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq
from breathing_model.model.transformer.utils import BreathType, split_dataset, load_yaml

PAD_LABEL = BreathType.SILENCE
IGNORE_INDEX = -100    # value to give to CrossEntropyLoss for ignored positions

def run_train_epoch(model: nn.Module,
                    data_loader: DataLoader,
                    loss_function: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device):
    model.train()

    total_loss_weighted = 0.0
    total_valid_frames = 0
    total_correct_predictions = 0

    for batch_index, (spectograms_batch, labels_batch, padding_mask_batch) in enumerate(data_loader):
        spectograms_batch = spectograms_batch.to(device)
        labels_batch = labels_batch.to(device)
        padding_mask_batch = padding_mask_batch.to(device)

        labels_for_loss = labels_batch.clone()
        labels_for_loss[padding_mask_batch] = IGNORE_INDEX  # ignored by CrossEntropyLoss

        optimizer.zero_grad()

        # Forward pass: pass src_key_padding_mask (True = padded positions)
        outputs = model(spectograms_batch, src_key_padding_mask=padding_mask_batch) # [batch_size, time_frames, num_classes]

        # Compute loss: CrossEntropyLoss expects [N, C] logits and [N] targets
        logits_flat = outputs.view(-1, outputs.size(-1))  # [B*T, num_classes]
        targets_flat = labels_for_loss.view(-1)  # [B*T]

        batch_loss = loss_function(logits_flat, targets_flat)

        # Count valid frames in this batch to weight loss averaging correctly
        valid_frames_in_batch = (~padding_mask_batch).sum().item()  # number of frames with real labels
        # If there are no valid frames in this batch (all padding) skip metric counting but still propagate loss
        # Note: loss_function will return 0 if there are no valid elements; guard divide by zero below.
        batch_loss_value = batch_loss.item()

        # Backprop and optimizer step
        batch_loss.backward()
        optimizer.step()

        # Accumulate weighted loss
        total_loss_weighted += batch_loss_value * valid_frames_in_batch
        total_valid_frames += valid_frames_in_batch

        # Compute predictions and accumulate correct count for valid frames
        predicted_labels = torch.argmax(outputs, dim=-1)  # [B, T]
        # valid_frame_mask: True where we HAVE real labels (not pad)
        valid_frame_mask = ~padding_mask_batch
        correct_predictions_in_batch = ((predicted_labels == labels_batch) & valid_frame_mask).sum().item()
        total_correct_predictions += correct_predictions_in_batch

    average_loss = (total_loss_weighted / total_valid_frames) if total_valid_frames > 0 else 0.0
    accuracy = (total_correct_predictions / total_valid_frames) if total_valid_frames > 0 else 0.0

    return average_loss, accuracy

def run_validation_epoch(model: nn.Module,
                         data_loader: DataLoader,
                         loss_function: nn.Module,
                         device: torch.device) -> Tuple[float, float]:
    """
    Runs validation epoch (no grad). Returns average loss per valid frame and accuracy on valid frames.
    """
    model.eval()

    total_loss_weighted = 0.0
    total_valid_frames = 0
    total_correct_predictions = 0

    with torch.no_grad():
        for spectrograms_batch, labels_batch, padding_mask_batch in data_loader:
            spectrograms_batch = spectrograms_batch.to(device)
            labels_batch = labels_batch.to(device)
            padding_mask_batch = padding_mask_batch.to(device)

            labels_for_loss = labels_batch.clone()
            labels_for_loss[padding_mask_batch] = IGNORE_INDEX

            outputs = model(spectrograms_batch, src_key_padding_mask=padding_mask_batch)  # [B, T, num_classes]

            logits_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = labels_for_loss.view(-1)

            batch_loss = loss_function(logits_flat, targets_flat)
            valid_frames_in_batch = (~padding_mask_batch).sum().item()
            total_loss_weighted += batch_loss.item() * valid_frames_in_batch
            total_valid_frames += valid_frames_in_batch

            predicted_labels = torch.argmax(outputs, dim=-1)
            valid_frame_mask = ~padding_mask_batch
            total_correct_predictions += ((predicted_labels == labels_batch) & valid_frame_mask).sum().item()

    average_loss = (total_loss_weighted / total_valid_frames) if total_valid_frames > 0 else 0.0
    accuracy = (total_correct_predictions / total_valid_frames) if total_valid_frames > 0 else 0.0

    return average_loss, accuracy


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                num_epochs: int,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler,
                save_directory: str,
                patience: int = 6) -> None:
    """
    Full training loop with early stopping based on validation loss.
    """
    os.makedirs(save_directory, exist_ok=True)

    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = run_train_epoch(model, train_loader, loss_function, optimizer, device)
        print(f"Epoch {epoch} / {num_epochs} - Train Loss: {train_loss:.6f} | Train Acc: {train_accuracy:.4f}")

        val_loss, val_accuracy = run_validation_epoch(model, val_loader, loss_function, device)
        print(f"Epoch {epoch} / {num_epochs} - Val   Loss: {val_loss:.6f} | Val   Acc: {val_accuracy:.4f}")

        # Step scheduler (if provided)
        if scheduler is not None:
            scheduler.step()

        # Checkpointing based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_path = os.path.join(save_directory, f"best_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")


if __name__ == "__main__":
    # Load config (YAML)
    config = load_yaml("./config.yaml")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {'cuda' if device.type == 'cuda' else 'cpu'}")

    # Create dataset and split
    dataset = BreathDataset(
        data_dir=config['data']['data_dir'],
        label_dir=config['data']['label_dir'],
        sample_rate=config['data'].get('sample_rate', 44100),
        n_mels=config['model'].get('n_mels', 128),
        n_fft=config['data'].get('n_fft', 2048),
        hop_length=config['data'].get('hop_length', 512),
    )

    train_dataset, val_dataset = split_dataset(dataset)

    # DataLoader
    batch_size = config['train'].get('batch_size', 8)
    num_workers = config['train'].get('num_workers', 4)
    pin_memory_flag = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              drop_last=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory_flag)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory_flag)

    # Model instantiation: force 2 classes (exhale=0, inhale=1)
    model = BreathPhaseTransformerSeq(
        n_mels=config['model'].get('n_mels', 128),
        d_model=config['model'].get('d_model', 192),
        nhead=config['model'].get('nhead', 8),
        num_layers=config['model'].get('num_layers', 6),
        num_classes=config['model'].get('num_classes', 3)
    ).to(device)

    # Optimizer & scheduler
    learning_rate = config['train'].get('learning_rate', 1e-3)
    weight_decay = config['train'].get('weight_decay', 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_step_size = config['train'].get('scheduler_step_size', 10)
    scheduler_gamma = config['train'].get('scheduler_gamma', 0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Train
    num_epochs = config['train'].get('num_epochs', 50)
    patience = config['train'].get('patience', 6)
    save_directory = config['train'].get('save_dir', 'checkpoints')

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=num_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                save_directory=save_directory,
                patience=patience)
