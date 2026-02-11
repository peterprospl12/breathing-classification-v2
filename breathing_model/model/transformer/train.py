import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import os
import copy

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR

import torchaudio.transforms as T

from breathing_model.model.transformer.dataset import BreathDataset, collate_fn
from breathing_model.model.transformer.model import BreathPhaseTransformerSeq
from breathing_model.model.transformer.utils import BreathType, split_dataset, load_yaml

IGNORE_INDEX = -100    # value to give to CrossEntropyLoss for ignored positions


class FocalLoss(nn.Module):
    """
    Focal Loss: focuses training on hard-to-classify frames.
    Down-weights easy examples (high confidence), up-weights hard ones.
    Particularly useful for breathing transitions where model is uncertain.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights [num_classes]
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        # Standard cross-entropy with label smoothing
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)

        # Compute pt (probability of correct class)
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma

        # Optional per-class weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss
        return loss.mean()


def mixup_data(spectrograms: torch.Tensor, labels: torch.Tensor,
               padding_mask: torch.Tensor, alpha: float = 0.3
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation: creates convex combinations of training pairs.
    Returns mixed spectrograms, two label sets, padding mask, and lambda.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = spectrograms.size(0)
    index = torch.randperm(batch_size, device=spectrograms.device)

    mixed_spec = lam * spectrograms + (1 - lam) * spectrograms[index]
    labels_a = labels
    labels_b = labels[index]
    # Use union of padding masks (conservative: mark as padded if EITHER is padded)
    mixed_mask = padding_mask | padding_mask[index]

    return mixed_spec, labels_a, labels_b, mixed_mask, lam

def run_train_epoch(model: nn.Module,
                    data_loader: DataLoader,
                    loss_function: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    scheduler,
                    gradient_clip_norm: float = 0.0,
                    mixup_alpha: float = 0.0):
    model.train()

    total_loss_weighted = 0.0
    total_valid_frames = 0
    total_correct_predictions = 0
    use_mixup = mixup_alpha > 0

    for batch_index, (spectrograms_batch, labels_batch, padding_mask_batch) in enumerate(data_loader):
        spectrograms_batch = spectrograms_batch.to(device)
        labels_batch = labels_batch.to(device)
        padding_mask_batch = padding_mask_batch.to(device)

        optimizer.zero_grad()

        if use_mixup:
            mixed_spec, labels_a, labels_b, mixed_mask, lam = mixup_data(
                spectrograms_batch, labels_batch, padding_mask_batch, mixup_alpha
            )
            labels_a_loss = labels_a.clone()
            labels_b_loss = labels_b.clone()
            labels_a_loss[mixed_mask] = IGNORE_INDEX
            labels_b_loss[mixed_mask] = IGNORE_INDEX

            outputs = model(mixed_spec, src_key_padding_mask=mixed_mask)

            logits_flat = outputs.view(-1, outputs.size(-1))
            targets_a_flat = labels_a_loss.view(-1)
            targets_b_flat = labels_b_loss.view(-1)

            batch_loss = lam * loss_function(logits_flat, targets_a_flat) + \
                         (1 - lam) * loss_function(logits_flat, targets_b_flat)

            valid_frames_in_batch = (~mixed_mask).sum().item()

            # For accuracy, use original (non-mixed) data to get meaningful metric
            with torch.no_grad():
                orig_labels_for_loss = labels_batch.clone()
                orig_labels_for_loss[padding_mask_batch] = IGNORE_INDEX
                orig_outputs = model(spectrograms_batch, src_key_padding_mask=padding_mask_batch)
                predicted_labels = torch.argmax(orig_outputs, dim=-1)
                valid_frame_mask = ~padding_mask_batch
                correct_predictions_in_batch = ((predicted_labels == labels_batch) & valid_frame_mask).sum().item()
                valid_frames_for_acc = valid_frame_mask.sum().item()
        else:
            labels_for_loss = labels_batch.clone()
            labels_for_loss[padding_mask_batch] = IGNORE_INDEX

            outputs = model(spectrograms_batch, src_key_padding_mask=padding_mask_batch)

            logits_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = labels_for_loss.view(-1)

            batch_loss = loss_function(logits_flat, targets_flat)

            valid_frames_in_batch = (~padding_mask_batch).sum().item()
            valid_frames_for_acc = valid_frames_in_batch

            predicted_labels = torch.argmax(outputs, dim=-1)
            valid_frame_mask = ~padding_mask_batch
            correct_predictions_in_batch = ((predicted_labels == labels_batch) & valid_frame_mask).sum().item()

        batch_loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss_weighted += batch_loss.item() * valid_frames_in_batch
        total_valid_frames += valid_frames_in_batch
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
                scheduler,
                save_directory: str,
                patience: int = 6,
                label_smoothing: float = 0.0,
                gradient_clip_norm: float = 0.0,
                resume_from: Optional[str] = None,
                loss_type: str = 'cross_entropy',
                focal_gamma: float = 2.0,
                mixup_alpha: float = 0.0,
                swa_start_epoch: int = 0,
                swa_lr: float = 1e-4) -> None:
    """
    Full training loop with early stopping based on validation loss.
    Supports: resume, focal loss, mixup, SWA.
    """
    os.makedirs(save_directory, exist_ok=True)

    # --- Loss function ---
    if loss_type == 'focal':
        loss_function = FocalLoss(gamma=focal_gamma, ignore_index=IGNORE_INDEX,
                                  label_smoothing=label_smoothing)
        print(f"Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
        print(f"Using CrossEntropy Loss (label_smoothing={label_smoothing})")

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    start_epoch = 1

    # --- SWA setup ---
    use_swa = swa_start_epoch > 0
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"SWA enabled: starts at epoch {swa_start_epoch}, swa_lr={swa_lr}")

    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from checkpoint: {resume_from} (epoch {checkpoint['epoch']}, val_loss={best_val_loss:.6f})")

    for epoch in range(start_epoch, num_epochs + 1):
        in_swa_phase = use_swa and epoch >= swa_start_epoch

        train_loss, train_accuracy = run_train_epoch(
            model, train_loader, loss_function, optimizer, device, scheduler,
            gradient_clip_norm, mixup_alpha=mixup_alpha
        )
        print(f"Epoch {epoch} / {num_epochs} - Train Loss: {train_loss:.6f} | Train Acc: {train_accuracy:.4f}")

        val_loss, val_accuracy = run_validation_epoch(model, val_loader, loss_function, device)
        print(f"Epoch {epoch} / {num_epochs} - Val   Loss: {val_loss:.6f} | Val   Acc: {val_accuracy:.4f}")

        # Step scheduler (if provided and not in SWA phase)
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  SWA: updated averaged model (lr={optimizer.param_groups[0]['lr']:.6f})")
        elif scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        # Checkpointing based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_path = os.path.join(save_directory, f"best_model_epoch_{epoch}.pth")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, save_path)
            print(f"Saved best model to {save_path}")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break

    # --- Finalize SWA ---
    if use_swa and swa_model is not None:
        print("Finalizing SWA: updating batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Evaluate SWA model
        swa_val_loss, swa_val_acc = run_validation_epoch(swa_model, val_loader, loss_function, device)
        print(f"SWA Model - Val Loss: {swa_val_loss:.6f} | Val Acc: {swa_val_acc:.4f}")

        swa_save_path = os.path.join(save_directory, "swa_model.pth")
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'val_loss': swa_val_loss,
        }, swa_save_path)
        print(f"Saved SWA model to {swa_save_path}")

    print("Training finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pth file to resume training from')
    args = parser.parse_args()

    # === Load config ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_yaml(os.path.join(script_dir, "config.yaml"))

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === SpecAugment transforms (applied to mel spectrogram during training) ===
    augment_cfg = config['augment']
    spec_transforms = []
    if augment_cfg.get('enabled', True):
        for _ in range(augment_cfg.get('num_freq_masks', 2)):
            spec_transforms.append(T.FrequencyMasking(
                freq_mask_param=augment_cfg.get('freq_mask_param', 15)
            ))
        for _ in range(augment_cfg.get('num_time_masks', 2)):
            spec_transforms.append(T.TimeMasking(
                time_mask_param=augment_cfg.get('time_mask_param', 25)
            ))

    # === Datasets (separate augmentation for train vs. val) ===
    # Resolve relative data paths against the script directory
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(script_dir, p))

    common_data_kwargs = dict(
        data_dir=resolve_path(config['data']['data_dir']),
        label_dir=resolve_path(config['data']['label_dir']),
        sample_rate=config['data']['sample_rate'],
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
    )

    train_full_dataset = BreathDataset(
        **common_data_kwargs,
        augment=augment_cfg.get('enabled', True),
        transforms=spec_transforms if spec_transforms else None,
        p_noise=augment_cfg.get('p_noise', 0.3),
        p_volume=augment_cfg.get('p_volume', 0.3),
        p_shift=augment_cfg.get('p_shift', 0.2),
        volume_range=tuple(augment_cfg.get('volume_range', [0.3, 1.0])),
        noise_factor_range=tuple(augment_cfg.get('noise_factor_range', [1e-5, 1e-3])),
        max_shift_seconds=augment_cfg.get('max_shift_seconds', 0.3),
        seed=augment_cfg.get('seed', 123456),
    )

    val_full_dataset = BreathDataset(
        **common_data_kwargs,
        augment=False,
    )

    # Reproducible train/val split by indices
    n_total = len(train_full_dataset)
    n_train = int(n_total * 0.8)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    print(f"Dataset: {n_total} total | {len(train_dataset)} train | {len(val_dataset)} val")

    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        persistent_workers=True,
    )

    # === Model ===
    model_cfg = config['model']
    model = BreathPhaseTransformerSeq(
        n_mels=model_cfg['n_mels'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_layers'],
        num_classes=model_cfg['num_classes'],
        dim_feedforward=model_cfg.get('dim_feedforward', 1024),
        dropout=model_cfg.get('dropout', 0.15),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # === Optimizer (AdamW for proper weight decay) ===
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
    )

    # === Scheduler ===
    scheduler_cfg = config['scheduler']
    scheduler_type = scheduler_cfg['type'].lower()

    if scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_cfg['max_lr'],
            epochs=config['train']['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_cfg['pct_start'],
            anneal_strategy=scheduler_cfg['anneal_strategy'],
            div_factor=scheduler_cfg['div_factor'],
            final_div_factor=scheduler_cfg['final_div_factor'],
        )
    elif scheduler_type == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg['step_size'],
            gamma=scheduler_cfg['gamma']
        )
    else:
        scheduler = None

    # === Training ===
    train_cfg = config['train']
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=train_cfg['num_epochs'],
        optimizer=optimizer,
        scheduler=scheduler,
        save_directory=resolve_path(train_cfg['save_dir']),
        patience=train_cfg['patience'],
        label_smoothing=train_cfg.get('label_smoothing', 0.0),
        gradient_clip_norm=train_cfg.get('gradient_clip_norm', 0.0),
        resume_from=args.resume,
        loss_type=train_cfg.get('loss_type', 'cross_entropy'),
        focal_gamma=train_cfg.get('focal_gamma', 2.0),
        mixup_alpha=train_cfg.get('mixup_alpha', 0.0),
        swa_start_epoch=train_cfg.get('swa_start_epoch', 0),
        swa_lr=train_cfg.get('swa_lr', 1e-4),
    )


if __name__ == "__main__":
    main()