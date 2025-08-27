import torch
import os
import numpy as np

def calculate_class_weights(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.flatten().tolist())

    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


def calculate_binary_metrics(outputs, targets):
    _, predicted = torch.max(outputs, dim=-1)

    # Flatten tensors
    predicted_flat = predicted.view(-1)
    targets_flat = targets.view(-1)

    # Basic accuracy
    total = targets_flat.numel()
    correct = (predicted_flat == targets_flat).sum().item()
    accuracy = correct / total if total > 0 else 0

    # Binary classification metrics
    tp = ((predicted_flat == 1) & (targets_flat == 1)).sum().item()
    fp = ((predicted_flat == 1) & (targets_flat == 0)).sum().item()
    fn = ((predicted_flat == 0) & (targets_flat == 1)).sum().item()
    tn = ((predicted_flat == 0) & (targets_flat == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, tp, fp, fn, tn


def run_train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate binary metrics
        accuracy, precision, recall, f1, _, _, _, _ = calculate_binary_metrics(outputs, labels)
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    train_loss = running_loss / len(train_loader.dataset)
    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    return train_loss, avg_accuracy, avg_precision, avg_recall, avg_f1


def run_val_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            val_loss += loss.item() * inputs.size(0)

            # Calculate binary metrics
            accuracy, precision, recall, f1, _, _, _, _ = calculate_binary_metrics(outputs, labels)
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    return avg_val_loss, avg_accuracy, avg_precision, avg_recall, avg_f1


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs=25,
                patience=6,
                save_dir="checkpoints"):
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = run_train_epoch(model, train_loader, criterion,
                                                                                 optimizer, device)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1 = run_val_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, f"best_model_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")


if __name__ == '__main__':
    from utils import load_yaml, split_dataset
    from dataset import BreathDataset, collate_fn
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from model import BreathPhaseTransformerSeq

    config_path = "./config.yaml"
    config = load_yaml(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {"gpu" if torch.cuda.is_available() else "cpu"}")

    dataset = BreathDataset(config['data']['data_dir'], config['data']['label_dir'])
    data_train, data_val = split_dataset(dataset)

    train_loader = DataLoader(data_train,
                              batch_size=config['train']['batch_size'],
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(data_val,
                            batch_size=config['train']['batch_size'],
                            collate_fn=collate_fn,
                            shuffle=False,
                            drop_last=False,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available())

    model = BreathPhaseTransformerSeq(
        n_mels=config['model']['n_mels'],
        num_classes=config['model']['num_classes'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers']
    )

    class_weights = calculate_class_weights(data_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device)
