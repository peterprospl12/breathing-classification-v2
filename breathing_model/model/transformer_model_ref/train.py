import torch
import os


def run_train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_frames = 0
    correct_frames = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=-1)
        total_frames += labels.numel()
        correct_frames += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_frames / total_frames
    return train_loss, train_acc


def run_val_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=-1)
            val_total += labels.numel()
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    return avg_val_loss, val_acc


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
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = run_val_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

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

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device)
