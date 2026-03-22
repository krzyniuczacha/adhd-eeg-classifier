import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from src.model import EEG_CNN


def cross_validate(model, X, y, groups, n_splits=5):
    """Cross-validation z grupami pacjentow (bez data leakage)"""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    fold_f1s = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold}: acc={acc:.4f}, f1={f1:.4f}")

    print(f"  Srednia acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    print(f"  Srednia f1:  {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
    return fold_accs, fold_f1s


def train_final_model(model, X_train, y_train):
    """Trenuje finalny model sklearn na danych treningowych"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def _augment_batch(X_batch):
    """Augmentacja danych EEG w trakcie treningu.

    - Gaussian noise: symuluje szum pomiarowy elektrod
    - Time shift: przesuniecie sygnalu w czasie (circular)
    - Amplitude scaling: losowa zmiana amplitudy per probka
    - Channel dropout: losowe zerowanie kanalow (symuluje uszkodzona elektrode)
    """
    batch_size, n_channels, n_samples = X_batch.shape

    # gaussian noise
    noise = torch.randn_like(X_batch) * 0.15
    X_batch = X_batch + noise

    # time shift (do +/- 10 probek)
    shift = torch.randint(-10, 11, (1,)).item()
    if shift != 0:
        X_batch = torch.roll(X_batch, shifts=shift, dims=2)

    # amplitude scaling (0.8 - 1.2x) - per probka
    scale = 0.8 + 0.4 * torch.rand(batch_size, 1, 1)
    X_batch = X_batch * scale

    # channel dropout - (p=0.1 per kanal)
    channel_mask = (torch.rand(batch_size, n_channels, 1) > 0.1).float()
    X_batch = X_batch * channel_mask

    return X_batch


def train_cnn(X_train, y_train, X_test, y_test, n_channels=19, n_samples=256,
              epochs=80, batch_size=64, lr=3e-4, patience=15):
    # Trenuje CNN na surowych sygnałach EEG z augmentacja i early stopping.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    smooth = 0.05
    y_train_smooth = y_train * (1 - 2 * smooth) + smooth

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train_smooth)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)

    X_val_t = torch.FloatTensor(X_test).to(device)
    y_val_t = torch.FloatTensor(y_test).to(device)

    model = EEG_CNN(n_channels=n_channels, n_samples=n_samples).to(device)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- trening z augmentacja ---
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = _augment_batch(X_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(epoch)

        # --- walidacja (bez augmentacji) ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze()
            val_loss = criterion(val_logits, y_val_t).item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping w epoce {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_val_t).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    return model, preds, probs, {'mean': mean, 'std': std}
