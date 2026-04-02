from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_extraction import load_psd_features
from utils import PROJECT_ROOT


@dataclass
class EEGNetConfig:
    f1: int = 8
    d: int = 2
    kernel_length: int = 16
    dropout: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 8
    epochs: int = 20
    test_size: float = 0.2
    random_state: int = 42


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, optim, DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for EEGNet baseline. Install with: pip install torch"
        ) from exc


def load_subject_level_psd(summary_csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    summary_csv_path = Path(summary_csv_path)
    if not summary_csv_path.is_absolute():
        summary_csv_path = PROJECT_ROOT / summary_csv_path
    summary_df = pd.read_csv(summary_csv_path)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in summary_df.itertuples(index=False):
        npz_path = Path(row.output_file)
        if not npz_path.is_absolute():
            npz_path = PROJECT_ROOT / npz_path
        data = load_psd_features(npz_path)
        # Subject-level representation: mean PSD over retained epochs -> (channels, freq_bins)
        subject_psd = np.mean(data["psd_features"], axis=0, dtype=np.float32)
        x_list.append(subject_psd)
        y_list.append(int(row.label))

    x = np.stack(x_list, axis=0).astype(np.float32)  # (n_subjects, n_channels, n_freq_bins)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y


def build_eegnet(n_channels: int, n_freq_bins: int, n_classes: int, cfg: EEGNetConfig):
    torch, nn, *_ = _import_torch()
    f2 = cfg.f1 * cfg.d

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, cfg.f1, kernel_size=(1, cfg.kernel_length), padding=(0, cfg.kernel_length // 2), bias=False),
                nn.BatchNorm2d(cfg.f1),
                # depthwise spatial convolution
                nn.Conv2d(cfg.f1, f2, kernel_size=(n_channels, 1), groups=cfg.f1, bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2)),
                nn.Dropout(cfg.dropout),
            )
            self.block2 = nn.Sequential(
                # separable temporal convolution approximation
                nn.Conv2d(f2, f2, kernel_size=(1, 8), padding=(0, 4), groups=f2, bias=False),
                nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 2)),
                nn.Dropout(cfg.dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_freq_bins)
                out = self.block2(self.block1(dummy))
                flat_dim = out.view(1, -1).shape[1]
            self.classifier = nn.Linear(flat_dim, n_classes)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return EEGNet()


def train_eegnet_baseline(summary_csv_path: str | Path, cfg: EEGNetConfig | None = None) -> dict[str, float]:
    cfg = cfg or EEGNetConfig()
    torch, nn, optim, DataLoader, TensorDataset = _import_torch()

    x, y = load_subject_level_psd(summary_csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Add CNN input channel dimension: (N, 1, channels, freq_bins)
    x_train_t = torch.tensor(x_train[:, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test[:, None, :, :], dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=cfg.batch_size, shuffle=False)

    n_channels = x.shape[1]
    n_freq_bins = x.shape[2]
    n_classes = int(np.unique(y).shape[0])

    model = build_eegnet(n_channels, n_freq_bins, n_classes, cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _ in range(cfg.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    def accuracy(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                pred = model(batch_x).argmax(dim=1)
                correct += int((pred == batch_y).sum().item())
                total += int(batch_y.shape[0])
        return correct / max(total, 1)

    train_acc = accuracy(train_loader)
    test_acc = accuracy(test_loader)

    return {
        "n_subjects": float(x.shape[0]),
        "n_classes": float(n_classes),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }

