"""Lightweight forecasting utilities (1D-CNN and RNN) for per-cell short-term prediction.

Design notes
- The module defers importing `torch` until training/prediction time so that
  unit tests and environments without PyTorch can still import the package.
- Training data is built from the CSV produced by `process_video()` which has
  rows: frame_idx,time_s,cell_idx,density,acceleration,entropy,score,category
- Each training sample is a sliding window (seq_len) of per-cell features
  `[density,acceleration,entropy]` and the supervised target is the average
  `density` over the forecast horizon (in seconds) for the same cell.
"""

from __future__ import annotations

import csv
import math
import os
from typing import List, Tuple

import numpy as np


def _read_csv_per_cell(csv_path: str):
    """Read CSV into a dict mapping (cell_idx) -> list of rows sorted by frame_idx.

    Returns mapping cell_idx -> list of dict rows.
    """
    cells = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                cell = int(r["cell_idx"])
                frame = int(r["frame_idx"])
                density = float(r.get("density", 0.0))
                acc = float(r.get("acceleration", 0.0))
                entropy = float(r.get("entropy", 0.0))
            except Exception:
                continue

            cells.setdefault(cell, []).append({
                "frame_idx": frame,
                "density": density,
                "acceleration": acc,
                "entropy": entropy,
            })

    # sort each list by frame_idx
    for k in list(cells.keys()):
        cells[k].sort(key=lambda x: x["frame_idx"])

    return cells


def build_sequences_from_csv(
    csv_path: str,
    seq_len: int,
    horizon_sec: float,
    fps: float,
    sample_rate: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build X,y arrays from CSV.

    X shape: (N, seq_len, 3) features=[density,acceleration,entropy]
    y shape: (N,) target = future average density across horizon frames
    """
    cells = _read_csv_per_cell(csv_path)
    horizon_frames = max(1, int(round(horizon_sec * fps)))

    X_list = []
    y_list = []

    for cell_idx, rows in cells.items():
        n = len(rows)
        # Build arrays for features and density
        feats = np.zeros((n, 3), dtype=np.float32)
        dens = np.zeros((n,), dtype=np.float32)
        frames = [r["frame_idx"] for r in rows]
        for i, r in enumerate(rows):
            feats[i, 0] = r["density"]
            feats[i, 1] = r["acceleration"]
            feats[i, 2] = r["entropy"]
            dens[i] = r["density"]

        # sliding windows
        for start in range(0, n - seq_len - horizon_frames + 1, sample_rate):
            end = start + seq_len
            future_start = end
            future_end = end + horizon_frames
            if future_end > n:
                break
            x = feats[start:end]
            y = float(dens[future_start:future_end].mean())
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        return np.zeros((0, seq_len, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _ensure_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.utils.data as data
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for training/prediction: %s" % e)

    return torch, nn, data


def train_forecast(
    csv_path: str,
    model_type: str = "conv1d",
    seq_len: int = 16,
    horizon_sec: float = 10.0,
    fps: float = 10.0,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    sample_rate: int = 1,
    out_path: str | None = None,
):
    """Train a forecast model (1D-CNN or RNN) and save weights.

    Returns path to saved model.
    """
    torch, nn, data = _ensure_torch()

    X, y = build_sequences_from_csv(csv_path, seq_len, horizon_sec, fps, sample_rate)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples found in CSV: %s" % csv_path)

    # train/val split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    class NumpyDataset(data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size)

    in_channels = X.shape[2]
    seq_len_local = X.shape[1]

    if model_type == "conv1d":
        class ConvModel(nn.Module):
            def __init__(self, in_ch, seq_len):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

            def forward(self, x):
                # x: (B, seq_len, features) -> conv expects (B, C, L)
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = self.fc(x)
                return x.squeeze(-1)

        model = ConvModel(in_channels, seq_len_local)

    elif model_type in ("lstm", "gru"):
        rnn_type = model_type

        class RNNModel(nn.Module):
            def __init__(self, in_ch, hidden=64, rtype="lstm"):
                super().__init__()
                if rtype == "lstm":
                    self.rnn = nn.LSTM(input_size=in_ch, hidden_size=hidden, batch_first=True)
                else:
                    self.rnn = nn.GRU(input_size=in_ch, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                # take last timestep
                last = out[:, -1, :]
                return self.fc(last).squeeze(-1)

        model = RNNModel(in_channels, hidden=64, rtype=rnn_type)

    else:
        raise ValueError("Unknown model_type: %s" % model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_val:
            best_val = val_loss
            # save best
            if out_path is None:
                out_path = os.path.join("models", f"forecast_{model_type}.pt")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict(), "type": model_type}, out_path)

        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    return out_path


def predict_from_csv_last_sequence(
    csv_path: str,
    model_path: str,
    cell_idx: int,
    seq_len: int = 16,
    fps: float = 10.0,
):
    """Load model and predict the next horizon average density for `cell_idx` using
    the last `seq_len` frames available in the CSV for that cell.
    Returns a float prediction.
    """
    torch, nn, data = _ensure_torch()

    cells = _read_csv_per_cell(csv_path)
    if cell_idx not in cells:
        raise RuntimeError("Cell %d not found in CSV" % cell_idx)

    rows = cells[cell_idx]
    if len(rows) < seq_len:
        raise RuntimeError("Not enough frames for cell %d: need %d got %d" % (cell_idx, seq_len, len(rows)))

    last = rows[-seq_len:]
    x = np.zeros((1, seq_len, 3), dtype=np.float32)
    for i, r in enumerate(last):
        x[0, i, 0] = r["density"]
        x[0, i, 1] = r["acceleration"]
        x[0, i, 2] = r["entropy"]

    # load model
    state = torch.load(model_path, map_location="cpu")
    model_type = state.get("type", "conv1d")

    if model_type == "conv1d":
        class ConvModel(nn.Module):
            def __init__(self, in_ch):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(in_ch, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 1))

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = self.fc(x)
                return x.squeeze(-1)

        model = ConvModel(in_ch=3)

    elif model_type in ("lstm", "gru"):
        rtype = model_type

        class RNNModel(nn.Module):
            def __init__(self, in_ch, hidden=64, rtype="lstm"):
                super().__init__()
                if rtype == "lstm":
                    self.rnn = nn.LSTM(input_size=in_ch, hidden_size=hidden, batch_first=True)
                else:
                    self.rnn = nn.GRU(input_size=in_ch, hidden_size=hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                last = out[:, -1, :]
                return self.fc(last).squeeze(-1)

        model = RNNModel(in_ch=3, rtype=rtype)

    else:
        raise RuntimeError("Unknown saved model type: %s" % model_type)

    model.load_state_dict(state["model_state"])
    model.eval()

    with torch.no_grad():
        tx = torch.from_numpy(x).float()
        pred = model(tx)
        return float(pred.item())
